"""
EV Registration Forecasting — Logistic S-Curve Model
=====================================================
Fits a logistic growth curve to historical EV registration data by state
and projects forward under three carrying-capacity (K) scenarios.
 
Inputs:
    ev_registrations.csv          — required; rows=states, columns=years + "State"
                                    (e.g. the AFDC export: State,2016,2017,...,2024)
    total_vehicles.csv            — optional; same shape, total light-duty vehicles
                                    per state. Used to compute K = fraction × total.
                                    If absent, falls back to built-in 2024 estimates.
 
Usage:
    1. Set EV_CSV_PATH (and optionally TOTAL_CSV_PATH) near the top of main().
    2. pip install numpy scipy pandas matplotlib
    3. python ev_forecast.py
 
Outputs:
    ev_forecast_results.csv              — fitted params + forecasts per state/scenario
    ev_forecast_{scenario}_2050.csv      — one per scenario; State x year (2016-2050),
                                           historical values then fitted/forecast values;
                                           same shape as the input CSV
    ev_forecast_plots/                   — per-state charts (controlled by PLOT_STATES)
    ev_forecast_national.png             — national aggregate chart
"""
 
import os
import warnings
 
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning, curve_fit
 
 
# ── Configuration ──────────────────────────────────────────────────────────────
 
FORECAST_YEARS = list(range(2025, 2051))  # through 2050
 
# Carrying capacity (K) as a fraction of total registered vehicles.
# Literature range: ~0.30 (conservative) to ~0.75 (aggressive).
K_SCENARIOS = {
    "conservative": 0.35,
    "mid":          0.55,
    "aggressive":   0.75,
}
 
# State-level plots: list of state names, True for all, False to skip.
PLOT_STATES = ["California", "Texas", "Florida", "New York", "Washington"]
PLOT_NATIONAL = True
 
# State choropleth maps: years and shapefile path
MAP_YEARS     = [2031, 2040, 2050]
SHAPEFILE_PATH = "cb_2021_us_county_20m.shp"  # same file used in county_charger_weights.py
PLOT_MAPS     = True
 
OUTPUT_DIR = "ev_forecast_plots"
 
# Fallback total light-duty vehicle registrations per state (2024, AFDC).
# Only used when --total CSV is not supplied.
TOTAL_VEHICLES_FALLBACK = {
    "Alabama": 4_884_400, "Alaska": 570_600, "Arizona": 6_587_200,
    "Arkansas": 2_736_700, "California": 37_421_700, "Colorado": 5_497_100,
    "Connecticut": 3_023_700, "Delaware": 919_000, "District of Columbia": 308_100,
    "Florida": 18_741_500, "Georgia": 9_702_400, "Hawaii": 1_081_100,
    "Idaho": 2_019_900, "Illinois": 10_124_700, "Indiana": 6_214_400,
    "Iowa": 3_178_900, "Kansas": 2_652_900, "Kentucky": 3_989_200,
    "Louisiana": 3_781_400, "Maine": 1_247_000, "Maryland": 5_031_000,
    "Massachusetts": 5_540_700, "Michigan": 8_581_600, "Minnesota": 5_185_400,
    "Mississippi": 2_725_900, "Missouri": 5_726_700, "Montana": 1_038_800,
    "Nebraska": 1_986_400, "Nevada": 2_607_600, "New Hampshire": 1_400_600,
    "New Jersey": 7_426_300, "New Mexico": 1_955_900, "New York": 11_328_500,
    "North Carolina": 9_180_700, "North Dakota": 806_900, "Ohio": 10_390_200,
    "Oklahoma": 4_242_600, "Oregon": 3_850_800, "Pennsylvania": 10_245_600,
    "Rhode Island": 872_800, "South Carolina": 5_114_000, "South Dakota": 979_900,
    "Tennessee": 6_599_000, "Texas": 26_154_400, "Utah": 3_140_100,
    "Vermont": 587_200, "Virginia": 7_816_800, "Washington": 6_830_800,
    "West Virginia": 1_520_900, "Wisconsin": 5_569_800, "Wyoming": 667_200,
}
 
 
# ── Data loading ────────────────────────────────────────────────────────────────
 
def load_registrations(path: str):
    """
    Load a CSV with a 'State' column and year columns (e.g. 2016..2024).
    Returns:
        ev_data   — {state: [count_year0, count_year1, ...]}
        years     — [2016, 2017, ..., 2024]  (sorted, integer)
    Numbers may be formatted with commas (e.g. "1,234") — handled automatically.
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
 
    state_col = next(
        (c for c in df.columns if c.lower() == "state"), None
    )
    if state_col is None:
        raise ValueError(f"No 'State' column found in {path}. Columns: {list(df.columns)}")
 
    year_cols = sorted(
        [c for c in df.columns if c.isdigit() and len(c) == 4], key=int
    )
    if not year_cols:
        raise ValueError(f"No year columns (e.g. '2016') found in {path}.")
 
    years = [int(y) for y in year_cols]
    ev_data = {}
    for _, row in df.iterrows():
        state = str(row[state_col]).strip()
        counts = []
        for yc in year_cols:
            raw = str(row[yc]).replace(",", "").strip()
            counts.append(int(float(raw)) if raw not in ("", "nan") else 0)
        ev_data[state] = counts
 
    return ev_data, years
 
 
def load_total_vehicles(path: str) -> dict:
    """
    Load a total-vehicles CSV (same shape as the EV CSV).
    Returns {state: most_recent_year_count}.
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
 
    state_col = next((c for c in df.columns if c.lower() == "state"), None)
    if state_col is None:
        raise ValueError(f"No 'State' column in {path}.")
 
    year_cols = sorted(
        [c for c in df.columns if c.isdigit() and len(c) == 4], key=int
    )
    if not year_cols:
        raise ValueError(f"No year columns in {path}.")
 
    latest_col = year_cols[-1]
    totals = {}
    for _, row in df.iterrows():
        state = str(row[state_col]).strip()
        raw = str(row[latest_col]).replace(",", "").strip()
        totals[state] = int(float(raw)) if raw not in ("", "nan") else 0
 
    return totals
 
 
# ── Model ───────────────────────────────────────────────────────────────────────
 
def logistic(t, K, r, t0):
    """Standard logistic growth: N(t) = K / (1 + exp(-r * (t - t0)))"""
    return K / (1 + np.exp(-r * (t - t0)))
 
 
def fit_logistic(years: list, counts: list, K_fixed: float):
    """
    Fit logistic curve with K fixed; free parameters are r (growth rate)
    and t0 (inflection year). Returns (r, t0, success).
    """
    t = np.array(years, dtype=float)
    N = np.array(counts, dtype=float)
 
    def model(t, r, t0):
        return logistic(t, K_fixed, r, t0)
 
    t0_init = t[len(t) // 2]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            popt, _ = curve_fit(
                model, t, N,
                p0=[0.3, t0_init],
                bounds=([0.01, 2010], [2.0, 2060]),
                maxfev=10_000,
            )
        return popt[0], popt[1], True
    except Exception:
        return None, None, False
 
 
def forecast_state(state, years_hist, counts_hist, total_vehicles, scenarios, forecast_years):
    """Fit and forecast for one state across all K scenarios."""
    results = {"state": state, "scenarios": {}}
 
    for scenario_name, k_fraction in scenarios.items():
        K = k_fraction * total_vehicles
        K = max(K, counts_hist[-1] * 1.05)  # K must exceed last observed value
 
        r, t0, success = fit_logistic(years_hist, counts_hist, K)
        if not success:
            results["scenarios"][scenario_name] = None
            continue
 
        all_years = years_hist + forecast_years
        all_counts = [logistic(y, K, r, t0) for y in all_years]
        n = len(years_hist)
 
        results["scenarios"][scenario_name] = {
            "K": K,
            "r": round(r, 4),
            "t0": round(t0, 2),
            "fitted_hist": dict(zip(years_hist, [round(v) for v in all_counts[:n]])),
            "forecast":    dict(zip(forecast_years, [round(v) for v in all_counts[n:]])),
        }
 
    return results
 
 
# ── Plotting ────────────────────────────────────────────────────────────────────
 
COLORS = {"conservative": "#BA7517", "mid": "#185FA5", "aggressive": "#3B6D11"}
 
 
def _fmt_axis(ax):
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K" if x < 1e6 else f"{x/1e6:.2f}M")
    )
 
 
def plot_national(all_results, ev_data, years_hist, forecast_years):
    fig, ax = plt.subplots(figsize=(11, 6))
    all_years = years_hist + forecast_years
 
    nat_hist = [sum(ev_data[s][i] for s in ev_data) for i in range(len(years_hist))]
    ax.plot(years_hist, [v / 1e6 for v in nat_hist],
            "o-", color="#2C2C2A", linewidth=2, markersize=5,
            label="Historical (observed)", zorder=5)
 
    for scenario in ["conservative", "mid", "aggressive"]:
        fitted_nat, forecast_nat = [], []
        for y in years_hist:
            fitted_nat.append(sum(
                res["scenarios"][scenario]["fitted_hist"].get(y, 0)
                for _, _, res in all_results
                if res["scenarios"].get(scenario)
            ))
        for y in forecast_years:
            forecast_nat.append(sum(
                res["scenarios"][scenario]["forecast"].get(y, 0)
                for _, _, res in all_results
                if res["scenarios"].get(scenario)
            ))
        combined = fitted_nat + forecast_nat
        ax.plot(all_years, [v / 1e6 for v in combined],
                "--", color=COLORS[scenario], linewidth=1.5,
                label=f"{scenario.capitalize()} (K={int(K_SCENARIOS[scenario]*100)}%)")
 
    ax.axvspan(years_hist[-1] + 0.5, forecast_years[-1] + 0.5, alpha=0.05, color="gray")
    ax.axvline(years_hist[-1] + 0.5, color="#888780", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Cumulative EV registrations (millions)", fontsize=12)
    ax.set_title("U.S. EV Registration Forecast — Logistic S-Curve Model\n"
                 "Three carrying-capacity scenarios", fontsize=13)
    ax.legend(fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}M"))
    ax.set_xlim(years_hist[0] - 0.5, forecast_years[-1] + 0.5)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("ev_forecast_national.png", dpi=150)
    plt.close()
    print("Saved ev_forecast_national.png")
 
 
def plot_state(state, counts_hist, res, years_hist, forecast_years):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(years_hist, counts_hist, "o", color="#2C2C2A",
            markersize=6, label="Observed", zorder=5)
 
    for scenario in ["conservative", "mid", "aggressive"]:
        data = res["scenarios"].get(scenario)
        if not data:
            continue
        fitted   = [data["fitted_hist"].get(y, 0) for y in years_hist]
        forecast = [data["forecast"].get(y, 0) for y in forecast_years]
        ax.plot(years_hist, fitted, "--", color=COLORS[scenario],
                linewidth=1.2, alpha=0.6)
        ax.plot(forecast_years, forecast, "-", color=COLORS[scenario],
                linewidth=2,
                label=f"{scenario.capitalize()} K={int(K_SCENARIOS[scenario]*100)}%")
 
    ax.axvspan(years_hist[-1] + 0.5, forecast_years[-1] + 0.5, alpha=0.05, color="gray")
    ax.axvline(years_hist[-1] + 0.5, color="#888780", linewidth=0.8, linestyle=":")
    ax.set_title(f"{state} — EV Registration Forecast", fontsize=12)
    ax.set_xlabel("Year")
    ax.set_ylabel("Cumulative EV registrations")
    _fmt_axis(ax)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, f"{state.replace(' ', '_')}.png")
    plt.savefig(fname, dpi=130)
    plt.close()
    print(f"  Saved {fname}")
 
 
# ── State choropleth maps ────────────────────────────────────────────────────────
 
# Full state name → 2-letter abbreviation, for joining to shapefile
STATE_ABBREV = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "District of Columbia": "DC", "Florida": "FL", "Georgia": "GA", "Hawaii": "HI",
    "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI",
    "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX",
    "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
}
 
 
def plot_state_maps(all_results, shapefile_path, years=MAP_YEARS):
    """
    For each scenario, produce a 1×3 choropleth map of cumulative EV
    registrations per state for each year in `years` (contiguous US only).
    Output: ev_map_{scenario}.png
    """
    if not os.path.exists(shapefile_path):
        print(f"  Shapefile not found at {shapefile_path} — skipping state maps.")
        return
 
    print("  Building state geometries from shapefile...")
    counties = gpd.read_file(shapefile_path).to_crs("EPSG:5070")
    # Exclude non-contiguous states and territories
    non_conus = {"02", "15", "60", "66", "69", "72", "78"}
    conus_counties = counties[~counties["STATEFP"].isin(non_conus)]
    states_geo = conus_counties.dissolve(by="STUSPS").reset_index()[["STUSPS", "geometry"]]
 
    for scenario in K_SCENARIOS:
        data_rows = []
        for state, counts, res in all_results:
            sdata = res["scenarios"].get(scenario)
            if not sdata:
                continue
            abbrev = STATE_ABBREV.get(state)
            if not abbrev:
                continue
            for y in years:
                data_rows.append({
                    "STUSPS":   abbrev,
                    "year":     y,
                    "ev_count": sdata["forecast"].get(y, 0),
                })
 
        df = pd.DataFrame(data_rows)
 
        # Shared log-scale colour bounds across all three years
        all_vals = df[df["ev_count"] > 0]["ev_count"].values
        vmin = max(all_vals.min(), 1)
        vmax = all_vals.max()
        norm = LogNorm(vmin=vmin, vmax=vmax)
 
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
 
        for ax, yr in zip(axes, years):
            yr_df = df[df["year"] == yr]
            merged = states_geo.merge(
                yr_df[["STUSPS", "ev_count"]], on="STUSPS", how="left"
            )
            ax.axis("off")
            # Grey fill for states with no data
            merged.plot(ax=ax, color="#dddddd", edgecolor="white", linewidth=0.5)
            merged[merged["ev_count"] > 0].plot(
                ax=ax, column="ev_count", cmap="YlOrRd", norm=norm,
                edgecolor="white", linewidth=0.5, legend=False,
            )
            ax.set_title(str(yr), fontsize=12)
 
        # Shared colorbar
        sm = plt.cm.ScalarMappable(cmap="YlOrRd", norm=norm)
        sm.set_array([])
        cbar_ax = fig.add_axes([0.25, 0.06, 0.50, 0.03])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
        cbar.set_label("Cumulative EV registrations (log scale)", fontsize=10)
 
        fig.suptitle(
            f"Cumulative EV Registrations by State — {scenario.capitalize()} scenario",
            fontsize=13,
        )
 
        out_path = f"ev_map_{scenario}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  Saved {out_path}")
 
 
def plot_saturation_maps(all_results, shapefile_path, years=MAP_YEARS):
    """
    For each scenario, produce a 1×3 choropleth map of each state's EV fleet
    as a percentage of its carrying capacity K (0–100%), contiguous US only.
    Output: ev_map_saturation_{scenario}.png
    """
    if not os.path.exists(shapefile_path):
        return
 
    counties = gpd.read_file(shapefile_path).to_crs("EPSG:5070")
    non_conus = {"02", "15", "60", "66", "69", "72", "78"}
    states_geo = (counties[~counties["STATEFP"].isin(non_conus)]
                  .dissolve(by="STUSPS").reset_index()[["STUSPS", "geometry"]])
 
    for scenario in K_SCENARIOS:
        data_rows = []
        for state, counts, res in all_results:
            sdata = res["scenarios"].get(scenario)
            if not sdata or sdata["K"] == 0:
                continue
            abbrev = STATE_ABBREV.get(state)
            if not abbrev:
                continue
            for y in years:
                pct = sdata["forecast"].get(y, 0) / sdata["K"] * 100
                data_rows.append({"STUSPS": abbrev, "year": y, "pct_k": pct})
 
        df = pd.DataFrame(data_rows)
 
        # Linear 0–100 scale, shared across all years
        norm = plt.Normalize(vmin=0, vmax=100)
 
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
 
        for ax, yr in zip(axes, years):
            yr_df = df[df["year"] == yr]
            merged = states_geo.merge(
                yr_df[["STUSPS", "pct_k"]], on="STUSPS", how="left"
            )
            ax.axis("off")
            merged.plot(ax=ax, color="#dddddd", edgecolor="white", linewidth=0.5)
            merged[merged["pct_k"].notna()].plot(
                ax=ax, column="pct_k", cmap="YlOrRd", norm=norm,
                edgecolor="white", linewidth=0.5, legend=False,
            )
            ax.set_title(str(yr), fontsize=12)
 
        sm = plt.cm.ScalarMappable(cmap="YlOrRd", norm=norm)
        sm.set_array([])
        cbar_ax = fig.add_axes([0.25, 0.06, 0.50, 0.03])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
        cbar.set_label("EV fleet as % of carrying capacity (K)", fontsize=10)
 
        fig.suptitle(
            f"EV Fleet Saturation of Carrying Capacity — {scenario.capitalize()} scenario",
            fontsize=13,
        )
 
        out_path = f"ev_map_saturation_{scenario}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  Saved {out_path}")
 
 
# ── Main ────────────────────────────────────────────────────────────────────────
 
def main():
    # ── Input file paths — edit these to point to your files ──────────────────
    EV_CSV_PATH    = "data/ev_clean/ev_registrations_historical.csv"   # required: State + year columns
    TOTAL_CSV_PATH = None                      # optional: same shape; None to use fallback
    # ─────────────────────────────────────────────────────────────────────────
 
    # Load EV data
    print(f"Loading EV registrations from: {EV_CSV_PATH}")
    ev_data, years_hist = load_registrations(EV_CSV_PATH)
    print(f"  {len(ev_data)} states/regions, years {years_hist[0]}–{years_hist[-1]}")
 
    # Load or fall back to total vehicles
    if TOTAL_CSV_PATH:
        print(f"Loading total vehicles from: {TOTAL_CSV_PATH}")
        total_vehicles = load_total_vehicles(TOTAL_CSV_PATH)
    else:
        print("No TOTAL_CSV_PATH set; using built-in 2024 estimates.")
        total_vehicles = TOTAL_VEHICLES_FALLBACK
 
    # Fit and forecast per state
    all_results = []
    for state, counts in ev_data.items():
        tv = total_vehicles.get(state, 2_000_000)
        if tv == 0:
            tv = 2_000_000
        res = forecast_state(state, years_hist, counts, tv, K_SCENARIOS, FORECAST_YEARS)
        all_results.append((state, counts, res))
 
    # Output CSV
    rows = []
    for state, counts, res in all_results:
        for scenario, data in res["scenarios"].items():
            if data is None:
                continue
            row = {"state": state, "scenario": scenario,
                   "K": round(data["K"]), "r": data["r"], "t0": data["t0"]}
            for i, y in enumerate(years_hist):
                row[f"hist_{y}"]   = counts[i]
                row[f"fitted_{y}"] = data["fitted_hist"].get(y, "")
            for y in FORECAST_YEARS:
                row[f"forecast_{y}"] = data["forecast"].get(y, "")
            rows.append(row)
 
    out_df = pd.DataFrame(rows)
    out_df.to_csv("ev_forecast_results.csv", index=False)
    print(f"\nSaved ev_forecast_results.csv ({len(out_df)} rows)")
 
    # Per-scenario wide CSVs: same shape as input (State + year columns, 2016–2050).
    # Historical years use the observed values; forecast years use fitted curve values.
    all_years_wide = years_hist + FORECAST_YEARS
    for scenario in K_SCENARIOS:
        wide_rows = []
        for state, counts, res in all_results:
            data = res["scenarios"].get(scenario)
            if data is None:
                continue
            row = {"State": state}
            # Historical years — observed values
            for i, y in enumerate(years_hist):
                row[str(y)] = counts[i]
            # Forecast years — logistic curve values
            for y in FORECAST_YEARS:
                row[str(y)] = data["forecast"].get(y, "")
            wide_rows.append(row)
 
        wide_df = pd.DataFrame(wide_rows, columns=["State"] + [str(y) for y in all_years_wide])
        fname = f"ev_forecast_{scenario}_2050.csv"
        wide_df.to_csv(fname, index=False)
        print(f"Saved {fname} ({len(wide_df)} states, {len(all_years_wide)} years)")
 
    # Charts
    if PLOT_NATIONAL:
        plot_national(all_results, ev_data, years_hist, FORECAST_YEARS)
 
    if PLOT_STATES:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        states_to_plot = list(ev_data.keys()) if PLOT_STATES is True else PLOT_STATES
        for state, counts, res in all_results:
            if state in states_to_plot:
                plot_state(state, counts, res, years_hist, FORECAST_YEARS)
 
    if PLOT_MAPS:
        print("\nGenerating state choropleth maps...")
        plot_state_maps(all_results, SHAPEFILE_PATH)
        plot_saturation_maps(all_results, SHAPEFILE_PATH)
 
    # Summary
    print("\n── 2030 national forecast by scenario ──")
    total_all = sum(total_vehicles.get(s, 2_000_000) for s in ev_data)
    for scenario in ["conservative", "mid", "aggressive"]:
        total_2030 = sum(
            res["scenarios"][scenario]["forecast"].get(2030, 0)
            for _, _, res in all_results
            if res["scenarios"].get(scenario)
        )
        pct = total_2030 / total_all * 100 if total_all else 0
        print(f"  {scenario.capitalize():12s}: {total_2030/1e6:.2f}M EVs  ({pct:.1f}% of all vehicles)")
 
    print("\nDone.")
 
 
if __name__ == "__main__":
    main()
 
