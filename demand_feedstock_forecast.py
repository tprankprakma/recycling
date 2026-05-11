"""
EV Mineral Demand & Battery Feedstock Availability
===================================================
Takes the outputs of ev_forecast.py and battery_chemistry.py and computes:

  1. Annual new EV additions (difference of cumulative registration counts)
  2. Mineral demand from new batteries each year (kg per mineral)
  3. Battery feedstock availability — EVs registered N years ago become
     end-of-life and enter the recycling stream (default: 15 years)

Methodology
-----------
  New additions(y) = cumulative_registrations(y) - cumulative_registrations(y-1)
     (national total, summed across states)

  Mineral demand(y) = new_additions(y)
                      × avg_pack_size_kWh(y)
                      × sum_over_chemistries[ chemistry_share(y,c)
                                              × mineral_intensity(c, m) ]

  Feedstock(y) = new_additions(y - LIFETIME_YEARS)
                 × avg_pack_size_kWh(y - LIFETIME_YEARS)
                 × chemistry_mix(y - LIFETIME_YEARS)   [in kWh and kg]

Inputs
------
  ev_registrations_forecast.csv   — wide format, State × year (from ev_forecast.py)
                                    one scenario per file; pass one at a time
  battery_chemistry_results.csv   — long format, year/scenario/NMC/NCA/LFP/Other

Outputs (per combination of EV scenario × chemistry scenario)
--------------------------------------------------------------
  mineral_demand_{ev_scenario}_{chem_scenario}.csv
  mineral_demand_plot_{ev_scenario}_{chem_scenario}.png

Sources
-------
  Mineral intensities (kg/kWh):
    IEA, "The Role of Critical Minerals in Clean Energy Transitions" (2021),
    Annex Table — mineral demand per unit of battery capacity.
    Cross-checked: Argonne BatPaC ANL-20/55 (Nelson et al., 2020).
    NMC representative: NMC 811 (dominant US variant by ~2022, IEA GEO 2024).

  Average pack size (kWh):
    IEA Global EV Outlook 2024. US sales-weighted average.
    Linear ramp 50→80 kWh from 2016→2022, held at 80 kWh 2022→2024,
    then 80→85 kWh through 2030, held at 85 kWh thereafter.

  Battery lifetime assumption: 15 years (conservative midpoint).
    Range in literature: 10–20 years depending on use and climate.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Configuration ──────────────────────────────────────────────────────────────

# Input file paths — edit these
EV_FORECAST_FILES = {
    "conservative": "data/output/ev_forecast_conservative_2050.csv",
    "mid":          "data/output/ev_forecast_mid_2050.csv",
    "aggressive":   "data/output/ev_forecast_aggressive_2050.csv",
}

CHEMISTRY_FILE = "data/output/battery_chemistry_conservative.csv"

# Battery end-of-life assumption (years after registration)
LIFETIME_YEARS = 15

# ── Average pack size (kWh) by year ───────────────────────────────────────────
# Source: IEA GEO 2024; US sales-weighted average.
# Ramp 50→80 kWh (2016→2022), hold at 80 kWh (2022→2024),
# ramp 80→85 kWh (2024→2030), hold at 85 kWh (2030+).
def avg_pack_kwh(year: int) -> float:
    if year <= 2016:
        return 50.0
    if year <= 2022:
        return 50.0 + (year - 2016) / (2022 - 2016) * (80.0 - 50.0)
    if year <= 2024:
        return 80.0
    if year <= 2030:
        return 80.0 + (year - 2024) / (2030 - 2024) * (85.0 - 80.0)
    return 85.0


# ── Mineral intensities (kg per kWh of battery capacity) ─────────────────────
# Source: Olivetti et al. 2017, Stochiometry for LFP. 
# NMC = NMC 811 (dominant US variant; slightly overstates Ni, understates Co
#   vs a true NMC blend — acceptable given broader uncertainty).
# "Other" category uses a simple average of NMC811 and LFP as a placeholder.

MINERAL_INTENSITY = {
    #         Li      Ni      Co      Mn      Fe      P       
    "High-Nickel": {"Li": 0.111, "Ni": 0.750, "Co": 0.094, "Mn": 0.088,
            "Fe": 0.000, "P":  0.000},
    "LFP": {"Li": 0.090, "Ni": 0.000, "Co": 0.000, "Mn": 0.000,
            "Fe": 0.723, "P":  0.401}
}
MINERALS = ["Li", "Ni", "Co", "Mn", "Fe", "P"]
CHEMISTRIES = ["High-Nickel", "LFP"]

# ── Colours ────────────────────────────────────────────────────────────────────
MINERAL_COLORS = {
    "Li": "#185FA5", "Ni": "#BA7517", "Co": "#3B6D11",
    "Mn": "#7B4F9E", "Fe": "#888780", "P": "#C94040",
}


# ── Data loading ───────────────────────────────────────────────────────────────

def load_ev_forecast(path: str) -> pd.Series:
    """
    Load wide-format EV forecast CSV (State × year columns).
    Returns a Series of national cumulative totals indexed by year (int).
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    year_cols = sorted([c for c in df.columns if c.isdigit() and len(c) == 4], key=int)
    for c in year_cols:
        df[c] = pd.to_numeric(
            df[c].astype(str).str.replace(",", ""), errors="coerce"
        ).fillna(0)
    national = df[year_cols].sum()
    national.index = national.index.astype(int)
    return national


def load_chemistry(path: str) -> pd.DataFrame:
    """
    Load long-format chemistry CSV.
    Returns DataFrame indexed by (scenario, year) with columns NMC/NCA/LFP/Other.
    """
    df = pd.read_csv(path)
    df["year"] = df["year"].astype(int)
    df = df.set_index(["scenario", "year"])
    return df


# ── Core computation ───────────────────────────────────────────────────────────

def compute_gross_additions(cumulative: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Derive gross new EV sales and retirements from the target cumulative fleet.

    The S-curve forecast gives net registered EVs each year (fleet size).
    EVs retire after LIFETIME_YEARS, so the fleet evolves as:

        fleet(y) = fleet(y-1) + gross_sales(y) - retirements(y)
        retirements(y) = gross_sales(y - LIFETIME_YEARS)

    Rearranging:
        gross_sales(y) = fleet(y) - fleet(y-1) + retirements(y)
                       = Δfleet(y) + gross_sales(y - LIFETIME_YEARS)

    We solve this iteratively forward in time. Before LIFETIME_YEARS of
    gross_sales history exists, retirements = 0 (no cars old enough yet).

    Returns:
        gross_sales  — pd.Series indexed by year: new vehicles added
        retirements  — pd.Series indexed by year: vehicles leaving the fleet
    """
    years = sorted(cumulative.index)
    gross_sales  = {}
    retirements  = {}

    for i, y in enumerate(years):
        if i == 0:
            # No prior year: all cumulative EVs were sold this year
            ret = 0
        else:
            eol_year = y - LIFETIME_YEARS
            ret = gross_sales.get(eol_year, 0)

        retirements[y] = ret

        if i == 0:
            delta_fleet = cumulative[y]
        else:
            delta_fleet = cumulative[y] - cumulative[years[i - 1]]

        # Gross sales must be >= 0; negative delta (fleet shrinks faster than
        # retirements) is floored at 0 — can't have negative sales.
        gross_sales[y] = max(delta_fleet + ret, 0)

    return pd.Series(gross_sales), pd.Series(retirements)


def blend_mineral_intensity(chem_row: pd.Series) -> dict:
    """
    Given a row of chemistry shares (NMC%, NCA%, LFP%, Other%),
    return the share-weighted mineral intensity (kg/kWh).
    """
    blended = {m: 0.0 for m in MINERALS}
    for chem in CHEMISTRIES:
        share = chem_row[chem] / 100.0
        for m in MINERALS:
            blended[m] += share * MINERAL_INTENSITY[chem][m]
    return blended


def get_chem_row(chem_df, chem_scenario, year):
    """Return the chemistry row for a given scenario and year, with fallback."""
    if (chem_scenario, year) in chem_df.index:
        return chem_df.loc[(chem_scenario, year)]
    available = [idx[1] for idx in chem_df.index if idx[0] == chem_scenario]
    nearest = min(available, key=lambda ay: abs(ay - year))
    return chem_df.loc[(chem_scenario, nearest)]


def run(ev_scenario: str, chem_scenario: str,
        cumulative: pd.Series, chem_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mineral demand and feedstock availability for one
    (ev_scenario, chem_scenario) combination.

    gross_sales(y)  = Δfleet(y) + retirements(y)
                    = change in registered EVs + EVs that aged out this year
    retirements(y)  = gross_sales(y - LIFETIME_YEARS)

    Mineral demand  = gross_sales(y) × pack_size(y) × intensity(chemistry(y))
    Feedstock       = retirements(y) × pack_size(y-LIFETIME) × intensity(chemistry(y-LIFETIME))

    Returns a DataFrame with one row per year.
    """
    gross_sales, retirements = compute_gross_additions(cumulative)
    years = sorted(cumulative.index)

    rows = []
    for y in years:
        pack_kwh  = avg_pack_kwh(y)
        new_evs   = gross_sales[y]
        total_kwh = new_evs * pack_kwh
        chem_row  = get_chem_row(chem_df, chem_scenario, y)
        intensity = blend_mineral_intensity(chem_row)

        row = {
            "year":                   y,
            "ev_scenario":            ev_scenario,
            "chem_scenario":          chem_scenario,
            "cumulative_evs":         cumulative[y],
            "retirements":            retirements[y],
            "gross_new_sales":        new_evs,
            "avg_pack_kwh":           pack_kwh,
            "total_new_battery_kwh":  total_kwh,
        }

        for chem in CHEMISTRIES:
            row[f"share_{chem}_pct"] = chem_row[chem]

        # Mineral demand from new batteries sold this year
        for m in MINERALS:
            row[f"demand_kg_{m}"] = total_kwh * intensity[m]

        # Feedstock: batteries from vehicles retiring this year
        eol_year = y - LIFETIME_YEARS
        eol_evs  = retirements[y]
        if eol_evs > 0 and eol_year in cumulative.index:
            eol_pack_kwh  = avg_pack_kwh(eol_year)
            eol_total_kwh = eol_evs * eol_pack_kwh
            eol_chem      = get_chem_row(chem_df, chem_scenario, eol_year)
            eol_intensity = blend_mineral_intensity(eol_chem)
            row["feedstock_source_year"] = eol_year
            row["feedstock_evs"]         = eol_evs
            row["feedstock_total_kwh"]   = eol_total_kwh
            for m in MINERALS:
                row[f"feedstock_kg_{m}"] = eol_total_kwh * eol_intensity[m]
        else:
            row["feedstock_source_year"] = None
            row["feedstock_evs"]         = 0
            row["feedstock_total_kwh"]   = 0
            for m in MINERALS:
                row[f"feedstock_kg_{m}"] = 0

        rows.append(row)

    return pd.DataFrame(rows)


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_results(df: pd.DataFrame, ev_scenario: str, chem_scenario: str,
                 out_path: str):
    """Four-panel chart: new additions, key mineral demand, feedstock vs demand."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    years = df["year"].values

    def fmt_millions(ax):
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6
                                  else f"{x/1e3:.0f}K"))

    # Panel 1: Gross new EV sales per year (net additions + replacements)
    ax = axes[0, 0]
    ax.bar(years, df["gross_new_sales"], color="#185FA5", alpha=0.8, width=0.8)
    ax.set_title("Gross New EV Sales per Year\n(net additions + replacements)")
    ax.set_ylabel("Vehicles")
    fmt_millions(ax)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Panel 2: Key mineral demand (Li, Ni, Co) — stacked
    ax = axes[0, 1]
    highlight = ["Li", "Ni", "Co"]
    bottom = np.zeros(len(years))
    for m in highlight:
        vals = df[f"demand_kg_{m}"].values / 1e6  # → thousand tonnes
        ax.fill_between(years, bottom, bottom + vals,
                        color=MINERAL_COLORS[m], alpha=0.85, label=m)
        bottom += vals
    ax.set_title("Annual Mineral Demand (new batteries)")
    ax.set_ylabel("Thousand tonnes")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Panel 3: Demand vs feedstock for Li and Ni (most policy-relevant)
    ax = axes[1, 0]
    for m, ls in [("Li", "-"), ("Ni", "-"), ("Co", "-")]:
        demand   = df[f"demand_kg_{m}"].values / 1e6
        feedstk  = df[f"feedstock_kg_{m}"].values / 1e6
        ax.plot(years, demand,  color=MINERAL_COLORS[m], lw=2, ls=ls,
                label=f"{m} demand")
        ax.plot(years, feedstk, color=MINERAL_COLORS[m], lw=1.5, ls=":",
                alpha=0.7, label=f"{m} feedstock")
    ax.set_title("Demand vs. Recycling Feedstock\n(Li, Ni, Co; dotted = feedstock)")
    ax.set_ylabel("Thousand tonnes")
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Panel 4: Feedstock as % of demand (recycled content potential)
    ax = axes[1, 1]
    for m in ["Li", "Ni", "Co"]:
        demand   = df[f"demand_kg_{m}"].values
        feedstk  = df[f"feedstock_kg_{m}"].values
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(demand > 0, feedstk / demand * 100, 0)
        ax.plot(years, ratio, color=MINERAL_COLORS[m], lw=2, label=m)
    ax.axhline(100, color="#888780", lw=0.8, ls="--", alpha=0.6)
    ax.set_title("Feedstock as % of Annual Demand\n(recycling self-sufficiency potential)")
    ax.set_ylabel("Feedstock / Demand (%)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_ylim(0)

    for ax in axes.flat:
        ax.set_xlim(years[0], years[-1])
        ax.set_xlabel("Year")

    fig.suptitle(
        f"EV Mineral Demand & Feedstock Availability\n"
        f"EV scenario: {ev_scenario}  |  Chemistry scenario: {chem_scenario}  "
        f"|  Battery lifetime: {LIFETIME_YEARS} yr",
        fontsize=12
    )
    note = ("Mineral intensities: IEA Critical Minerals (2021) / Argonne BatPaC ANL-20/55. "
            "Pack size: IEA GEO 2024. Feedstock = EVs registered "
            f"{LIFETIME_YEARS} years prior entering recycling stream.")
    fig.text(0.5, -0.02, note, ha="center", fontsize=8,
             color="#5F5E5A", style="italic")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("Loading chemistry data...")
    chem_df = load_chemistry(CHEMISTRY_FILE)
    chem_scenarios = chem_df.index.get_level_values("scenario").unique().tolist()

    all_dfs = []

    for ev_scenario, ev_path in EV_FORECAST_FILES.items():
        if not os.path.exists(ev_path):
            print(f"  Skipping {ev_scenario}: file not found ({ev_path})")
            continue

        print(f"Loading EV forecast: {ev_path}")
        cumulative = load_ev_forecast(ev_path)

        for chem_scenario in chem_scenarios:
            print(f"  Computing: EV={ev_scenario}, chem={chem_scenario}")
            df = run(ev_scenario, chem_scenario, cumulative, chem_df)
            all_dfs.append(df)

            stem = f"mineral_demand_{ev_scenario}_{chem_scenario}"

            # One CSV per scenario combination
            float_cols = df.select_dtypes("float").columns
            df[float_cols] = df[float_cols].round(0)
            df.to_csv(f"{stem}.csv", index=False)
            print(f"  Saved {stem}.csv")

            plot_results(df, ev_scenario, chem_scenario, f"{stem}.png")

    # Print a readable summary for the mid/mid scenario
    all_dfs_nonempty = [d for d in all_dfs if not d.empty]
    if all_dfs_nonempty:
        combined_for_summary = pd.concat(all_dfs_nonempty, ignore_index=True)
        if "mid" in EV_FORECAST_FILES and os.path.exists(EV_FORECAST_FILES["mid"]):
            sub = combined_for_summary[
                (combined_for_summary["ev_scenario"] == "mid") &
                (combined_for_summary["chem_scenario"] == "mid")
            ]
            key_years = [2024, 2026, 2028, 2030, 2035, 2040, 2050]
            print(f"\n── Summary: EV=mid, chem=mid ──")
            print(f"{'Year':>6}  {'Gross Sales':>11}  {'Retirements':>11}  "
                  f"{'Li (t)':>9}  {'Ni (t)':>9}  {'Co (t)':>9}  "
                  f"{'Feedstk EVs':>12}  {'Feedstk Li (t)':>14}")
            for _, r in sub[sub["year"].isin(key_years)].iterrows():
                print(f"{int(r['year']):>6}  {int(r['gross_new_sales']):>11,}  "
                      f"{int(r['retirements']):>11,}  "
                      f"{int(r['demand_kg_Li']/1000):>9,}  "
                      f"{int(r['demand_kg_Ni']/1000):>9,}  "
                      f"{int(r['demand_kg_Co']/1000):>9,}  "
                      f"{int(r['feedstock_evs']):>12,}  "
                      f"{int(r['feedstock_kg_Li']/1000):>14,}")

    print("\nDone.")


if __name__ == "__main__":
    main()