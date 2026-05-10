"""
EV Battery Chemistry Mix Forecast — U.S. Light-Duty Vehicles
=============================================================
Produces a year-by-year chemistry share table (2016–2050) by interpolating
between three citable anchors and a user-chosen 2030 scenario.
 
Methodology
-----------
Three anchors, linearly interpolated between:
  - Pre-2022: 2022 values held constant backward (no reliable US data)
  - 2022:     Citable from IEA Global EV Outlook 2023
  - 2024:     Citable from IEA Global EV Outlook 2025
  - 2030:     Scenario-based (conservative / mid / aggressive LFP growth)
  - 2050:     Extrapolated linearly from the 2024→2030 slope, capped so
              shares stay non-negative and sum to 100%
 
Sources
-------
- IEA Global EV Outlook 2023  https://iea.org/reports/global-ev-outlook-2023/trends-in-batteries
- IEA Global EV Outlook 2024  https://iea.org/reports/global-ev-outlook-2024/trends-in-electric-vehicle-batteries
- IEA Global EV Outlook 2025  https://iea.org/reports/global-ev-outlook-2025/electric-vehicle-batteries
- IEA "The Battery Industry Has Entered a New Phase" (2025)
- CSIS "A New Phase for the U.S. Battery Industry" (April 2026)
 
Chemistries tracked
-------------------
  NMC   — lithium nickel manganese cobalt oxide (dominant in US/Europe OEMs)
  NCA   — lithium nickel cobalt aluminium oxide (Tesla US historically)
  LFP   — lithium iron phosphate (growing, still <10% US through 2024)
  Other — low-nickel NMC, NMCA, emerging (Na-ion, solid-state, LMR etc.)
 
Usage
-----
    pip install numpy pandas matplotlib
    python battery_chemistry.py
 
Outputs
-------
    battery_chemistry_results.csv   — all scenarios, all years
    battery_chemistry_plot.png      — stacked area chart per scenario
"""
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
 
# ── Configuration ──────────────────────────────────────────────────────────────
 
OUTPUT_YEARS = list(range(2016, 2051))
 
# Choose which scenarios to run. Any subset of the keys in ANCHOR_2030.
SCENARIOS_TO_RUN = ["conservative", "mid", "aggressive"]
 
# ── Anchors ────────────────────────────────────────────────────────────────────
# All values are % of EV battery capacity (share sums to 100).
# Chemistries: NMC, NCA, LFP, Other
 
# Anchor 1: 2022
# Source: IEA Global EV Outlook 2023.
# Global: NMC ~60%, LFP ~29%, NCA ~8%. US adjustment: shift ~20pp from LFP
# to NCA (Tesla fleet effect), consistent with IEA noting LFP <10% in US.
ANCHOR_2022 = {"NMC": 62, "NCA": 26, "LFP": 8, "Other": 4}
 
# Anchor 2: 2024
# Source: IEA Global EV Outlook 2025 — LFP "below 10%" in US,
# high-nickel dominant; Tesla accounts for 85% of US LFP vehicles.
ANCHOR_2024 = {"NMC": 58, "NCA": 28, "LFP": 9, "Other": 5}
 
# Anchor 3: 2030 — three scenarios.
# Basis: CSIS (2026) and IEA GEO 2025 / "Battery Industry Has Entered a New
# Phase" (2025). LFP range of 15–35% by 2030 in the US depending on whether
# domestic LFP production comes online and IRA rules hold.
#
# Conservative: IRA foreign-entity rules hold, domestic LFP slow to scale.
#               NMC/NCA maintain dominance. LFP grows modestly.
# Mid:          Some domestic LFP online (ONE, FREYR), affordable EVs launch
#               with LFP (new Bolt, Ford truck). Moderate shift.
# Aggressive:   IRA rules soften or domestic LFP scales fast. Strong LFP push
#               from affordable EV segment. Na-ion begins appearing.
ANCHOR_2030 = {
    "conservative": {"NMC": 48, "NCA": 23, "LFP": 22, "Other": 7},
    "mid":          {"NMC": 42, "NCA": 20, "LFP": 30, "Other": 8},
    "aggressive":   {"NMC": 34, "NCA": 16, "LFP": 38, "Other": 12},
}
 
CHEMISTRIES = ["NMC", "NCA", "LFP", "Other"]
COLORS = {"NMC": "#185FA5", "NCA": "#3B6D11", "LFP": "#BA7517", "Other": "#888780"}
 
# ── Interpolation ──────────────────────────────────────────────────────────────
 
def interpolate_shares(year, anchors):
    """
    Piecewise linear interpolation across a list of (year, {chem: share}) anchors.
    Before first anchor: hold first anchor constant.
    After last anchor:   extrapolate linearly from last two anchors,
                         then normalise so shares remain non-negative and sum to 100.
    """
    years_a = [a[0] for a in anchors]
    shares_a = [a[1] for a in anchors]
 
    if year <= years_a[0]:
        return shares_a[0].copy()
 
    if year >= years_a[-1]:
        # Linear extrapolation from last segment
        y0, y1 = years_a[-2], years_a[-1]
        s0, s1 = shares_a[-2], shares_a[-1]
        t = (year - y1) / (y1 - y0)
        raw = {c: s1[c] + t * (s1[c] - s0[c]) for c in CHEMISTRIES}
        # Clip negatives, renormalise
        clipped = {c: max(raw[c], 0.0) for c in CHEMISTRIES}
        total = sum(clipped.values())
        return {c: clipped[c] / total * 100 for c in CHEMISTRIES}
 
    # Find surrounding anchors
    for i in range(len(years_a) - 1):
        if years_a[i] <= year <= years_a[i + 1]:
            t = (year - years_a[i]) / (years_a[i + 1] - years_a[i])
            return {
                c: shares_a[i][c] + t * (shares_a[i + 1][c] - shares_a[i][c])
                for c in CHEMISTRIES
            }
 
 
def build_scenario(scenario_name):
    """
    Build year-by-year shares for one scenario.
    Anchors: [2022, 2024, 2030], with pre-2022 held at 2022 values.
    2030→2050 extrapolated linearly then normalised.
    """
    anchor_2030 = ANCHOR_2030[scenario_name]
    anchors = [
        (2022, ANCHOR_2022),
        (2024, ANCHOR_2024),
        (2030, anchor_2030),
    ]
 
    rows = []
    for year in OUTPUT_YEARS:
        shares = interpolate_shares(year, anchors)
        rows.append({"year": year, "scenario": scenario_name, **shares})
 
    return rows
 
 
# ── Main ───────────────────────────────────────────────────────────────────────
 
def main():
    all_rows = []
    for scenario in SCENARIOS_TO_RUN:
        all_rows.extend(build_scenario(scenario))
 
    df = pd.DataFrame(all_rows)
 
    # Round for readability
    for chem in CHEMISTRIES:
        df[chem] = df[chem].round(1)
 
    # One CSV per scenario
    for scenario in SCENARIOS_TO_RUN:
        sub = df[df["scenario"] == scenario]
        fname = f"battery_chemistry_{scenario}.csv"
        sub.to_csv(f"data/output/{fname}", index=False)
        print(f"Saved {fname} ({len(sub)} rows)")
 
    # ── Print summary tables ───────────────────────────────────────────────────
    key_years = [2016, 2020, 2022, 2024, 2026, 2028, 2030, 2035, 2040, 2050]
    for scenario in SCENARIOS_TO_RUN:
        sub = df[df["scenario"] == scenario].set_index("year")
        print(f"\n── {scenario.capitalize()} scenario ──")
        print(f"{'Year':>6}  {'NMC':>7}  {'NCA':>7}  {'LFP':>7}  {'Other':>7}")
        for y in key_years:
            if y in sub.index:
                r = sub.loc[y]
                print(f"{y:>6}  {r['NMC']:>6.1f}%  {r['NCA']:>6.1f}%  {r['LFP']:>6.1f}%  {r['Other']:>6.1f}%")
 
    # ── Plot ───────────────────────────────────────────────────────────────────
    n = len(SCENARIOS_TO_RUN)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]
 
    for ax, scenario in zip(axes, SCENARIOS_TO_RUN):
        sub = df[df["scenario"] == scenario].sort_values("year")
        years = sub["year"].values
        bottom = np.zeros(len(years))
 
        for chem in CHEMISTRIES:
            vals = sub[chem].values
            ax.fill_between(years, bottom, bottom + vals,
                            color=COLORS[chem], alpha=0.85, label=chem)
            # Label chemistry in the middle of its band at 2040
            mid_year_idx = np.searchsorted(years, 2040)
            mid_y = bottom[mid_year_idx] + vals[mid_year_idx] / 2
            if vals[mid_year_idx] > 4:
                ax.text(2040, mid_y, chem, ha="center", va="center",
                        fontsize=9, fontweight="bold", color="white")
            bottom += vals
 
        # Mark anchors
        for ay in [2022, 2024, 2030]:
            ax.axvline(ay, color="white", linewidth=0.8, linestyle="--", alpha=0.6)
 
        # Shade pre-2022 (held constant / no data)
        ax.axvspan(2016, 2022, color="white", alpha=0.08)
        ax.text(2019, 102, "← held\nconstant", ha="center", va="bottom",
                fontsize=7, color="#5F5E5A", style="italic")
 
        ax.set_xlim(2016, 2050)
        ax.set_ylim(0, 110)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
        ax.set_xlabel("Year", fontsize=11)
        ax.set_title(f"{scenario.capitalize()}", fontsize=12)
        ax.grid(axis="y", linestyle="--", alpha=0.3, color="white")
 
    axes[0].set_ylabel("Share of EV battery capacity (%)", fontsize=11)
 
    # Shared legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[c], alpha=0.85)
               for c in CHEMISTRIES]
    fig.legend(handles, CHEMISTRIES, loc="lower center", ncol=4,
               fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, -0.05))
 
    fig.suptitle("U.S. EV Battery Chemistry Mix Forecast\n"
                 "Piecewise linear interpolation between IEA/CSIS anchors",
                 fontsize=12, y=1.02)
 
    note = ("Anchors: 2022 (IEA GEO 2023), 2024 (IEA GEO 2025), "
            "2030 (CSIS 2026 / IEA 2025 scenarios).\n"
            "Pre-2022 held constant at 2022 values (no reliable US-specific data). "
            "Post-2030 linearly extrapolated.")
    fig.text(0.5, -0.1, note, ha="center", fontsize=8,
             color="#5F5E5A", style="italic", wrap=True)
 
    plt.tight_layout()
    plt.savefig("battery_chemistry_plot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\nSaved battery_chemistry_plot.png")
    print("\nDone.")
 
 
if __name__ == "__main__":
    main()
