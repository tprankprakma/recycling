
"""
EV Battery Chemistry Mix Forecast — U.S. Light-Duty Vehicles
=============================================================
Produces a year-by-year chemistry share table (2016–2050) using two
categories: High-Nickel and LFP.
 
Methodology
-----------
  - Pre-2022: 2022 values held constant backward (no reliable US data)
  - 2022:     IEA, "Electric vehicle battery sales share by chemistry and
              region, 2022-2024" (2025). LFP ~8% in US.
  - 2024:     Same source. LFP ~9% in US.
  - 2024+:    Held constant at 2024 values through 2050. No future projection
              is attempted given the uncertainty in US chemistry mix evolution.
 
Chemistries
-----------
  High-Nickel — NMC 811 and NCA combined. NMC 811 used as representative
                for mineral intensity (Olivetti et al. 2017, Table 1).
                NCA mineral intensities are nearly identical so the
                approximation introduces negligible error.
  LFP         — lithium iron phosphate. Mineral intensities derived from
                LiFePO4 stoichiometry (MW = 157.757 g/mol, theoretical
                capacity 170 mAh/g, nominal voltage 3.2 V, 90% utilization).
 
Sources
-------
  IEA, "Electric vehicle battery sales share by chemistry and region,
    2022-2024" (2025)
    https://www.iea.org/data-and-statistics/charts/electric-vehicle-battery-sales-share-by-chemistry-and-region-2022-2024
  Olivetti et al. (2017), Joule 1(2), 229-243
    https://doi.org/10.1016/j.joule.2017.08.003
 
Usage
-----
    pip install numpy pandas matplotlib
    python battery_chemistry.py
 
Outputs
-------
    battery_chemistry.csv     — single CSV (one scenario, constant mix)
    battery_chemistry_plot.png
"""
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
# ── Configuration ──────────────────────────────────────────────────────────────
 
OUTPUT_YEARS = list(range(2016, 2051))
CHEMISTRIES = ["High-Nickel", "LFP"]
COLORS = {"High-Nickel": "#185FA5", "LFP": "#BA7517"}
 
# ── Anchors ────────────────────────────────────────────────────────────────────
# Source: IEA, "Electric vehicle battery sales share by chemistry and
# region, 2022-2024" (2025).
 
ANCHOR_2022 = {"LFP": 8,  "High-Nickel": 92}
ANCHOR_2024 = {"LFP": 9,  "High-Nickel": 91}
 
 
# ── Interpolation ──────────────────────────────────────────────────────────────
 
def get_shares(year):
    """
    Return chemistry shares for a given year.
    - Before 2022: held constant at 2022 values.
    - 2022-2024:   linearly interpolated between anchors.
    - After 2024:  held constant at 2024 values.
    """
    if year <= 2022:
        return ANCHOR_2022.copy()
    if year >= 2024:
        return ANCHOR_2024.copy()
    # Linear interpolation between 2022 and 2024
    t = (year - 2022) / (2024 - 2022)
    return {c: ANCHOR_2022[c] + t * (ANCHOR_2024[c] - ANCHOR_2022[c])
            for c in CHEMISTRIES}
 
 
# ── Main ───────────────────────────────────────────────────────────────────────
 
def main():
    rows = []
    for year in OUTPUT_YEARS:
        shares = get_shares(year)
        rows.append({"year": year, **shares})
 
    df = pd.DataFrame(rows)
    for chem in CHEMISTRIES:
        df[chem] = df[chem].round(1)
 
    df.to_csv("battery_chemistry.csv", index=False)
    print(f"Saved battery_chemistry.csv ({len(df)} rows)")
 
    # Print summary
    key_years = [2016, 2020, 2022, 2023, 2024, 2030, 2040, 2050]
    print(f"\n{'Year':>6}  {'High-Nickel':>12}  {'LFP':>7}")
    for y in key_years:
        r = df[df["year"] == y].iloc[0]
        print(f"{y:>6}  {r['High-Nickel']:>11.1f}%  {r['LFP']:>6.1f}%")
 
    # Plot
    fig, ax = plt.subplots(figsize=(9, 4))
    years = df["year"].values
    bottom = np.zeros(len(years))
 
    for chem in CHEMISTRIES:
        vals = df[chem].values
        ax.fill_between(years, bottom, bottom + vals,
                        color=COLORS[chem], alpha=0.85, label=chem)
        mid_idx = np.searchsorted(years, 2035)
        mid_y = bottom[mid_idx] + vals[mid_idx] / 2
        if vals[mid_idx] > 5:
            ax.text(2035, mid_y, chem, ha="center", va="center",
                    fontsize=10, fontweight="bold", color="white")
        bottom += vals
 
    # Mark anchors and plateau
    for ay in [2022, 2024]:
        ax.axvline(ay, color="white", linewidth=0.8, linestyle="--", alpha=0.6)
 
    ax.axvspan(2016, 2022, color="white", alpha=0.08)
    ax.text(2019, 102, "← held\nconstant", ha="center", va="bottom",
            fontsize=7, color="#5F5E5A", style="italic")
    ax.axvspan(2024, 2050, color="white", alpha=0.05)
    ax.text(2037, 102, "held constant at 2024 →", ha="center", va="bottom",
            fontsize=7, color="#5F5E5A", style="italic")
 
    ax.set_xlim(2016, 2050)
    ax.set_ylim(0, 110)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Share of EV battery capacity (%)", fontsize=11)
    ax.set_title("U.S. EV Battery Chemistry Mix", fontsize=12)
    ax.legend(loc="center right", fontsize=10, framealpha=0.9)
    ax.grid(axis="y", linestyle="--", alpha=0.3, color="white")
 
    note = ("Anchors: 2022 and 2024 (IEA battery sales share by chemistry, 2025). "
            "Pre-2022 and post-2024 held constant.")
    fig.text(0.5, -0.04, note, ha="center", fontsize=8,
             color="#5F5E5A", style="italic")
 
    plt.tight_layout()
    plt.savefig("battery_chemistry_plot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved battery_chemistry_plot.png")
    print("\nDone.")
 
 
if __name__ == "__main__":
    main()
