"""
County-Level EV Charger Weights & Feedstock Downscaling
========================================================
Takes the AFDC alternative fuel stations CSV (public EV chargers, already
filtered to US/public/open) and:
 
  1. Assigns each station to a county using ZIP → county lookup
     (zipcodes package, data bundled locally — no network needed).
     Stations missing a ZIP are resolved via vectorized KD-tree nearest-
     neighbour search against ZIP centroid coordinates.
 
  2. Counts total EVSE ports per county (Level 1 + Level 2 + DC Fast).
 
  3. Computes each county's share of its state's total ports → weight.
 
  4. Applies those weights to national-level mineral feedstock from
     mineral_demand_{ev}_{chem}.csv to produce county-level estimates.
 
  5. Outputs:
       county_charger_counts.csv         — ports & weights per county (static)
       county_feedstock_{ev}_{chem}.csv  — one per scenario combination
 
Inputs
------
  STATIONS_CSV  — AFDC alt_fuel_stations_locations.csv
  MINERAL_FILES — dict mapping (ev_scenario, chem_scenario) to CSV paths
 
Methodology note
----------------
  Charger port density is used as a proxy for EV density within each state.
  This is a simplifying assumption: charger deployment broadly tracks EV
  adoption, but also reflects infrastructure investment patterns that may
  not perfectly mirror registration geography. It is the best publicly
  available county-level proxy without proprietary DMV microdata.
 
Sources
-------
  Station data:  AFDC Alternative Fuel Stations, https://afdc.energy.gov/stations
  ZIP→county:    zipcodes Python package (bundled GeoNames/USPS data, no network)
  County bounds: ZIP centroid KD-tree fallback for stations with missing ZIPs
"""
 
import os
import numpy as np
import pandas as pd
import zipcodes
from scipy.spatial import cKDTree
 
# ── Configuration ──────────────────────────────────────────────────────────────
 
STATIONS_CSV = "data/raw/alt_fuel_stations_locations.csv"
 
MINERAL_FILES = {
    ("conservative", "conservative"): "data/output/mineral_demand_conservative_conservative.csv",
    # ("conservative", "mid"):          "mineral_demand_conservative_mid.csv",
    # ("conservative", "aggressive"):   "mineral_demand_conservative_aggressive.csv",
    ("mid",          "conservative"): "data/output/mineral_demand_mid_conservative.csv",
    # ("mid",          "mid"):          "mineral_demand_mid_mid.csv",
    # ("mid",          "aggressive"):   "mineral_demand_mid_aggressive.csv",
    ("aggressive",   "conservative"): "data/output/mineral_demand_aggressive_conservative.csv",
    # ("aggressive",   "mid"):          "mineral_demand_aggressive_mid.csv",
    # ("aggressive",   "aggressive"):   "mineral_demand_aggressive_aggressive.csv",
}
 
FEEDSTOCK_PREFIX = "feedstock_kg_"
 
# ── ZIP → county lookup + KD-tree ─────────────────────────────────────────────
 
def build_lookup_and_tree():
    """
    Build:
      lookup     — {zip5: (county, state)} for direct ZIP matching
      cent_df    — DataFrame of ZIP centroids for KD-tree fallback
      tree       — scipy cKDTree on unit-sphere coordinates
    """
    all_zips = zipcodes.list_all()
    lookup = {}
    centroid_rows = []
 
    for z in all_zips:
        zc     = z.get("zip_code", "")
        county = z.get("county", "")
        state  = z.get("state", "")
        try:
            lat = float(z.get("lat") or 0)
            lon = float(z.get("long") or 0)
        except (ValueError, TypeError):
            lat, lon = 0.0, 0.0
 
        if zc and county:
            lookup[zc] = (county, state)
        if lat and lon and county:
            centroid_rows.append((lat, lon, county, state))
 
    cent_df = pd.DataFrame(centroid_rows, columns=["lat", "lon", "county", "state"])
 
    # Convert to unit-sphere Cartesian for fast nearest-neighbour on a globe
    lats_r = np.radians(cent_df["lat"].values)
    lons_r = np.radians(cent_df["lon"].values)
    xs = np.cos(lats_r) * np.cos(lons_r)
    ys = np.cos(lats_r) * np.sin(lons_r)
    zs = np.sin(lats_r)
    tree = cKDTree(np.column_stack([xs, ys, zs]))
 
    return lookup, cent_df, tree
 
 
# ── Station loading & county assignment ───────────────────────────────────────
 
def load_and_assign_counties(stations_path: str,
                              lookup: dict,
                              cent_df: pd.DataFrame,
                              tree) -> pd.DataFrame:
    """
    Load stations, assign county via ZIP lookup then KD-tree fallback,
    compute total EVSE ports per station.
    """
    print(f"Loading stations from: {stations_path}")
    df = pd.read_csv(stations_path, usecols=[
        "ZIP", "State", "Latitude", "Longitude",
        "EV Level1 EVSE Num", "EV Level2 EVSE Num", "EV DC Fast Count",
    ], low_memory=False)
 
    # Port counts — treat missing as 0, then floor at 1 per station
    for col in ["EV Level1 EVSE Num", "EV Level2 EVSE Num", "EV DC Fast Count"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["total_ports"] = (df["EV Level1 EVSE Num"]
                         + df["EV Level2 EVSE Num"]
                         + df["EV DC Fast Count"]).clip(lower=1)
 
    df["ZIP5"] = df["ZIP"].astype(str).str.extract(r"(\d{5})")[0].fillna("")
 
    # Primary: ZIP lookup
    zip_to_county = {k: v[0] for k, v in lookup.items()}
    df["county"] = df["ZIP5"].map(zip_to_county)
 
    # Fallback: vectorized KD-tree for rows without a ZIP match
    mask = df["county"].isna()
    n_fallback = mask.sum()
    if n_fallback > 0:
        fb = df[mask]
        lats_r = np.radians(fb["Latitude"].values)
        lons_r = np.radians(fb["Longitude"].values)
        xq = np.cos(lats_r) * np.cos(lons_r)
        yq = np.cos(lats_r) * np.sin(lons_r)
        zq = np.sin(lats_r)
        _, idx = tree.query(np.column_stack([xq, yq, zq]))
        df.loc[mask, "county"] = cent_df["county"].values[idx]
 
    df["state"] = df["State"]  # Use dataset's State column as authoritative
 
    n_zip     = len(df) - n_fallback
    n_unres   = df["county"].isna().sum()
    print(f"  ZIP matched:      {n_zip:,}")
    print(f"  KD-tree fallback: {n_fallback:,}")
    print(f"  Unresolved:       {n_unres:,}")
 
    return df
 
 
# ── Aggregate to county weights ────────────────────────────────────────────────
 
def build_county_weights(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sum EVSE ports by state + county.
    Compute county_weight = county_ports / state_ports.
    """
    county_agg = (df.groupby(["state", "county"])["total_ports"]
                    .sum().reset_index()
                    .rename(columns={"total_ports": "county_ports"}))
 
    state_totals = (county_agg.groupby("state")["county_ports"]
                               .sum().reset_index()
                               .rename(columns={"county_ports": "state_ports"}))
 
    county_agg = county_agg.merge(state_totals, on="state")
    county_agg["county_weight"] = (county_agg["county_ports"]
                                   / county_agg["state_ports"])
    return county_agg
 
 
# ── Downscale feedstock ────────────────────────────────────────────────────────
 
def downscale_feedstock(county_weights: pd.DataFrame,
                         ev_scenario: str, chem_scenario: str,
                         mineral_path: str) -> pd.DataFrame:
    """
    Distribute national-level feedstock columns to counties proportionally
    to each county's share of national EVSE ports.
    """
    state_df = pd.read_csv(mineral_path)
    feedstock_cols = [c for c in state_df.columns if c.startswith(FEEDSTOCK_PREFIX)]
 
    national_ports = county_weights["county_ports"].sum()
    cw = county_weights.copy()
    cw["national_weight"] = cw["county_ports"] / national_ports
 
    # Cross join years × counties, then scale
    state_df["_key"] = 1
    cw["_key"] = 1
    merged = state_df.merge(cw, on="_key").drop(columns="_key")
 
    out = pd.DataFrame({
        "ev_scenario":           ev_scenario,
        "chem_scenario":         chem_scenario,
        "year":                  merged["year"].astype(int),
        "state":                 merged["state"],
        "county":                merged["county"],
        "county_ports":          merged["county_ports"],
        "county_weight":         merged["county_weight"].round(6),
        "feedstock_source_year": merged.get("feedstock_source_year", pd.NA),
        "feedstock_evs":         (merged["feedstock_evs"]
                                  * merged["national_weight"]).round(0).astype(int),
        "feedstock_total_kwh":   (merged["feedstock_total_kwh"]
                                  * merged["national_weight"]).round(0).astype(int),
    })
    for col in feedstock_cols:
        kg_vals = (merged[col] * merged["national_weight"]).round(1)
        out[col] = kg_vals
        tonne_col = col.replace(FEEDSTOCK_PREFIX, "feedstock_t_")
        out[tonne_col] = (kg_vals / 1000).round(4)
 
    return out
 
 
# ── Main ───────────────────────────────────────────────────────────────────────
 
def main():
    print("Building ZIP→county lookup and KD-tree...")
    lookup, cent_df, tree = build_lookup_and_tree()
    print(f"  {len(lookup):,} ZIP codes in lookup, "
          f"{len(cent_df):,} centroids in KD-tree")
 
    station_df = load_and_assign_counties(STATIONS_CSV, lookup, cent_df, tree)
    county_weights = build_county_weights(station_df)
 
    # Save county charger counts
    county_out = (county_weights[["state", "county", "county_ports",
                                   "state_ports", "county_weight"]]
                  .sort_values(["state", "county_ports"], ascending=[True, False]))
    county_out.to_csv("county_charger_counts.csv", index=False)
    print(f"\nSaved county_charger_counts.csv — "
          f"{len(county_out):,} counties across "
          f"{county_out['state'].nunique()} states")
 
    print("\nTop 10 counties by EVSE ports:")
    print(county_out.nlargest(10, "county_ports")
          [["state", "county", "county_ports", "county_weight"]]
          .to_string(index=False))
 
    # Downscale feedstock for each scenario combination
    print("\nDownscaling feedstock to county level...")
    for (ev_s, chem_s), path in MINERAL_FILES.items():
        if not os.path.exists(path):
            print(f"  Skipping {ev_s}/{chem_s}: {path} not found")
            continue
        print(f"  EV={ev_s}, chem={chem_s}...", end=" ", flush=True)
        county_feed = downscale_feedstock(county_weights, ev_s, chem_s, path)
        out_path = f"county_feedstock_{ev_s}_{chem_s}.csv"
        county_feed.to_csv(out_path, index=False)
        print(f"saved {out_path} ({len(county_feed):,} rows)")
 
    print("\nDone.")
 
 
if __name__ == "__main__":
    main()
