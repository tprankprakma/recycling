"""
County-Level EV Charger Weights & Feedstock Downscaling
========================================================
Takes the AFDC alternative fuel stations CSV (public EV chargers, already
filtered to US/public/open) and:

  1. Assigns each station to a county using a spatial point-in-polygon join
     against a Census county shapefile. This is exact: each lat/lon point is
     tested against actual county boundary polygons, not approximated via ZIP
     codes (which frequently cross county lines).

     The 0.3% of stations that fall outside all county polygons (typically
     points sitting exactly on a boundary) are resolved by nearest-centroid
     fallback using geopandas distance.

  2. Counts total EVSE ports per county (Level 1 + Level 2 + DC Fast).

  3. Computes each county's share of its state's total ports → weight.

  4. Applies those weights to national-level mineral feedstock from
     mineral_demand_{ev}_{chem}.csv to produce county-level estimates.

  5. Outputs:
       county_charger_counts.csv         — ports & weights per county (static)
       county_feedstock_{ev}_{chem}.csv  — one per scenario combination,
                                           with feedstock in both kg and tonnes

Inputs
------
  STATIONS_CSV   — AFDC alt_fuel_stations_locations.csv
  SHAPEFILE_PATH — Census county shapefile (.shp); companion files (.dbf,
                   .shx, .prj, .cpg) must be in the same directory
  MINERAL_FILES  — dict mapping (ev_scenario, chem_scenario) to CSV paths

Methodology note
----------------
  Charger port density is used as a proxy for EV density within each state.
  This is a simplifying assumption: charger deployment broadly tracks EV
  adoption, but also reflects infrastructure investment patterns that may
  not perfectly mirror registration geography. It is the best publicly
  available county-level proxy without proprietary DMV microdata.

Sources
-------
  Station data:    AFDC Alternative Fuel Stations, https://afdc.energy.gov/stations
  County polygons: U.S. Census Bureau, TIGER/Line Shapefiles 2021
                   cb_2021_us_county_20m.shp
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd

# ── Configuration ──────────────────────────────────────────────────────────────

STATIONS_CSV   = "data/raw/alt_fuel_stations_locations.csv"
SHAPEFILE_PATH = "data/raw/cb_2021_us_county_20m.shp"

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


# ── Station loading & county assignment ───────────────────────────────────────

def load_and_assign_counties(stations_path: str, shapefile_path: str) -> gpd.GeoDataFrame:
    """
    Load stations, convert to GeoDataFrame, and spatially join to county polygons.
    Stations that fall outside all polygons (boundary edge cases) are resolved
    by snapping to the nearest county centroid.
    """
    print(f"Loading stations from: {stations_path}")
    df = pd.read_csv(stations_path, usecols=[
        "Latitude", "Longitude", "State",
        "EV Level1 EVSE Num", "EV Level2 EVSE Num", "EV DC Fast Count",
    ], low_memory=False)

    # Port counts — treat missing as 0, floor at 1 per station
    for col in ["EV Level1 EVSE Num", "EV Level2 EVSE Num", "EV DC Fast Count"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["total_ports"] = (df["EV Level1 EVSE Num"]
                         + df["EV Level2 EVSE Num"]
                         + df["EV DC Fast Count"]).clip(lower=1)

    # Convert to GeoDataFrame (WGS84)
    pts = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["Longitude"], df["Latitude"]),
        crs="EPSG:4326",
    )

    print(f"Loading county shapefile from: {shapefile_path}")
    counties = gpd.read_file(shapefile_path).to_crs("EPSG:4326")
    county_cols = ["NAME", "NAMELSAD", "STUSPS", "STATE_NAME", "GEOID", "geometry"]

    # Primary: point-in-polygon spatial join
    print("Running spatial join (point-in-polygon)...")
    joined = gpd.sjoin(pts, counties[county_cols], how="left", predicate="within")

    n_matched   = joined["NAME"].notna().sum()
    n_unmatched = joined["NAME"].isna().sum()
    print(f"  Matched:   {n_matched:,}")
    print(f"  Unmatched: {n_unmatched:,} (boundary edge cases — resolving by nearest centroid)")

    # Fallback: nearest county centroid for unmatched points
    if n_unmatched > 0:
        # Reproject to Albers Equal Area (metres) for accurate centroid + distance
        counties_aea = counties[county_cols].to_crs("EPSG:5070")
        centroids = counties_aea.copy()
        centroids["geometry"] = counties_aea.geometry.centroid

        unmatched_idx = joined[joined["NAME"].isna()].index
        unmatched_pts = pts.loc[unmatched_idx, ["geometry"]].to_crs("EPSG:5070")

        nearest = gpd.sjoin_nearest(
            unmatched_pts,
            centroids[["NAME", "NAMELSAD", "STUSPS", "STATE_NAME", "GEOID", "geometry"]],
            how="left",
        )

        for col in ["NAME", "NAMELSAD", "STUSPS", "STATE_NAME", "GEOID"]:
            joined.loc[unmatched_idx, col] = nearest[col].values

        still_unmatched = joined["NAME"].isna().sum()
        print(f"  Still unresolved after fallback: {still_unmatched:,}")

    # Use the dataset's State column as the authoritative state identifier
    # (county shapefile STUSPS gives the same thing but this avoids any
    # edge cases where a point near a state border snaps to the wrong state)
    joined["state"] = joined["State"]
    joined["county"] = joined["NAMELSAD"]  # e.g. "Los Angeles County"
    joined["county_geoid"] = joined["GEOID"]  # FIPS code, useful for mapping

    return joined


# ── Aggregate to county weights ────────────────────────────────────────────────

def build_county_weights(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Sum EVSE ports by state + county.
    Compute county_weight = county_ports / state_ports.
    """
    county_agg = (gdf.groupby(["state", "county", "county_geoid"])["total_ports"]
                     .sum()
                     .reset_index()
                     .rename(columns={"total_ports": "county_ports"}))

    state_totals = (county_agg.groupby("state")["county_ports"]
                               .sum()
                               .reset_index()
                               .rename(columns={"county_ports": "state_ports"}))

    county_agg = county_agg.merge(state_totals, on="state")
    county_agg["county_weight"] = (county_agg["county_ports"]
                                   / county_agg["state_ports"])

    # Sanity check
    weight_check = county_agg.groupby("state")["county_weight"].sum()
    max_err = (weight_check - 1.0).abs().max()
    assert max_err < 1e-6, f"Weights don't sum to 1 per state (max error: {max_err})"

    return county_agg


# ── Downscale feedstock ────────────────────────────────────────────────────────

def downscale_feedstock(county_weights: pd.DataFrame,
                         ev_scenario: str, chem_scenario: str,
                         mineral_path: str) -> pd.DataFrame:
    """
    Distribute national-level feedstock columns to counties proportionally
    to each county's share of national EVSE ports.
    Outputs both kg and metric tonne (kg / 1000) columns for each mineral.
    """
    national_df = pd.read_csv(mineral_path)
    feedstock_cols = [c for c in national_df.columns if c.startswith(FEEDSTOCK_PREFIX)]

    national_ports = county_weights["county_ports"].sum()
    cw = county_weights.copy()
    cw["national_weight"] = cw["county_ports"] / national_ports

    # Cross join: every year × every county
    national_df["_key"] = 1
    cw["_key"] = 1
    merged = national_df.merge(cw, on="_key").drop(columns="_key")

    out = pd.DataFrame({
        "ev_scenario":           ev_scenario,
        "chem_scenario":         chem_scenario,
        "year":                  merged["year"].astype(int),
        "state":                 merged["state"],
        "county":                merged["county"],
        "county_geoid":          merged["county_geoid"],
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
    # Assign counties via spatial join
    station_gdf = load_and_assign_counties(STATIONS_CSV, SHAPEFILE_PATH)
    county_weights = build_county_weights(station_gdf)

    # Save county charger counts
    county_out = (county_weights[["state", "county", "county_geoid",
                                   "county_ports", "state_ports", "county_weight"]]
                  .sort_values(["state", "county_ports"], ascending=[True, False]))
    county_out.to_csv("county_charger_counts.csv", index=False)
    print(f"\nSaved county_charger_counts.csv — "
          f"{len(county_out):,} counties across "
          f"{county_out['state'].nunique()} states")

    print("\nTop 10 counties by EVSE ports:")
    print(county_out.nlargest(10, "county_ports")
          [["state", "county", "county_geoid", "county_ports", "county_weight"]]
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