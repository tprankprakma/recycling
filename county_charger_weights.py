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
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm


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
 
# Maps: years, minerals, and output directory
MAP_YEARS    = [2031, 2040, 2050]
MAP_MINERALS = ["Li", "Ni", "Co", "Graphite"]   # subset with most policy relevance
MAP_DIR      = "feedstock_maps"
 
# Colour palettes per mineral (light → dark, perceptually uniform)
MAP_COLORS = {
    "Li":       ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"],
    "Ni":       ["#fff5eb", "#fdd0a2", "#fd8d3c", "#d94701", "#7f2704"],
    "Co":       ["#f7fcf5", "#c7e9c0", "#74c476", "#238b45", "#00441b"],
    "Graphite": ["#f7f7f7", "#cccccc", "#969696", "#525252", "#252525"],
}
 
 
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
 
def build_county_weights(gdf: gpd.GeoDataFrame, shapefile_path: str) -> pd.DataFrame:
    """
    Sum EVSE ports by state + county, then left-join onto the full county list
    from the shapefile so counties with no chargers are retained with 0 ports
    and 0 weight (rather than being dropped entirely).
    """
    # Aggregate charger ports from stations that were matched
    county_agg = (gdf.groupby(["state", "county", "county_geoid"])["total_ports"]
                     .sum()
                     .reset_index()
                     .rename(columns={"total_ports": "county_ports"}))
 
    # Full county list from shapefile — ensures all counties are present
    all_counties = gpd.read_file(shapefile_path)[
        ["GEOID", "NAMELSAD", "STUSPS"]
    ].rename(columns={"GEOID": "county_geoid",
                       "NAMELSAD": "county",
                       "STUSPS": "state"})
    all_counties["county_geoid"] = (all_counties["county_geoid"]
                                    .astype(str).str.zfill(5))
 
    # Left join: all counties get a row; unmatched ones get NaN → 0
    county_agg["county_geoid"] = (county_agg["county_geoid"]
                                   .astype(str).str.zfill(5))
    full = all_counties.merge(county_agg, on=["county_geoid", "county", "state"],
                               how="left")
    full["county_ports"] = full["county_ports"].fillna(0)
 
    n_with    = (full["county_ports"] > 0).sum()
    n_without = (full["county_ports"] == 0).sum()
    print(f"  Counties with chargers:    {n_with:,}")
    print(f"  Counties without chargers: {n_without:,} (weight = 0)")
 
    # State totals and weights
    state_totals = (full.groupby("state")["county_ports"]
                        .sum()
                        .reset_index()
                        .rename(columns={"county_ports": "state_ports"}))
    full = full.merge(state_totals, on="state")
 
    # Weight = 0 for counties with no chargers; also 0 if entire state has none
    full["county_weight"] = np.where(
        full["state_ports"] > 0,
        full["county_ports"] / full["state_ports"],
        0.0,
    )
 
    # Sanity check: weights sum to 1 for states that have any chargers
    states_with_chargers = full[full["state_ports"] > 0]
    weight_check = states_with_chargers.groupby("state")["county_weight"].sum()
    max_err = (weight_check - 1.0).abs().max()
    assert max_err < 1e-6, f"Weights don't sum to 1 per state (max error: {max_err})"
 
    return full
 
 
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
 
 
# ── Mapping ────────────────────────────────────────────────────────────────────
 
def plot_feedstock_maps(feedstock_csv: str, shapefile_path: str,
                         ev_scenario: str, chem_scenario: str,
                         years: list = MAP_YEARS,
                         minerals: list = MAP_MINERALS,
                         out_dir: str = MAP_DIR):
    """
    Produce one PNG per mineral showing a 1×3 grid of choropleth maps
    (one column per year in `years`) of county-level feedstock in metric tonnes.
 
    Uses a log scale so both sparse rural counties and dense urban counties
    are readable in the same colour ramp. Counties with zero feedstock
    (no retirements yet in that year) are shown in light grey.
 
    Alaska and Hawaii are inset below the continental US.
    """
    os.makedirs(out_dir, exist_ok=True)
 
    # Load feedstock data
    df = pd.read_csv(feedstock_csv)
    df["county_geoid"] = df["county_geoid"].astype(str).str.zfill(5)
 
    # Load shapefile — reproject to Albers Equal Area for continental US display
    counties_geo = gpd.read_file(shapefile_path).to_crs("EPSG:5070")
    counties_geo["GEOID"] = counties_geo["GEOID"].astype(str).str.zfill(5)
 
    # Separate AK, HI, and CONUS for inset layout
    conus = counties_geo[~counties_geo["STATEFP"].isin(["02", "15", "60", "66", "69", "72", "78"])]
    alaska = counties_geo[counties_geo["STATEFP"] == "02"]
    hawaii = counties_geo[counties_geo["STATEFP"] == "15"]
 
    # Reproject AK and HI to local CRS for inset display
    alaska_disp = alaska.to_crs("EPSG:3338")   # Alaska Albers
    hawaii_disp = hawaii.to_crs("EPSG:26962")  # Hawaii zone
 
    for mineral in minerals:
        col = f"feedstock_t_{mineral}"
        colors = MAP_COLORS[mineral]
        cmap = mcolors.LinearSegmentedColormap.from_list(mineral, colors, N=256)
 
        # Global log-scale bounds across all 3 years (consistent colour scale)
        all_vals = []
        for yr in years:
            sub = df[df["year"] == yr]
            vals = sub[col].values
            all_vals.extend(vals[vals > 0].tolist())
        vmin = max(np.percentile(all_vals, 2), 1e-4)
        vmax = np.percentile(all_vals, 98)
        norm = LogNorm(vmin=vmin, vmax=vmax)
 
        fig = plt.figure(figsize=(18, 7))
        fig.patch.set_facecolor("#1a1a2e")
 
        # Title
        fig.text(0.5, 0.97,
                 f"EV Battery Feedstock — {mineral}  (metric tonnes)\n"
                 f"EV scenario: {ev_scenario}  |  Chemistry scenario: {chem_scenario}",
                 ha="center", va="top", fontsize=13, color="white", fontweight="bold")
 
        # 3 columns for the 3 years; leave bottom strip for AK/HI insets + colorbar
        col_w = 0.30
        col_starts = [0.02, 0.345, 0.67]
        main_bottom = 0.18
        main_height = 0.74
 
        # Compute CONUS bounds once
        minx, miny, maxx, maxy = conus.total_bounds
        aspect = (maxx - minx) / (maxy - miny)
 
        axes_main = []
        axes_ak   = []
        axes_hi   = []
 
        for i, yr in enumerate(years):
            sub = df[df["year"] == yr][["county_geoid", col]].copy()
            sub["county_geoid"] = sub["county_geoid"].astype(str).str.zfill(5)
 
            # Merge feedstock into geo
            merged_conus = conus.merge(sub, left_on="GEOID", right_on="county_geoid", how="left")
            merged_ak    = alaska_disp.merge(sub, left_on="GEOID", right_on="county_geoid", how="left")
            merged_hi    = hawaii_disp.merge(sub, left_on="GEOID", right_on="county_geoid", how="left")
 
            # ── Main CONUS panel ──
            ax = fig.add_axes([col_starts[i], main_bottom, col_w, main_height])
            ax.set_facecolor("#1a1a2e")
            ax.set_xlim(minx, maxx)
            ax.set_ylim(miny, maxy)
            ax.set_aspect("equal")
            ax.axis("off")
 
            # Zero / no-data counties
            no_data = merged_conus[merged_conus[col].isna() | (merged_conus[col] <= 0)]
            has_data = merged_conus[merged_conus[col] > 0]
 
            no_data.plot(ax=ax, color="#2a2a3e", linewidth=0.1, edgecolor="#3a3a4e")
            if len(has_data):
                has_data.plot(ax=ax, column=col, cmap=cmap, norm=norm,
                              linewidth=0.1, edgecolor="#1a1a2e", legend=False)
 
            ax.set_title(str(yr), color="white", fontsize=12, pad=4, fontweight="bold")
            axes_main.append(ax)
 
            # ── Alaska inset ──
            ak_x0 = col_starts[i]
            ak_ax = fig.add_axes([ak_x0, 0.01, col_w * 0.38, 0.16])
            ak_ax.set_facecolor("#1a1a2e")
            ak_ax.axis("off")
            no_data_ak = merged_ak[merged_ak[col].isna() | (merged_ak[col] <= 0)]
            has_data_ak = merged_ak[merged_ak[col] > 0]
            no_data_ak.plot(ax=ak_ax, color="#2a2a3e", linewidth=0.15, edgecolor="#3a3a4e")
            if len(has_data_ak):
                has_data_ak.plot(ax=ak_ax, column=col, cmap=cmap, norm=norm,
                                 linewidth=0.15, edgecolor="#1a1a2e", legend=False)
            axes_ak.append(ak_ax)
 
            # ── Hawaii inset ──
            hi_ax = fig.add_axes([ak_x0 + col_w * 0.40, 0.01, col_w * 0.28, 0.12])
            hi_ax.set_facecolor("#1a1a2e")
            hi_ax.axis("off")
            no_data_hi = merged_hi[merged_hi[col].isna() | (merged_hi[col] <= 0)]
            has_data_hi = merged_hi[merged_hi[col] > 0]
            no_data_hi.plot(ax=hi_ax, color="#2a2a3e", linewidth=0.15, edgecolor="#3a3a4e")
            if len(has_data_hi):
                has_data_hi.plot(ax=hi_ax, column=col, cmap=cmap, norm=norm,
                                 linewidth=0.15, edgecolor="#1a1a2e", legend=False)
            axes_hi.append(hi_ax)
 
        # ── Shared colorbar ──
        cbar_ax = fig.add_axes([0.25, 0.045, 0.50, 0.025])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
        cbar.set_label(f"{mineral} feedstock (metric tonnes, log scale)",
                       color="white", fontsize=10)
        cbar.ax.xaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.xaxis.get_ticklabels(), color="white", fontsize=8)
        cbar.outline.set_edgecolor("#555555")
 
        # Zero legend patch
        zero_patch = mpatches.Patch(color="#2a2a3e", label="No feedstock (retirements not yet due)")
        fig.legend(handles=[zero_patch], loc="lower right",
                   framealpha=0, fontsize=8,
                   labelcolor="white", bbox_to_anchor=(0.99, 0.01))
 
        out_path = os.path.join(out_dir,
                                f"feedstock_map_{mineral}_{ev_scenario}_{chem_scenario}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close()
        print(f"  Saved {out_path}")
 
 
# ── Main ───────────────────────────────────────────────────────────────────────
 
def main():
    # Assign counties via spatial join
    station_gdf = load_and_assign_counties(STATIONS_CSV, SHAPEFILE_PATH)
    county_weights = build_county_weights(station_gdf, SHAPEFILE_PATH)
 
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
 
        # Generate maps for this scenario combination
        print(f"  Mapping {ev_s}/{chem_s}...")
        plot_feedstock_maps(out_path, SHAPEFILE_PATH, ev_s, chem_s)
 
    print("\nDone.")
 
 
if __name__ == "__main__":
    main()
