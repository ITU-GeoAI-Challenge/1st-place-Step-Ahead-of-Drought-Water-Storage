"""Pull the ERA5 monthly fields used by the solution (setup steps: docs/SETUP.md).

Ruling #34403 permits external non-TWS covariates provided every source date is at or before
`t` for the row using it. That constraint is NOT enforced here -- this script only fetches
raw monthly fields. It is enforced downstream, in the feature builder, exactly as it already
is for the supplied covariates: a row at month t may read months <= t and never t+1.

Fetches one year per request so the cache is resumable and each request stays inside CDS
limits. Re-running skips years already on disk.

    python scripts/fetch_era5.py             # 2002-2018
    python scripts/fetch_era5.py 2015 2016   # a subset
    python scripts/fetch_era5.py --snow      # just snow_depth, into era5snow_YYYY.nc
"""
from __future__ import annotations

import pathlib
import sys
import cdsapi

OUT = pathlib.Path("data/era5")
DATASET = "reanalysis-era5-single-levels-monthly-means"

# The water balance: TWS is the running integral of (precipitation - evaporation - runoff),
# plus what is held as snow and in the soil column. The competition supplies SPEI and soil
# moisture, which are the OUTCOME of that balance; these are the forcing terms.
VARIABLES = [
    "total_precipitation",
    "evaporation",
    "runoff",
    "2m_temperature",
    # NOT "snow_depth_water_equivalent" -- that name does not exist for this dataset and
    # CDS silently DROPS an unknown variable rather than erroring, so the first pull came
    # back with 8 of 9 fields and no warning. ERA5's `snow_depth` (short name `sd`) is
    # already in metres of water equivalent, which is what we want.
    "snow_depth",
    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",
    "volumetric_soil_water_layer_3",
    "volumetric_soil_water_layer_4",
]

YEARS = [str(y) for y in range(2002, 2019)]
MONTHS = [f"{m:02d}" for m in range(1, 13)]


def request_for(year: str) -> dict:
    return {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": VARIABLES,
        "year": [year],
        "month": MONTHS,
        "time": ["00:00"],
        "data_format": "netcdf",
        "download_format": "unarchived",
        # 1 degree matches the competition grid exactly, so cells are sampled with no
        # regridding error. lat/lon are used to SAMPLE this grid, never as features.
        "grid": [1.0, 1.0],
    }


def main() -> None:

    """Download the missing ERA5 years into data/era5/, one request per year.

    Pass years as arguments to limit the download, or --snow to fetch only snow depth.
    """
    args = sys.argv[1:]
    snow_only = "--snow" in args
    args = [a for a in args if a != "--snow"]
    years = args or YEARS
    prefix, variables = ("era5", VARIABLES)
    if snow_only:
        # Backfill for the field the first pull silently dropped. Written to its own files
        # so the eight already on disk are not refetched; the loader globs both prefixes.
        prefix, variables = "era5snow", ["snow_depth"]

    OUT.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()
    todo = [y for y in years if not (OUT / f"{prefix}_{y}.nc").exists()]
    print(f"{len(variables)} variables x {len(years)} years; {len(todo)} still to fetch")

    for y in todo:
        target = OUT / f"{prefix}_{y}.nc"
        partial_download_path = target.with_suffix(".nc.part")
        print(f"  {y} ...", flush=True)
        req = request_for(y)
        req["variable"] = variables
        client.retrieve(DATASET, req, str(partial_download_path))
        partial_download_path.rename(target)
        print(f"  {y} done  {target.stat().st_size/1e6:.1f} MB", flush=True)

    have = sorted(OUT.glob(f"{prefix}_*.nc"))
    total = sum(f.stat().st_size for f in have) / 1e6
    print(f"\n{len(have)} files, {total:.0f} MB in {OUT}")


if __name__ == "__main__":
    main()
