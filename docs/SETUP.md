# Setup and run guide

This guide takes you from an empty machine to the final submission file. For the method itself,
read `submission_model/README.md`.

## 1. What you need

| Item | Notes |
|---|---|
| Linux machine | Tested on Ubuntu 24.04. 16 CPU threads and 32 GB RAM are enough. |
| NVIDIA GPU | Only for the two small CNN members. Tested on an RTX 3070 (8 GB), CUDA 13.2. |
| Python 3.12 | Tested on 3.12.3. |
| Disk space | About 2 GB for the data, the ERA5 files and the outputs. |
| Competition data | `Train.csv` and `Test.csv` from the challenge page. |
| Copernicus CDS account | Free. Needed to download ERA5. |

## 2. Get the code and the data

```bash
git clone <this repository> itu-drought-tws
cd itu-drought-tws
mkdir -p data
# Copy Train.csv and Test.csv from the challenge page into data/
```

## 3. Install the packages

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu132
```

The versions in `requirements.txt` are the ones that produced the submission. Other versions can
change the results slightly.

## 4. Download ERA5

ERA5 is a public climate reanalysis from the Copernicus Climate Data Store (CDS). The solution
uses 9 monthly fields on a 1 degree grid, 2002 to 2018 (about 124 MB).

1. Create a free account at https://cds.climate.copernicus.eu.
2. Open the dataset page for `reanalysis-era5-single-levels-monthly-means`, go to the
   download tab, and accept the licence (CC-BY). Without this step every request fails with
   `403 Forbidden: required licences not accepted`.
3. Copy your API key from your CDS profile page into `~/.cdsapirc`:

       url: https://cds.climate.copernicus.eu/api
       key: <your API key>

   Then run `chmod 600 ~/.cdsapirc`.
4. Download the fields:

       .venv/bin/python scripts/fetch_era5.py          # 2002-2018, into data/era5/
       .venv/bin/python scripts/fetch_era5.py --snow   # snow depth, into data/era5/

The script skips years that are already on disk, so you can run it again after a failure.

## 5. Run the full pipeline

```bash
submission_model/run_all.sh
```

This trains all 14 base members and then runs the 19 blend steps in order. It takes about
5 hours (measured: 280 minutes). The final file is:

    submission_model/output/scratch/final_dart.csv

Other outputs:

| Path | Contents |
|---|---|
| `submission_model/output/rebuilt/` | The retrained base members |
| `submission_model/output/logs/` | One log file per member |
| `submission_model/output/rebuild_members_report.json` | Run time of each member |

LightGBM training is not bit-for-bit stable between runs. The final file is therefore very
close to the scored file (RMS difference about 0.004), but not identical.

## 6. Run part of the pipeline

Retrain only some members:

```bash
.venv/bin/python submission_model/rebuild_members.py eof fieldcnn
```

The member names are in `submission_model/members.json`. Each member runs from a copy of the
code at the commit that produced it (`submission_model/snapshots/<commit>/`).

Rebuild the final file from member files that you already have:

```bash
.venv/bin/python submission_model/rebuild_final.py --members <folder with the member CSVs>
```

With the original archived members (not in this repository, see the main `README.md`), this
gives the scored file exactly, MD5 `72e6ab359311a9ea4fa019947a928503`, in about one minute.

## 7. Report figures

```bash
.venv/bin/python scripts/report_artifacts.py
```

This writes the bias tables and the SHAP figure and table to `report/artifacts/` (about
25 minutes).

## 8. Common problems

| Problem | Fix |
|---|---|
| `403 Forbidden: required licences not accepted` from CDS | Accept the licence on the dataset page (step 4.2). |
| `FileNotFoundError: data/Test.csv` | Copy the competition files into `data/`. |
| A member fails | Read its log in `submission_model/output/logs/<member>.log`. |
| CUDA errors in `fieldcnn` or `fielddirect3` | Check that the PyTorch build matches your CUDA driver. |
