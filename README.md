# A Step Ahead of Drought: Forecasting Global Water Storage

First-place solution to the ITU / Zindi challenge *A Step Ahead of Drought: Forecasting Global
Water Storage*. Author: Nhat Vo (Zindi: nhatvo).

The model forecasts next-month Total Water Storage (TWS) for every land cell on a 1 degree grid,
1 to 7 months ahead. It is a blend of 14 base models: LightGBM regressors on the supplied
covariates and ERA5 reanalysis, MLP bags, simple statistical members and two small convolutional
networks over the global TWS field.

| | |
|---|---|
| Public RMSE | 0.662143 |
| Private RMSE | 0.700299 (leaderboard rank 6; first place overall after the report phase) |
| Scored file MD5 | `72e6ab359311a9ea4fa019947a928503` |

## Documents

- Full solution documentation: [`submission_model/README.md`](submission_model/README.md)
- Trustworthiness report: [`report/PHASE2_REPORT.md`](report/PHASE2_REPORT.md)
- Technical report, including negative results: [`report/REPORT.md`](report/REPORT.md)

## Repository layout

| Path | Contents |
|---|---|
| `src/tws/` | Data loading, features, folds, metrics, ERA5 and GLDAS readers |
| `scripts/` | Training and data download scripts |
| `research/` | The model adapter and the JSON configs the solution reads |
| `submission_model/` | Member recipe, blend replay, rebuild scripts, member code snapshots |
| `report/` | Reports and their artefacts (bias tables, SHAP, emissions) |

## Run

1. Put the competition files `Train.csv` and `Test.csv` in `data/`. They are not in this
   repository; download them from the challenge page.
2. Install Python 3.12 and the packages:

       python3.12 -m venv .venv
       .venv/bin/pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu132
       .venv/bin/python scripts/fetch_era5.py      # needs a free Copernicus CDS account

3. Train every member and build the final file (about 5 hours on 16 CPU cores and one RTX 3070):

       submission_model/run_all.sh

The output is `submission_model/output/scratch/final_dart.csv`. LightGBM training is not bit-stable, so this gives a file very close to the scored file
(RMS difference 0.0039), not a byte-identical one.

## Notes

- The archived member predictions (`submission_model/members/*.csv`) are not included, because
  they are derived from the competition test data. `submission_model/members/SHA256SUMS` lists
  their checksums. With these files, `submission_model/rebuild_final.py` rebuilds the scored file
  exactly in about one minute (path A in `submission_model/README.md`, section 10).
- Some documents and code comments refer to `NOTES.md` sections or to `research/` scripts. These
  are the private working log and exploration scripts, which are not published. The reports
  summarise their results.
- The blend weights were solved from public leaderboard scores. `submission_model/README.md`
  section 1 describes this.
