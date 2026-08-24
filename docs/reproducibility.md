# Reproducibility

## Tested environment

Module 7.5D validation and Module 7.75D professionalization used:

- Windows, 64-bit Python 3.12.13;
- NumPy 2.3.5;
- pandas 3.0.1;
- scikit-learn 1.7.2;
- Matplotlib 3.10.9;
- SciPy 1.16.3;
- pytest 9.1.1; and
- nbformat 5.11.1.

`requirements.txt` defines supported ranges. `constraints-tested.txt` records the exact validated environment; deterministic byte identity is environment-conditional.

## Environment setup

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -c constraints-tested.txt
```

## NASA data acquisition

The required FD001 files are:

```text
train_FD001.txt
test_FD001.txt
RUL_FD001.txt
readme.txt
```

The source candidate does not bundle them. `run_analysis.py` uses the directory in `CMAPSS_DATA_DIR` when that environment variable is set. Otherwise it uses:

```text
../05_OUTPUTS/private_data_cache/FD001
```

If verified files are absent, the runner can obtain the official NASA archive, extract only FD001, and enforce the recorded SHA-256 hashes. Network access is therefore optional when a verified private cache already exists. Detailed rights and hashes are in [data/README.md](../data/README.md).

## One-command workflow

```powershell
.\.venv\Scripts\python.exe run_analysis.py
```

The command:

1. verifies/acquires the private FD001 cache;
2. runs the focused test suite;
3. validates schema, ordering, keys, and source hashes;
4. constructs the 30-cycle target and trailing features;
5. runs grouped development, nested calibration review, and fixed model selection;
6. fits the selected model on development data;
7. applies the frozen threshold to the official test;
8. calculates row/event/FMEA scenario results; and
9. writes reports, internal tables, figures, and `run_summary.json` under `../05_OUTPUTS`.

Typical execution in the validated environment is a few minutes, dominated by grouped Random Forest fits, nested calibration diagnostics, bootstrapping, and figure generation. Fixed seed: `20260824`; seeds are not searched.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

Expected validated result: `74 passed, 36 subtests passed`.

## Notebooks

Run the one-command workflow first. Then execute, in order:

1. `notebooks/01_data_and_failure_context.ipynb`;
2. `notebooks/02_predictive_modeling.ipynb`; and
3. `notebooks/03_condition_informed_fmea.ipynb`.

The notebooks are explanatory clients. They do not own hidden selection/model logic and are not required by the build.

## Output and publication boundary

Generated outputs remain outside the source candidate under `../05_OUTPUTS`. Aggregate figures/tables are only candidates for later review. Row/asset-level NASA derivatives, raw files, runtime environments, validation scratch material, and local caches are excluded from public export. See [publication_artifact_policy.md](publication_artifact_policy.md).

