# Data acquisition, preparation, and rights

This source candidate intentionally contains no NASA raw observations and no generated row-level dataset.

## Official source

The project uses NASA C-MAPSS FD001 from the Prognostics Center of Excellence repository:

- Repository: `https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/`
- Dataset record: `https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data`
- Official archive: `https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip`

Citation: A. Saxena and K. Goebel (2008), *Turbofan Engine Degradation Simulation Data Set*, NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, California.

## Required FD001 files

```text
train_FD001.txt
test_FD001.txt
RUL_FD001.txt
readme.txt
```

`run_analysis.py` extracts only those files. It does not alter any row.

## Private cache location

Set `CMAPSS_DATA_DIR` to an existing verified FD001 directory, or allow the runner to use the local default:

```text
../05_OUTPUTS/private_data_cache/FD001
```

When files are absent and network access is available, the runner downloads the official archive and checks its integrity before extracting FD001. When the private cache already exists, the workflow runs without network access.

The `data/external/` and `data/processed/` directories are intentionally empty staging boundaries. Raw or row-level data must not be added to the source candidate.

## Rights status

- `DATA_PROVENANCE = PUBLIC_VERIFIED`
- `LICENSE_STATUS = VERIFY_BEFORE_PUBLICATION`
- `REDISTRIBUTION_STATUS = VERIFY_BEFORE_PUBLICATION`

NASA's dataset record labels the record public but states `License not specified`. Public access is not treated as permission to redistribute. Do not copy, publish, commit, rehost, or bundle raw files until Module 8 records package-specific rights clearance.

Generated row- and asset-level derivatives also remain excluded pending Module 8. Aggregate figures and metrics require artifact-by-artifact review; they are not automatically cleared.

## Verified hashes

- Official outer archive SHA-256: `C9C5DEC12A945A82E8BB4446589D7FB3CC057B5E5D81FA1A12E25EE9912AD3B2`
- Nested `CMAPSSData.zip`: `74BEF434A34DB25C7BF72E668EA4CD52AFE5F2CF8E44367C55A82BFD91A5A34F`
- `train_FD001.txt`: `963B5E22825B34D8B21C69E1AEB4AF3E647050EB672EE8834BA4B5D91D2DE0F8`
- `test_FD001.txt`: `3CDA7109CE17BAFB5443F2AC926CFCF88154B941B8C4CF95EB55D1DDD6F52851`
- `RUL_FD001.txt`: `A19C8EC94931949D0485BDC35118206E9C81C4547B422EFB9CF86F4CEDDBCECA`
- `readme.txt`: `4F5270554B775C67E73AFF383C5436FD329D6E4CC3D3A116913276FAE511269B`

See [data_rights.md](../docs/data_rights.md) and [publication_artifact_policy.md](../docs/publication_artifact_policy.md).

