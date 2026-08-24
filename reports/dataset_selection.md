# Dataset selection

## Decision

Select **NASA C-MAPSS Turbofan Engine Degradation Simulation, FD001** as the one primary benchmark. Use no secondary dataset in Module 7.25D.

| Candidate | Provenance | License_Status | Redistribution_Status | Asset_Structure | Temporal_Structure | Future_Target_Suitability | Decision | Reason |
|---|---|---|---|---|---|---|---|---|
| NASA C-MAPSS FD001 | NASA Ames Prognostics Center of Excellence; Saxena & Goebel (2008) | Official record says `License not specified` | `VERIFY_BEFORE_PUBLICATION` | 100 development and 100 official-test simulated engines | Ordered per-engine cycles; development runs to EOL; test truncates before EOL with terminal RUL truth | High: independently define simulated EOL within H and validate on unseen engines | **SELECT** | Best match to future-event, lead-time, grouped-validation, and reproducibility requirements |
| UCI Condition Monitoring of Hydraulic Systems | Helwig, Pignanelli & Schuetze; experimental hydraulic test rig | CC BY 4.0 | `ALLOWED` with attribution | 2,205 controlled cycles from one rig, not an asset fleet | Repeated 60-second test cycles under deliberately varied component states | Low for the primary question: contemporaneous health labels, no natural run-to-failure endpoint | Reject as primary | Strong rights and measured signals, but cannot support the required asset-wise future-event design |
| UCI MetroPT-3 | Davari, Veloso, Ribeiro & Gama; real metro APU/compressor | CC BY 4.0 | `ALLOWED` with attribution | Conservatively one asset; 1,516,948 rows; four reported air-leak intervals | Timestamped February-August 2020 stream | Medium for anomaly/lead-time study; weak event/generalization sample | Reject as primary | Real operational data but no multi-asset split and only four reported events |

## Selected-data facts

- FD001 has one sea-level operating condition and one documented high-pressure-compressor degradation mode.
- Each row has unit ID, cycle, three operating settings, and 21 anonymized sensor channels.
- Development: 100 complete engine trajectories and 20,631 observations.
- Official test: 100 truncated engine trajectories and 13,096 observations.
- Total analytical scope: 200 namespaced assets and 33,727 observations.
- The official test truth supplies remaining cycles after the last observed row, enabling row-level target and event-cycle reconstruction without using RUL as a predictor.

## Rights record

- `DATA_PROVENANCE = PUBLIC_VERIFIED`
- `LICENSE_STATUS = VERIFY_BEFORE_PUBLICATION`
- `REDISTRIBUTION_STATUS = VERIFY_BEFORE_PUBLICATION`
- Official source: [NASA PCoE Data Set Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)
- Dataset metadata: [NASA CMAPSS Jet Engine Simulated Data](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data)
- Official archive: [NASA PCoE download](https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip)
- Archive SHA-256: `C9C5DEC12A945A82E8BB4446589D7FB3CC057B5E5D81FA1A12E25EE9912AD3B2`
- Inner archive SHA-256: `74BEF434A34DB25C7BF72E668EA4CD52AFE5F2CF8E44367C55A82BFD91A5A34F`

Public access is verified, but public access is not treated as an explicit redistribution license. Raw files are a local, ignored runtime cache only. Publication instructions download from NASA and acknowledge the dataset; they do not rehost the archive.

## Selection principle

The decision is based on target legitimacy, independent-asset structure, temporal ordering, event/lead-time evaluation, official holdout discipline, and reproducibility—not on which dataset yields the highest score.
