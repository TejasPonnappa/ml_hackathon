# Section 2 notes — Dataset understanding and EDA (Person A)

All numbers come from files produced by `python -m src.ml.choose_kr` and
`python -m src.ml.eda` (tables `results/a_eda_*.csv`, `results/a_eda_summary.json`,
plots `results/a_eda_*.png`, log `results/logs/a_eda.log`).
**Partitions used:** train (80 trips) and student_test (10 trips) for every analysis
unless stated otherwise. The evaluator-test partition was never accessed.

## 2.1 How the raw data is organised
- One folder per trip, four header-less whitespace files: `.track` (lon, lat, t),
  `.nodes` (lon, lat; row = node_id), `.arcs` (from_node, to_node; row = arc_id),
  `.route` (arc_id per row, chronological). Parsed dtypes: `results/a_eda_dtypes.csv`.
- **No missing files** in either partition, **0 NaN values**, 0 out-of-range
  coordinates, 0 arcs pointing to non-existent nodes, 0 route entries pointing to
  non-existent arcs (`a_eda_inventory.csv`, `a_eda_quality.csv`, plot `a_eda_missing_invalid.png`).
- A GPS row does not correspond to a network row: the link between GPS
  observations and arcs has to be inferred (that is the task).
- **Each trip has its own road network**, and networks are large:
  train median 9,608 nodes / 19,527 arcs, max 307,392 arcs
  (student_test median 6,597 nodes / 13,145 arcs). Plot: `a_eda_network.png`.

## 2.2 Trajectories and sampling (train / student_test)
- Observations: train 208,417 total, median 1,529 per trip (min 361, max 19,313);
  student_test 19,160 total, median 1,420 (min 262, max 3,958). Plot `a_eda_trip_sizes.png`.
- Timestamps strictly increase in every trip (0 non-monotonic trips).
- Sampling: **96.5 % of train steps and 98.6 % of student_test steps are exactly 1 s**;
  2.1 % of train steps are 2 s. Train has 1,216 steps > 5 s and 28 trips with a
  gap > 30 s; the largest train gap is 153,064 s (~42 h), i.e. one trip contains two
  recordings. student_test: 44 steps > 5 s, max 210 s. Plot `a_eda_time_gaps.png`.
- **Geographic extent: the trips are worldwide** (train trip centres span longitude
  -100 to +137, latitude -35 to +61: Europe, Russia, USA, Mexico, Brazil, Argentina,
  South Africa, Japan, Philippines, Vietnam...). Plot `a_eda_trip_locations.png`.
  Track length: train median 24.0 km (max 169.6 km); median speed per trip 12.7 m/s
  (≈46 km/h). Some trips contain GPS jumps > 60 m/s (`a_eda_extent.csv`).

## 2.3 Coordinate system decision
- Longitude/latitude are angular, so degree-Euclidean distance is invalid (at 55°N one
  degree of longitude ≈ 64 km but one degree of latitude ≈ 111 km).
- First assumption (single UTM zone 32N, from train trip 00000001 near Darmstadt) was
  **wrong**: it is only valid between 6°E and 12°E and produced infinite values for most trips.
- **Decision:** a per-trip local azimuthal-equidistant projection centred on the median
  GPS position of the track (uses the track only, so it is valid in the prediction
  pipeline). Check: projected step lengths vs haversine differ by at most 0.56 % (train)
  over all consecutive GPS pairs (`proj_vs_haversine_max_rel_err` in `a_eda_extent.csv`).

## 2.4 Road network structure (`a_eda_network.csv`)
- Arcs are **directed**: on average 92 % of arcs have a reverse twin (two-way roads stored
  as u→v and v→u). Consequence: two candidate arcs often lie at *identical* distance from a
  GPS point and only the direction of travel can separate them.
- Median arc length ≈ 20 m (trip 00000001), so at ~13 m/s a vehicle crosses roughly one arc
  every 1–2 s, sometimes several between two GPS fixes.
- 0 isolated nodes. Networks split into several weakly connected components (train median 20,
  max 421) — these are small fragments at the edge of the extracted area.
- Minor defects: 1,775 duplicate arcs (same from/to), 22 self-loops, 100 zero-length arcs in train.

## 2.5 Reference route (`a_eda_route.csv`)
- The route has far fewer rows than the track: median 0.19 route arcs per GPS observation (train).
- Consecutive route arcs are connected (to_node = next from_node) in 70 of 80 train trips
  and 9 of 10 student_test trips; the remaining trips have a few breaks.
- Route length is ~84 % of the GPS track length (median), mainly because the route does not
  always cover the start/end of the recording (e.g. train 00000002 in `a_eda_trajectories.png`).
- **Labels:** we align each GPS observation to a route arc with a monotone dynamic program
  (`src/common/labels.py`): the assigned position along the route may never decrease, and the
  total GPS-to-arc distance is minimised. Resulting GPS-to-reference distance: median ≈ 5 m,
  90th percentile 13.6 m on train after removing the two anomalous trips (`a_eda_gps_noise.png`).

## 2.6 Anomalies and data-quality decisions (`a_eda_anomalies.csv`)
| Issue | Evidence | Decision |
|---|---|---|
| Train 00000054, 00000033: route does not match the track | median GPS-to-route distance 140 m / 122 m vs ≈5 m elsewhere; most of the 19,313 points of trip 54 are a parked vehicle away from the route loop | excluded from model fitting (kept for EDA) |
| Observations > 50 m from their reference arc | label unreliable / outside candidate radius | excluded from training rows only; student_test is evaluated on **all** observations |
| Time gaps > 30 s (28 train trips) | `a_eda_time.csv` | kept; neighbour features use `dt_prev`, the HMM (Person B) breaks/penalises transitions over gaps |
| GPS jumps > 60 m/s | `a_eda_extent.csv` | kept; tree models are robust to outlier speeds |
| Duplicate / zero-length arcs | `a_eda_network.csv` | kept; duplicates make an exact `arc_id` label slightly ambiguous (noted as a label limitation) |

The reference route is treated as reference information, not verified ground truth:
labels near junctions and at route breaks are uncertain.

## 2.7 Candidate generation (`a_eda_kr_grid.csv`, `a_eda_candidate_miss.png`)
- For each GPS point: the K nearest arcs within R metres (KD-tree over points sampled every 10 m along arcs).
- Miss rate (reference arc not among candidates), train excluding anomalous trips:
  K=10, R=50 m → **1.66 %**; student_test → **0.26 %**. K=16, R=100 m would only reduce
  the train miss rate to 0.78 % while making the table ~60 % larger. **Chosen: K=10, R=50 m.**
- Reference arc rank among candidates (train): rank 0 49.8 %, rank 1 32.5 % — the
  rank-0/rank-1 split is mostly the two-way twin tie.

## 2.8 Features extracted and why → see `src/ml/features.py` docstring (19 features).
Key ones: distance to arc, **heading difference between motion and arc direction** (breaks
two-way ties), distance of neighbouring GPS points to the same arc, rank / gap to the nearest
candidate, speed, position along the arc, arc length, node degrees.

## Folium maps
`src/viz/eda_folium.py` → `results/a_eda_folium_train_00000001.html`,
`a_eda_folium_train_00000054.html` (anomalous trip), `a_eda_folium_student_test_00000007.html`.
Layers: raw GPS (blue), road network within 200 m of the track (grey), reference route (green),
start/end markers, layer control and legend.
