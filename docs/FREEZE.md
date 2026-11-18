# Freeze record

Generated 2026-11-18T12:06:53 on the `core-ml` branch, before the evaluator
test folder was received. Nothing below may change after the evaluator data arrives; the API only
points `PATHSNAP_EVAL_DIR` at that folder.

- Code commit at freeze time: `032b02ccd18ec16fc1e4b018c4a9fcc5d752c9a3`
- Final bundle: `artifacts/pathsnap_model.joblib`  SHA-256 `af093e19ce884a0060b1e2827de318b1eaad84eafb7516139d6e29ff0c87766c`
- Model: `lightgbm_tuned`; trained on train split, 78 trips, 1,429,277 candidate rows
- Model parameters: `{"n_estimators": 800, "learning_rate": 0.020297676378643766, "num_leaves": 250, "min_child_samples": 94, "subsample": 0.8199094425189012, "subsample_freq": 1, "colsample_bytree": 0.957522308602883, "reg_lambda": 2.1822435418877868e-08, "reg_alpha": 2.7440553357911863}`
- Candidates: radius 50.0 m, at most 16 per point; label cutoff 50.0 m
- HMM decoder: `{"beta": 20.0, "emission_weight": 1.0, "unreachable_logp": -40.0, "sp_limit": 1500.0, "gap_dt": 60.0, "gap_dist": 1000.0, "backward_tol": 15.0}`
- Features (29): `dist, t_on_arc, beyond_end_m, arc_len, cos_heading, abs_dheading, heading_valid, speed, cos_heading_long, heading_long_valid, cos_gap_long, dist_ctx-5, dist_ctx-2, dist_ctx-1, dist_ctx+1, dist_ctx+2, dist_ctx+5, dist_ctx_mean, dist_ctx_max, has_twin, in_deg_u, out_deg_v, log_dt_prev, log_dt_next, jitter, n_cands, dist_gap, ctx_gap, cos_gap`
- Seed: 352
- Data used for any fitting, tuning or selection: `train` (80 trips, 78 with usable references) and
  `student_test` (10 trips, comparison only). The evaluator partition was never available.

## Student-test rehearsal (full pipeline, exactly as the API runs it)

- Point-weighted Top-1: 0.9472
- Mean sequence F1: 0.9850; mean coverage F1: 0.9875
- Mean route continuity: 0.9943
- Prediction wall time per trip: max 2250 ms; scoring max 410 ms

| trip | points | Top-1 | sequence F1 | coverage F1 | continuity | predict ms |
|---|---|---|---|---|---|---|
| 00000000 | 2503 | 0.9588 | 0.9957 | 0.9957 | 0.9888 | 909 |
| 00000007 | 1354 | 0.9815 | 0.9872 | 0.9872 | 0.9918 | 484 |
| 00000009 | 1049 | 0.8837 | 0.9635 | 0.9667 | 0.9894 | 811 |
| 00000040 | 2995 | 0.9773 | 0.9971 | 0.9985 | 0.9934 | 1160 |
| 00000052 | 262 | 0.9847 | 0.9984 | 0.9984 | 1.0000 | 198 |
| 00000056 | 3958 | 0.9616 | 0.9751 | 0.9758 | 0.9975 | 2250 |
| 00000065 | 768 | 0.9245 | 0.9863 | 0.9863 | 0.9924 | 619 |
| 00000076 | 1486 | 0.9744 | 0.9899 | 0.9899 | 0.9971 | 1564 |
| 00000077 | 3911 | 0.8998 | 0.9647 | 0.9837 | 0.9929 | 2210 |
| 00000091 | 874 | 0.9428 | 0.9924 | 0.9924 | 1.0000 | 679 |

## Every saved model (for the UI history view)

| file | SHA-256 |
|---|---|
| `artifacts/models/catboost.joblib` | `15e451190ca318b8d5e68e1d9eee4152fb0d5c4d35afdc9bbedd45977d0ae075` |
| `artifacts/models/distance_gaussian.joblib` | `4a41e5d90fee65a8e8ea65d4230336a81a3ee908c5bef341190909a5a6857951` |
| `artifacts/models/gaussian_nb.joblib` | `d157f74db93cf0a1c1768b9a21aad792618cf8efb2c9f67eb5b100685cb47a15` |
| `artifacts/models/knn.joblib` | `a40098517f188b1f065bf5d5b9a84b9f6968bd6ad14bc42656b106cc174597e3` |
| `artifacts/models/lightgbm.joblib` | `846a341a329632036c0e4c49f5df686d6cd9d8c672e3e5041e134f2bb3d7bc44` |
| `artifacts/models/lightgbm_tuned.joblib` | `4b4a995aa5a0d19512cd534fae7881d78cc6c6c5fab83e9447eda5af2b407e6c` |
| `artifacts/models/lightgbm_v1feats.joblib` | `e52323afb50b81f2daa166b1185ad9be013da86bb1540a8455db8a2192aa6a19` |
| `artifacts/models/logreg.joblib` | `3b7d6d1afddcec9f4bdd2c5576c33ac14f84623e55278df5a03b3756a7335638` |
| `artifacts/models/mlp_torch.joblib` | `0698d4ab859cb807e9d0ea5308108db625fc64275eff42768e7d588210ad0786` |
| `artifacts/models/random_forest.joblib` | `66915ada09e36645c5b404615f3cf501fe29e51ebb44d92f529ba66a340fe728` |
| `artifacts/models/ranker_torch.joblib` | `712704017ca4723c3579850f0b73f53fe67a6d51a78060fee781935fd7d21e56` |
| `artifacts/models/xgboost.joblib` | `a8e6d4bbba6914d40050d2015bf2f5da228039a09de2f618add484b8aa769c8c` |
