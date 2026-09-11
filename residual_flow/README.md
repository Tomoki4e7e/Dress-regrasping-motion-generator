# Residual Flow Experiment

This package keeps ShareSet read-only and adds a bounded residual policy around
the frozen SAMDAMSARNN base policy.

Implemented comparisons:

1. deterministic temporal residual regression;
2. conditional flow matching over residual action chunks;
3. force-aware conditional flow matching over residual and future
   `external_torque` chunks.

All three use the same compact image encoder and GRU history encoder. Training
statistics are fitted on train episodes only. Action residual and external
torque are normalized separately.

## Current data audit

Run:

```bash
cd dress_regrasping
python3 -m residual_flow.audit \
  --data-root ../ShareSet/share_folder/data \
  --manifest ../ShareSet/share_folder/data/dataset_sock.yaml \
  --output artifacts/data_audit.json
```

The checked-in audit records the current state of
`data_center_general_depth_n5_sock`. Its 11 episodes have aligned 18-column
`angle.csv`, `torque.csv`, and `external_torque.csv`, but currently have:

- no `depth_mask/leg_depth` frames;
- no generated `features/` packages;
- no `touch.csv` (expected; `external_torque.csv` is used instead).

The referenced sock model log also has `data.json` and `parameter.json`, but no
`.pth` checkpoint is present in the workspace. Consequently, a scientifically
valid cross-fitted residual dataset and real-data model comparison cannot be
run until the missing leg masks/features and held-out base checkpoints are
provided. The tools fail explicitly instead of filling these inputs.

## Base predictions and cross-fitting

Create a JSON object mapping each episode name to a checkpoint trained without
that episode. At least two distinct checkpoints are required by
`--require-crossfit`.

```bash
python3 -m residual_flow.base_predictions \
  --data-root ../ShareSet/share_folder/data \
  --manifest ../ShareSet/share_folder/data/dataset_sock.yaml \
  --shareset-src ../ShareSet/share_folder/src \
  --stats ../ShareSet/share_folder/log/model_SARNN/SARNN_20260113_2237_data_center_general_depth_n5_sock/data.json \
  --checkpoint-map crossfit_checkpoints.json \
  --require-crossfit \
  --output-root artifacts/predictions
```

Predictions are teacher-forced while the SAMDAMSARNN recurrent state is carried
through each episode. The residual target is:

```text
recorded_angle[t + 1] - base_prediction[t]
```

## Model comparison

```bash
python3 -m residual_flow.run_comparison \
  --data-root ../ShareSet/share_folder/data \
  --manifest ../ShareSet/share_folder/data/dataset_sock.yaml \
  --prediction-root artifacts/predictions \
  --output artifacts/runs/default \
  --history 10 --horizon 4 --epochs 100
```

The output contains all three checkpoints, train-only normalization, and
normalized test metrics including a zero-residual baseline. Use episode-level
splits; never split overlapping windows at random.

## Runtime safety adapter

`ResidualController`:

- samples several flow candidates;
- ranks future-force and residual-deviation costs;
- executes only the first residual;
- limits per-cycle magnitude and rate;
- preserves gripper mimic-joint coupling;
- disables residual output at a configured hard external-torque norm.

It is a pure adapter and does not edit or publish from ShareSet. A ROS wrapper
must retain the existing joint/workspace/keepout checks and independent
impedance, pause, and retreat behavior.

## Anomaly replay check

Save held-out future-force predictions and observations as an NPZ containing
`predicted` and `observed`, then run:

```bash
python3 -m residual_flow.validate_anomaly \
  --input force_predictions.npz \
  --output artifacts/anomaly.json
```

Injected spikes validate detector mechanics only. They do not establish
real-world snag/slip detection or closed-loop force reduction. Those claims
require labelled perturbation and robot trials.

## Tests

```bash
python3 -m pytest -q
```
