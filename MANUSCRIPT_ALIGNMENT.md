# Manuscript-to-artifact alignment

This release is aligned with the revised manuscript **“Advanced Drone Sensing Framework for Smart X Situational Intelligence.”**

| Manuscript element | Release artifact |
|---|---|
| Controlled mission: 160 records, 114 normal, 46 anomalies | `data/simulated_drone_sensing.csv`, `data/image_features.csv` |
| Two Isolation Forest branches, 200 estimators, contamination 0.10, seeds 12/24 | `model_core.py` (`ModelConfig`, `compute_anomaly_scores`) |
| Pollutant-weather and image-feature rule scores | `model_core.py` |
| Ensemble alpha = 0.60 and thresholds 0.45/0.60/0.75 | `model_core.py` (`ModelConfig`) |
| Baseline metrics and confusion counts | `outputs/baseline_metrics.json` |
| Scenario-level detection (Table 6) | `outputs/scenario_detection.csv` |
| Alpha sensitivity and modality ablation (Table 7) | `outputs/alpha_sensitivity.csv`, `outputs/modality_ablation.csv` |
| Controlled-noise robustness (Table 8) | `outputs/noise_robustness.csv` |
| Scored mission timeline / Figure 6 source values | `outputs/baseline_scored_mission.csv` |
| Reproducible execution | `experiment_runner.py`, `requirements-lock.txt` |
| Human validation / audit support | `app.py`, `logs/` |

## Important boundary

The archive supports the controlled synthetic experiment and Streamlit prototype. It does not implement production Digital Twin, Data Lakehouse, or distributed edge-cloud services, and it does not support field-performance claims.
