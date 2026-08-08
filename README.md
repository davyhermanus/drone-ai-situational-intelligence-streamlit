
## Experimental v1.3.0 candidate (not yet archived)

- Revises the synthetic mission generator to include weak/moderate/strong event intensity, partial modality conflict, and limited overlap with elevated normal background conditions.
- Preserves the 160-record mission and the original three event windows.
- Fits the baseline scaler and Isolation Forest models once and reuses them for perturbation testing.
- Robustness at 5%, 10%, and 20% is summarized across 30 independent Gaussian feature-noise realizations rather than a single random seed.
- The revision is intended to avoid trivially separable synthetic labels; it does not target any predetermined accuracy or F1 score.

# Advanced Drone Sensing Framework for Smart X Situational Intelligence
## Streamlit Reproducibility Package — Experimental Version 1.3.0

This release is the reproducibility package associated with the revised manuscript **“Advanced Drone Sensing Framework for Smart X Situational Intelligence.”** It preserves the controlled synthetic experiment and Streamlit prototype used to support the quantitative results reported in the paper.

## Scope of this release

The package contains **controlled synthetic data and generated sample images only**. It does **not** contain field measurements and must not be interpreted as field validation, deployment certification, or evidence of operational accuracy in real missions.

The implemented prototype demonstrates a closed sensing-to-decision workflow with:

1. **Pollutant-weather anomaly branch**
   - ten standardized features: PM2.5, PM10, CO2, NO2, SO2, VOC, temperature, humidity, wind speed, and altitude;
   - a 200-tree Isolation Forest with contamination 0.10 and random seed 12;
   - a transparent pollutant rule score;
   - branch fusion using 0.72 Isolation-Forest evidence and 0.28 rule evidence.

2. **Image-feature anomaly branch**
   - six lightweight descriptors: red index, smoke index, brightness, contrast, texture, and edge density;
   - a 200-tree Isolation Forest with contamination 0.10 and random seed 24;
   - a transparent visual rule score;
   - branch fusion using 0.70 Isolation-Forest evidence and 0.30 rule evidence.

3. **Ensemble decision layer**
   - baseline pollutant-weather weight alpha = 0.60;
   - operational thresholds: Watch = 0.45, Warning = 0.60, Critical = 0.75;
   - Normal, Watch, Warning, and Critical status mapping;
   - recommendation generation and persistent human-validation records.

## Controlled mission

The frozen controlled mission contains **160 records at two-minute intervals (318 minutes)**:

- 114 normal records;
- 18 pollution-plume anomaly records;
- 16 fire-smoke anomaly records;
- 12 industrial-dust anomaly records.

The experiment evaluates agreement with these known synthetic labels. It does not report field-validated performance.

## Quantitative outputs

Running `experiment_runner.py` regenerates the paper-supporting outputs in `outputs/`:

- `baseline_metrics.json`
- `baseline_scored_mission.csv`
- `scenario_detection.csv`
- `alpha_sensitivity.csv`
- `modality_ablation.csv`
- `noise_robustness.csv`

The experimental v1.3.0 baseline yields 114 true negatives, 32 true positives, 0 false positives, and 14 false negatives. Accuracy = 0.9125, precision = 1.0000, recall = 0.6957, F1-score = 0.8205, ROC-AUC = 0.9773, and PR-AUC = 0.9592. These values assess agreement with controlled synthetic labels and must not be interpreted as field accuracy.

Alpha sensitivity evaluates 0.20, 0.40, 0.50, 0.60, and 0.80. Modality ablation compares pollutant-weather only, image-feature only, and the baseline ensemble. Controlled feature-space perturbation is evaluated at 0%, 5%, 10%, and 20%. For 5–20% noise, the fixed clean-baseline scaler and Isolation Forest models are reused across 30 independent perturbation realizations; the models are not re-fitted for each noisy realization.

## Folder structure

```text
.
├── app.py
├── model_core.py
├── experiment_runner.py
├── requirements.txt
├── requirements-lock.txt
├── data_dictionary.csv
├── data/
│   ├── simulated_drone_sensing.csv
│   ├── image_features.csv
│   └── images/
├── outputs/
│   ├── baseline_metrics.json
│   ├── baseline_scored_mission.csv
│   ├── scenario_detection.csv
│   ├── alpha_sensitivity.csv
│   ├── modality_ablation.csv
│   └── noise_robustness.csv
├── logs/
├── docs/
├── LICENSE
├── LICENSE-DATA.md
├── CITATION.cff
├── RELEASE_NOTES.md
└── MANUSCRIPT_ALIGNMENT.md
```

## Reproduce the experiment

Create a Python environment and install the locked dependencies:

```bash
pip install --upgrade pip
pip install -r requirements-lock.txt
```

Run the controlled experiment:

```bash
python experiment_runner.py
```

Run the Streamlit prototype:

```bash
streamlit run app.py
```

## Interpretation boundaries

- All sensing variables and image examples in this release are synthetic.
- Perfect baseline metrics reflect the deliberately controlled event-generation and scoring setting.
- Digital Twin, Data Lakehouse, and distributed edge-cloud services are architectural-readiness concepts in the paper and are not production services in this release.
- The prototype is for reproducibility and research demonstration only and must not be used for real environmental, emergency, or safety decisions.

## Licensing

- Source code: MIT License (`LICENSE`).
- Synthetic data and generated sample images: CC BY 4.0 (`LICENSE-DATA.md`).

## Citation

After this release is published in Zenodo, cite the **new version DOI assigned by Zenodo**. Do not reuse an earlier version DOI as the identifier for Version 1.3.0.

## v1.3.1 single-drone YOLO smoke extension

This candidate adds an optional **single-drone raw-image smoke-inference tab**. It expects the previously trained Boreal Forest Fire UAV `best.pt` weights at `models/best.pt` (or another user-specified path). The v1.3 synthetic pollutant-weather experiments, sensitivity, ablation, and fixed-model robustness tests are unchanged.

The extension deliberately excludes the later 1–4 drone selection and multi-drone collaboration work so that the current IJ-AI resubmission can remain within a compact single-drone scope.


## v1.3.2 corrected single-drone YOLO evidence reporting

- Separates **candidate smoke boxes** from **accepted smoke detections**.
- Keeps the low-level YOLO candidate threshold separate from the accepted-smoke reporting threshold.
- Treats the default accepted threshold as provisional until a held-out batch validation is completed.
- Computes candidate and accepted **bounding-box coverage ratios from the union of boxes**, preventing overlap double-counting.
- Explicitly states that bounding-box coverage is not smoke-pixel segmentation.
- Keeps the visual score as maximum candidate smoke confidence; coverage is not blended into the score.
- Leaves the controlled v1.3 Isolation-Forest baseline, alpha sensitivity, modality ablation, and robustness experiments unchanged.
