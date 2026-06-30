# Drone-AI Ensemble Anomaly Detection Prototype

This repository contains a lightweight Streamlit prototype for illustrating the paper concept: **Drone-AI as a situational intelligence layer for Smart X**. The app uses simulated drone sensing data and simulated drone-view images to demonstrate a sensing-to-decision workflow.

The prototype is intended for reproducibility, GitHub sharing, and Zenodo archiving. It is **not field data** and should be described as a conceptual proof-of-concept or illustrative prototype.

## What the prototype does

The app demonstrates a simple ensemble anomaly detection workflow using two sensing branches:

1. **Pollutant and weather anomaly branch**
   - Uses simulated telemetry, pollutants, and microclimate data.
   - Detects anomalies using an Isolation Forest model and rule-based threshold signals.
   - Variables include PM2.5, PM10, CO2, NO2, SO2, VOC, temperature, humidity, wind speed, altitude, latitude, and longitude.

2. **Drone image anomaly branch**
   - Uses simulated drone images generated as simple aerial scenes.
   - Extracts lightweight visual features such as red intensity, smoke/darkness index, brightness, contrast, texture, and edge density.
   - Detects visual anomalies using an Isolation Forest model and simple image rules.

3. **Ensemble decision layer**
   - Combines pollutant anomaly score and image anomaly score.
   - Produces normal, watch, warning, or critical status.
   - Includes a human-in-the-loop field for validation decisions.

## Folder structure

```text
drone_ai_anomaly_streamlit/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   ├── simulated_drone_sensing.csv
│   ├── image_features.csv
│   └── images/
│       ├── frame_000.png
│       ├── frame_001.png
│       └── ...
└── docs/
    └── paper_insert_algorithm.txt
```

## Setup from a new Python environment

### 1. Create a project folder

```bash
mkdir drone_ai_anomaly_streamlit
cd drone_ai_anomaly_streamlit
```

If you downloaded the ZIP file, extract it first and enter the extracted folder.

### 2. Create a virtual environment

#### Windows PowerShell

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

#### Windows Command Prompt

```bash
python -m venv .venv
.venv\Scripts\activate.bat
```

#### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Regenerating the simulated dataset

The package already includes simulated CSV data and images. The app also provides a sidebar button to regenerate the synthetic mission data. Regeneration is deterministic when the same random seed is used.

## Notes for GitHub and Zenodo

Recommended files to upload:

- `app.py`
- `requirements.txt`
- `README.md`
- `data/simulated_drone_sensing.csv`
- `data/image_features.csv`
- `data/images/*.png`
- `docs/paper_insert_algorithm.txt`

Recommended Zenodo description:

> This dataset and code package provides a simulated Drone-AI anomaly detection prototype for Smart X situational intelligence. It contains synthetic pollutant, weather, telemetry, and drone-view image data, together with a Streamlit implementation of a two-branch ensemble anomaly detection workflow. The data are simulated and are not field measurements.

## Suggested citation text

If archived in Zenodo, replace the DOI placeholder below:

> D. R. Hermanus, S. H. Supangkat, and F. Hidayat, Drone-AI Ensemble Anomaly Detection Prototype for Smart X Situational Intelligence, Zenodo, 2026. DOI: [replace with Zenodo DOI].

## Disclaimer

All sensing data and drone images in this repository are synthetic. They are created only to demonstrate the proposed architecture and should not be used for real environmental or safety decisions.

## Citation and archive

This repository is prepared for archival through Zenodo. After creating a GitHub release, Zenodo will archive the release and mint a DOI. Please cite the archived release DOI when using the software or simulated dataset.

## Reproducibility note

The data provided in this repository are simulated pollutant-weather, telemetry, and drone-image feature data. They are intended to demonstrate the sensing-to-decision workflow and ensemble anomaly detection logic. They are not field measurements.

## Suggested repository citation

Hermanus, D. R., Supangkat, S. H., & Hidayat, F. (2026). *Drone AI Situational Intelligence Layer for Smart X: Streamlit Prototype and Simulated Dataset* (Version 1.0.0). Zenodo. DOI: to be added after release.
