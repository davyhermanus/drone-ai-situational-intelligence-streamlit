"""Streamlit prototype: Drone-AI ensemble anomaly detection.

This app uses synthetic drone sensing data and generated drone-view images to
illustrate a sensing-to-decision workflow for Smart X situational intelligence.
It is not intended for operational environmental monitoring.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageDraw, ImageFilter
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
IMAGE_DIR = DATA_DIR / "images"
SENSOR_CSV = DATA_DIR / "simulated_drone_sensing.csv"
FEATURE_CSV = DATA_DIR / "image_features.csv"

POLLUTANT_FEATURES = [
    "pm25",
    "pm10",
    "co2",
    "no2",
    "so2",
    "voc",
    "temperature",
    "humidity",
    "wind_speed",
    "altitude",
]
IMAGE_FEATURES = [
    "red_index",
    "smoke_index",
    "brightness",
    "contrast",
    "texture",
    "edge_density",
]


def ensure_dirs() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    IMAGE_DIR.mkdir(exist_ok=True)


def make_aerial_image(path: Path, idx: int, anomaly_type: str, seed: int = 42) -> Dict[str, float]:
    """Create a simple simulated drone-view image and return extracted features."""
    rng = np.random.default_rng(seed + idx)
    width, height = 480, 320

    # Base aerial background: green/brown land patches and road lines.
    base = Image.new("RGB", (width, height), (120, 160, 110))
    draw = ImageDraw.Draw(base)

    # Land patches.
    for _ in range(14):
        x0 = int(rng.integers(-60, width))
        y0 = int(rng.integers(-40, height))
        x1 = x0 + int(rng.integers(70, 180))
        y1 = y0 + int(rng.integers(40, 130))
        color = tuple(int(c) for c in rng.choice(
            np.array([[83, 139, 73], [143, 124, 75], [98, 160, 94], [126, 137, 95]]),
            1,
        )[0])
        draw.rectangle([x0, y0, x1, y1], fill=color)

    # Roads / industrial lines.
    for _ in range(3):
        y = int(rng.integers(30, height - 30))
        draw.line([(0, y), (width, y + int(rng.integers(-30, 30)))], fill=(80, 80, 80), width=8)
        draw.line([(0, y + 1), (width, y + int(rng.integers(-30, 30)) + 1)], fill=(180, 180, 160), width=1)

    # Small buildings.
    for _ in range(8):
        x = int(rng.integers(20, width - 60))
        y = int(rng.integers(20, height - 60))
        draw.rectangle([x, y, x + 35, y + 22], fill=(105, 110, 120), outline=(55, 55, 65), width=1)

    # Add anomaly visual patterns.
    if anomaly_type == "smoke":
        for _ in range(18):
            x = int(rng.integers(120, 330))
            y = int(rng.integers(70, 230))
            r = int(rng.integers(18, 45))
            shade = int(rng.integers(55, 105))
            draw.ellipse([x - r, y - r, x + r, y + r], fill=(shade, shade, shade))
        base = base.filter(ImageFilter.GaussianBlur(radius=2))
    elif anomaly_type == "fire":
        cx, cy = int(rng.integers(160, 320)), int(rng.integers(90, 220))
        for r, color in [(60, (150, 30, 20)), (42, (220, 65, 20)), (25, (255, 170, 30))]:
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
        for _ in range(10):
            x = int(rng.integers(cx - 70, cx + 70))
            y = int(rng.integers(cy - 75, cy + 50))
            draw.polygon([(x, y), (x + 14, y + 42), (x - 18, y + 38)], fill=(240, 90, 20))
    elif anomaly_type == "flood":
        for _ in range(5):
            x0 = int(rng.integers(-20, 120))
            y0 = int(rng.integers(120, height - 30))
            x1 = int(rng.integers(250, width + 40))
            y1 = y0 + int(rng.integers(25, 60))
            draw.rectangle([x0, y0, x1, y1], fill=(60, 115, 170))
        base = base.filter(ImageFilter.GaussianBlur(radius=1))
    elif anomaly_type == "dust":
        for _ in range(35):
            x = int(rng.integers(60, width - 60))
            y = int(rng.integers(40, height - 40))
            r = int(rng.integers(10, 25))
            draw.ellipse([x - r, y - r, x + r, y + r], fill=(170, 140, 95))
        base = base.filter(ImageFilter.GaussianBlur(radius=1.5))

    # Sensor noise overlay.
    arr = np.asarray(base).astype(np.int16)
    noise = rng.normal(0, 6, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    img.save(path)
    return extract_image_features(path)


def extract_image_features(path: Path) -> Dict[str, float]:
    img = Image.open(path).convert("RGB").resize((240, 160))
    arr = np.asarray(img).astype(np.float32) / 255.0
    gray = arr.mean(axis=2)
    red = arr[:, :, 0]
    green = arr[:, :, 1]
    blue = arr[:, :, 2]

    red_index = float(np.mean(np.maximum(red - (green + blue) / 2, 0)))
    smoke_index = float(np.mean((gray < 0.38).astype(float)) + np.std(gray) * 0.15)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    gy, gx = np.gradient(gray)
    grad = np.sqrt(gx**2 + gy**2)
    edge_density = float(np.mean(grad > 0.08))
    texture = float(np.mean(np.abs(gray[:, 1:] - gray[:, :-1])) + np.mean(np.abs(gray[1:, :] - gray[:-1, :])))

    return {
        "red_index": red_index,
        "smoke_index": smoke_index,
        "brightness": brightness,
        "contrast": contrast,
        "texture": texture,
        "edge_density": edge_density,
    }


def generate_simulated_data(n: int = 160, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ensure_dirs()
    rng = np.random.default_rng(seed)
    random.seed(seed)

    # Clean previous images.
    for img_path in IMAGE_DIR.glob("frame_*.png"):
        img_path.unlink()

    start = pd.Timestamp("2026-01-01 08:00:00")
    timestamps = [start + pd.Timedelta(minutes=2 * i) for i in range(n)]
    base_lat, base_lon = -6.9147, 107.6098  # Bandung-like synthetic location.

    rows: List[Dict[str, object]] = []
    image_rows: List[Dict[str, object]] = []

    anomaly_windows = {
        "pollution_plume": range(54, 72),
        "fire_smoke": range(96, 112),
        "dust_event": range(130, 142),
    }

    for i, ts in enumerate(timestamps):
        lat = base_lat + 0.006 * math.sin(i / 20) + rng.normal(0, 0.00025)
        lon = base_lon + 0.006 * math.cos(i / 24) + rng.normal(0, 0.00025)
        altitude = 75 + 12 * math.sin(i / 18) + rng.normal(0, 3)
        temp = 29 + 1.8 * math.sin(i / 30) + rng.normal(0, 0.45)
        humidity = 68 - 7 * math.sin(i / 33) + rng.normal(0, 2.5)
        wind = max(0.2, 2.2 + 0.8 * math.sin(i / 15) + rng.normal(0, 0.35))

        pm25 = 26 + 5 * math.sin(i / 17) + rng.normal(0, 2.3)
        pm10 = 52 + 9 * math.sin(i / 19) + rng.normal(0, 4.5)
        co2 = 430 + 22 * math.sin(i / 25) + rng.normal(0, 12)
        no2 = 18 + 4 * math.sin(i / 21) + rng.normal(0, 1.8)
        so2 = 7 + 1.5 * math.sin(i / 23) + rng.normal(0, 0.8)
        voc = 0.28 + 0.04 * math.sin(i / 16) + rng.normal(0, 0.025)

        label = "normal"
        visual = "normal"
        if i in anomaly_windows["pollution_plume"]:
            label = "pollution_plume"
            pm25 += rng.uniform(35, 55)
            pm10 += rng.uniform(45, 75)
            no2 += rng.uniform(10, 20)
            voc += rng.uniform(0.15, 0.35)
            visual = "dust"
        elif i in anomaly_windows["fire_smoke"]:
            label = "fire_smoke"
            pm25 += rng.uniform(45, 70)
            pm10 += rng.uniform(60, 90)
            co2 += rng.uniform(110, 180)
            temp += rng.uniform(3.0, 5.5)
            humidity -= rng.uniform(6, 12)
            visual = "fire" if i % 3 == 0 else "smoke"
        elif i in anomaly_windows["dust_event"]:
            label = "industrial_dust"
            pm10 += rng.uniform(75, 110)
            pm25 += rng.uniform(20, 35)
            so2 += rng.uniform(5, 9)
            visual = "dust"

        filename = f"frame_{i:03d}.png"
        image_path = IMAGE_DIR / filename
        features = make_aerial_image(image_path, i, visual, seed=seed)

        rows.append(
            {
                "timestamp": ts,
                "mission_minute": i * 2,
                "latitude": lat,
                "longitude": lon,
                "altitude": altitude,
                "pm25": max(pm25, 1),
                "pm10": max(pm10, 1),
                "co2": max(co2, 300),
                "no2": max(no2, 0.1),
                "so2": max(so2, 0.1),
                "voc": max(voc, 0.01),
                "temperature": temp,
                "humidity": max(min(humidity, 100), 1),
                "wind_speed": wind,
                "image_file": str(Path("data/images") / filename),
                "simulated_event": label,
            }
        )
        image_rows.append({"image_file": str(Path("data/images") / filename), "simulated_event": label, **features})

    sensor_df = pd.DataFrame(rows)
    feature_df = pd.DataFrame(image_rows)
    sensor_df.to_csv(SENSOR_CSV, index=False)
    feature_df.to_csv(FEATURE_CSV, index=False)
    return sensor_df, feature_df


def load_or_generate_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not SENSOR_CSV.exists() or not FEATURE_CSV.exists() or len(list(IMAGE_DIR.glob("frame_*.png"))) < 10:
        return generate_simulated_data()
    return pd.read_csv(SENSOR_CSV, parse_dates=["timestamp"]), pd.read_csv(FEATURE_CSV)


def normalize_score(values: np.ndarray) -> np.ndarray:
    values = values.reshape(-1, 1)
    return MinMaxScaler().fit_transform(values).ravel()


def compute_anomaly_scores(sensor_df: pd.DataFrame, feature_df: pd.DataFrame, contamination: float) -> pd.DataFrame:
    df = sensor_df.merge(feature_df, on=["image_file", "simulated_event"], how="left")

    # Pollutant branch: unsupervised model + explicit environmental threshold signal.
    x_pollutant = StandardScaler().fit_transform(df[POLLUTANT_FEATURES])
    pollutant_model = IsolationForest(n_estimators=200, contamination=contamination, random_state=12)
    pollutant_model.fit(x_pollutant)
    pollutant_if_score = normalize_score(-pollutant_model.decision_function(x_pollutant))

    pollutant_rule = (
        (df["pm25"] > 55).astype(float) * 0.35
        + (df["pm10"] > 100).astype(float) * 0.25
        + (df["co2"] > 550).astype(float) * 0.15
        + (df["no2"] > 35).astype(float) * 0.10
        + (df["so2"] > 12).astype(float) * 0.10
        + (df["voc"] > 0.55).astype(float) * 0.05
    )
    df["pollutant_score"] = np.clip(0.72 * pollutant_if_score + 0.28 * pollutant_rule, 0, 1)

    # Image branch: unsupervised model + visual rule signal.
    x_img = StandardScaler().fit_transform(df[IMAGE_FEATURES])
    image_model = IsolationForest(n_estimators=200, contamination=contamination, random_state=24)
    image_model.fit(x_img)
    image_if_score = normalize_score(-image_model.decision_function(x_img))
    image_rule = np.clip(
        df["red_index"] * 3.2 + df["smoke_index"] * 1.8 + (1 - df["brightness"]) * 0.15 + df["contrast"] * 0.5,
        0,
        1,
    )
    df["image_score"] = np.clip(0.70 * image_if_score + 0.30 * image_rule, 0, 1)

    df["ensemble_score"] = np.clip(0.60 * df["pollutant_score"] + 0.40 * df["image_score"], 0, 1)
    return df


def status_from_score(score: float, watch: float, warning: float, critical: float) -> str:
    if score >= critical:
        return "Critical"
    if score >= warning:
        return "Warning"
    if score >= watch:
        return "Watch"
    return "Normal"


def recommendation_from_status(status: str) -> str:
    return {
        "Normal": "Continue monitoring.",
        "Watch": "Increase sampling density and inspect nearby frames.",
        "Warning": "Request human validation and prepare response option.",
        "Critical": "Escalate to operator, validate field evidence, and prioritize response.",
    }[status]


def render_metric_cards(df: pd.DataFrame) -> None:
    latest = df.iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest PM2.5", f"{latest['pm25']:.1f}")
    c2.metric("Latest PM10", f"{latest['pm10']:.1f}")
    c3.metric("Latest score", f"{latest['ensemble_score']:.2f}")
    c4.metric("Detected frames", int((df["status"] != "Normal").sum()))


def plot_time_series(df: pd.DataFrame) -> None:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["mission_minute"], y=df["pm25"], mode="lines+markers", name="PM2.5"))
    fig.add_trace(go.Scatter(x=df["mission_minute"], y=df["pm10"], mode="lines+markers", name="PM10"))
    fig.add_hline(y=55, line_dash="dash", annotation_text="PM2.5 threshold")
    fig.update_layout(title="Pollutant time series", xaxis_title="Mission minute", yaxis_title="Concentration (simulated)")
    st.plotly_chart(fig, use_container_width=True)


def plot_scores(df: pd.DataFrame, watch: float, warning: float, critical: float) -> None:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["mission_minute"], y=df["pollutant_score"], mode="lines", name="Pollutant score"))
    fig.add_trace(go.Scatter(x=df["mission_minute"], y=df["image_score"], mode="lines", name="Image score"))
    fig.add_trace(go.Scatter(x=df["mission_minute"], y=df["ensemble_score"], mode="lines+markers", name="Ensemble score"))
    fig.add_hline(y=watch, line_dash="dot", annotation_text="Watch")
    fig.add_hline(y=warning, line_dash="dash", annotation_text="Warning")
    fig.add_hline(y=critical, line_dash="solid", annotation_text="Critical")
    fig.update_layout(title="Ensemble anomaly score", xaxis_title="Mission minute", yaxis_title="Score")
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Drone-AI Ensemble Anomaly Detection", layout="wide")
    st.title("Drone-AI Ensemble Anomaly Detection Prototype")
    st.caption("Synthetic pollutant, weather, telemetry, and drone-image data for Smart X situational intelligence.")

    with st.sidebar:
        st.header("Simulation and model settings")
        seed = st.number_input("Simulation seed", min_value=1, max_value=9999, value=42, step=1)
        n_samples = st.slider("Mission frames", min_value=80, max_value=240, value=160, step=20)
        contamination = st.slider("Isolation Forest contamination", 0.03, 0.25, 0.10, 0.01)
        watch = st.slider("Watch threshold", 0.20, 0.80, 0.45, 0.01)
        warning = st.slider("Warning threshold", 0.30, 0.90, 0.60, 0.01)
        critical = st.slider("Critical threshold", 0.40, 0.98, 0.75, 0.01)
        regenerate = st.button("Regenerate simulated data")

    if regenerate:
        sensor_df, feature_df = generate_simulated_data(n=n_samples, seed=int(seed))
        st.success("Synthetic mission data regenerated.")
    else:
        sensor_df, feature_df = load_or_generate_data()

    df = compute_anomaly_scores(sensor_df, feature_df, contamination=contamination)
    df["status"] = df["ensemble_score"].apply(lambda s: status_from_score(s, watch, warning, critical))
    df["recommendation"] = df["status"].apply(recommendation_from_status)

    render_metric_cards(df)

    tab1, tab2, tab3, tab4 = st.tabs(["Mission dashboard", "Drone images", "Human-in-the-loop", "Data and reproducibility"])

    with tab1:
        st.subheader("Sensing-to-decision dashboard")
        col_map, col_table = st.columns([1.2, 1])
        with col_map:
            fig_map = px.scatter_mapbox(
                df,
                lat="latitude",
                lon="longitude",
                color="status",
                size="ensemble_score",
                hover_data=["mission_minute", "pm25", "pm10", "co2", "simulated_event"],
                zoom=12,
                height=420,
                mapbox_style="open-street-map",
                title="Drone mission trajectory and anomaly status",
            )
            st.plotly_chart(fig_map, use_container_width=True)
        with col_table:
            st.dataframe(
                df[["mission_minute", "status", "ensemble_score", "pollutant_score", "image_score", "simulated_event", "recommendation"]]
                .sort_values("ensemble_score", ascending=False)
                .head(12),
                use_container_width=True,
            )
        plot_time_series(df)
        plot_scores(df, watch, warning, critical)

    with tab2:
        st.subheader("Drone-view image anomaly branch")
        top_anomalies = df.sort_values("ensemble_score", ascending=False).head(12).copy()
        cols = st.columns(4)
        for j, (_, row) in enumerate(top_anomalies.iterrows()):
            img_path = BASE_DIR / str(row["image_file"])
            with cols[j % 4]:
                st.image(str(img_path), caption=f"t={int(row['mission_minute'])} min | {row['status']} | score={row['ensemble_score']:.2f}")
                st.caption(f"Event: {row['simulated_event']}")

        st.markdown("Image feature table for the most anomalous frames")
        st.dataframe(
            top_anomalies[["mission_minute", "red_index", "smoke_index", "brightness", "contrast", "texture", "edge_density", "image_score"]],
            use_container_width=True,
        )

    with tab3:
        st.subheader("Human-in-the-loop validation")
        selected_minute = st.selectbox(
            "Select mission minute for review",
            options=df.sort_values("ensemble_score", ascending=False)["mission_minute"].astype(int).tolist(),
        )
        row = df[df["mission_minute"] == selected_minute].iloc[0]
        c1, c2 = st.columns([1, 1])
        with c1:
            st.image(str(BASE_DIR / str(row["image_file"])), caption="Selected drone frame")
        with c2:
            st.write("**Model output**")
            st.write(f"Status: **{row['status']}**")
            st.write(f"Pollutant score: {row['pollutant_score']:.3f}")
            st.write(f"Image score: {row['image_score']:.3f}")
            st.write(f"Ensemble score: {row['ensemble_score']:.3f}")
            st.write(f"Recommended action: {row['recommendation']}")
            validation = st.radio("Operator decision", ["Approve", "Correct", "Override", "Need field check"], horizontal=True)
            notes = st.text_area("Operator notes", placeholder="Example: verify source location, repeat sampling, notify field team...")
            if st.button("Save validation in session"):
                st.session_state.setdefault("validations", []).append(
                    {
                        "mission_minute": int(selected_minute),
                        "model_status": row["status"],
                        "operator_decision": validation,
                        "notes": notes,
                    }
                )
                st.success("Validation saved in current Streamlit session.")
        if "validations" in st.session_state:
            st.dataframe(pd.DataFrame(st.session_state["validations"]), use_container_width=True)

    with tab4:
        st.subheader("Data and reproducibility")
        st.write("The app uses synthetic data stored in the `data/` folder. The table below contains model outputs that can be exported for documentation.")
        st.dataframe(df.head(30), use_container_width=True)
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download scored mission CSV", data=csv_bytes, file_name="scored_drone_anomaly_mission.csv", mime="text/csv")
        st.info("All data are simulated and should not be interpreted as field measurements.")


if __name__ == "__main__":
    main()
