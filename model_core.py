from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import MinMaxScaler, StandardScaler

POLLUTANT_FEATURES = [
    "pm25", "pm10", "co2", "no2", "so2", "voc",
    "temperature", "humidity", "wind_speed", "altitude",
]
IMAGE_FEATURES = [
    "red_index", "smoke_index", "brightness", "contrast", "texture", "edge_density",
]

@dataclass(frozen=True)
class ModelConfig:
    contamination: float = 0.10
    alpha: float = 0.60
    watch: float = 0.45
    warning: float = 0.60
    critical: float = 0.75
    pollutant_if_weight: float = 0.72
    image_if_weight: float = 0.70
    pollutant_seed: int = 12
    image_seed: int = 24
    n_estimators: int = 200

    def validate(self) -> None:
        if not 0 < self.contamination <= 0.5:
            raise ValueError("contamination must be in (0, 0.5].")
        if not 0 <= self.alpha <= 1:
            raise ValueError("alpha must be in [0, 1].")
        if not (0 <= self.watch < self.warning < self.critical <= 1):
            raise ValueError("Thresholds must satisfy 0 <= watch < warning < critical <= 1.")


def make_aerial_image(path: Path, idx: int, anomaly_type: str, seed: int = 42) -> Dict[str, float]:
    rng = np.random.default_rng(seed + idx)
    width, height = 480, 320
    base = Image.new("RGB", (width, height), (120, 160, 110))
    draw = ImageDraw.Draw(base)
    for _ in range(14):
        x0 = int(rng.integers(-60, width)); y0 = int(rng.integers(-40, height))
        x1 = x0 + int(rng.integers(70, 180)); y1 = y0 + int(rng.integers(40, 130))
        color = tuple(int(c) for c in rng.choice(np.array([
            [83, 139, 73], [143, 124, 75], [98, 160, 94], [126, 137, 95]
        ]), 1)[0])
        draw.rectangle([x0, y0, x1, y1], fill=color)
    for _ in range(3):
        y = int(rng.integers(30, height - 30)); dy = int(rng.integers(-30, 30))
        draw.line([(0, y), (width, y + dy)], fill=(80, 80, 80), width=8)
        draw.line([(0, y + 1), (width, y + dy + 1)], fill=(180, 180, 160), width=1)
    for _ in range(8):
        x = int(rng.integers(20, width - 60)); y = int(rng.integers(20, height - 60))
        draw.rectangle([x, y, x + 35, y + 22], fill=(105, 110, 120), outline=(55, 55, 65), width=1)
    if anomaly_type == "smoke":
        for _ in range(18):
            x = int(rng.integers(120, 330)); y = int(rng.integers(70, 230)); r = int(rng.integers(18, 45)); shade = int(rng.integers(55, 105))
            draw.ellipse([x-r, y-r, x+r, y+r], fill=(shade, shade, shade))
        base = base.filter(ImageFilter.GaussianBlur(radius=2))
    elif anomaly_type == "fire":
        cx, cy = int(rng.integers(160, 320)), int(rng.integers(90, 220))
        for r, color in [(60, (150, 30, 20)), (42, (220, 65, 20)), (25, (255, 170, 30))]:
            draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=color)
        for _ in range(10):
            x = int(rng.integers(cx-70, cx+70)); y = int(rng.integers(cy-75, cy+50))
            draw.polygon([(x, y), (x+14, y+42), (x-18, y+38)], fill=(240, 90, 20))
    elif anomaly_type == "dust":
        for _ in range(35):
            x = int(rng.integers(60, width-60)); y = int(rng.integers(40, height-40)); r = int(rng.integers(10, 25))
            draw.ellipse([x-r, y-r, x+r, y+r], fill=(170, 140, 95))
        base = base.filter(ImageFilter.GaussianBlur(radius=1.5))
    arr = np.asarray(base).astype(np.int16)
    arr = np.clip(arr + rng.normal(0, 6, arr.shape), 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)
    return extract_image_features(path)


def extract_image_features(path: Path) -> Dict[str, float]:
    img = Image.open(path).convert("RGB").resize((240, 160))
    arr = np.asarray(img).astype(np.float32) / 255.0
    gray = arr.mean(axis=2); red, green, blue = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    red_index = float(np.mean(np.maximum(red - (green + blue) / 2, 0)))
    smoke_index = float(np.mean((gray < 0.38).astype(float)) + np.std(gray) * 0.15)
    brightness = float(np.mean(gray)); contrast = float(np.std(gray))
    gy, gx = np.gradient(gray); grad = np.sqrt(gx**2 + gy**2)
    edge_density = float(np.mean(grad > 0.08))
    texture = float(np.mean(np.abs(gray[:, 1:] - gray[:, :-1])) + np.mean(np.abs(gray[1:, :] - gray[:-1, :])))
    return dict(red_index=red_index, smoke_index=smoke_index, brightness=brightness,
                contrast=contrast, texture=texture, edge_density=edge_density)


def generate_simulated_data(data_dir: Path, n: int = 160, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    image_dir = data_dir / "images"; image_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed); random.seed(seed)
    for p in image_dir.glob("frame_*.png"): p.unlink()
    start = pd.Timestamp("2026-01-01 08:00:00")
    anomaly_windows = {"pollution_plume": range(54, 72), "fire_smoke": range(96, 112), "dust_event": range(130, 142)}
    rows: List[Dict[str, object]] = []; image_rows: List[Dict[str, object]] = []
    for i in range(n):
        ts = start + pd.Timedelta(minutes=2*i)
        lat = -6.9147 + 0.006*math.sin(i/20) + rng.normal(0, 0.00025)
        lon = 107.6098 + 0.006*math.cos(i/24) + rng.normal(0, 0.00025)
        altitude = 75 + 12*math.sin(i/18) + rng.normal(0, 3)
        temp = 29 + 1.8*math.sin(i/30) + rng.normal(0, 0.45)
        humidity = 68 - 7*math.sin(i/33) + rng.normal(0, 2.5)
        wind = max(0.2, 2.2 + 0.8*math.sin(i/15) + rng.normal(0, 0.35))
        pm25 = 26 + 5*math.sin(i/17) + rng.normal(0, 2.3)
        pm10 = 52 + 9*math.sin(i/19) + rng.normal(0, 4.5)
        co2 = 430 + 22*math.sin(i/25) + rng.normal(0, 12)
        no2 = 18 + 4*math.sin(i/21) + rng.normal(0, 1.8)
        so2 = 7 + 1.5*math.sin(i/23) + rng.normal(0, 0.8)
        voc = 0.28 + 0.04*math.sin(i/16) + rng.normal(0, 0.025)
        label, visual = "normal", "normal"
        if i in anomaly_windows["pollution_plume"]:
            label, visual = "pollution_plume", "dust"
            pm25 += rng.uniform(35, 55); pm10 += rng.uniform(45, 75); no2 += rng.uniform(10, 20); voc += rng.uniform(0.15, 0.35)
        elif i in anomaly_windows["fire_smoke"]:
            label, visual = "fire_smoke", ("fire" if i % 3 == 0 else "smoke")
            pm25 += rng.uniform(45, 70); pm10 += rng.uniform(60, 90); co2 += rng.uniform(110, 180)
            temp += rng.uniform(3.0, 5.5); humidity -= rng.uniform(6, 12)
        elif i in anomaly_windows["dust_event"]:
            label, visual = "industrial_dust", "dust"
            pm10 += rng.uniform(75, 110); pm25 += rng.uniform(20, 35); so2 += rng.uniform(5, 9)
        filename = f"frame_{i:03d}.png"; rel = str(Path("data/images") / filename)
        features = make_aerial_image(image_dir / filename, i, visual, seed)
        rows.append({"timestamp": ts, "mission_minute": i*2, "latitude": lat, "longitude": lon,
                     "altitude": altitude, "pm25": max(pm25,1), "pm10": max(pm10,1), "co2": max(co2,300),
                     "no2": max(no2,0.1), "so2": max(so2,0.1), "voc": max(voc,0.01), "temperature": temp,
                     "humidity": max(min(humidity,100),1), "wind_speed": wind, "image_file": rel, "simulated_event": label})
        image_rows.append({"image_file": rel, "simulated_event": label, **features})
    sensor_df, feature_df = pd.DataFrame(rows), pd.DataFrame(image_rows)
    sensor_df.to_csv(data_dir / "simulated_drone_sensing.csv", index=False)
    feature_df.to_csv(data_dir / "image_features.csv", index=False)
    return sensor_df, feature_df


def normalize_score(values: np.ndarray) -> np.ndarray:
    return MinMaxScaler().fit_transform(np.asarray(values).reshape(-1, 1)).ravel()


def add_noise(sensor_df: pd.DataFrame, feature_df: pd.DataFrame, level: float, seed: int = 123) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if level <= 0: return sensor_df.copy(), feature_df.copy()
    rng = np.random.default_rng(seed)
    s, f = sensor_df.copy(), feature_df.copy()
    for col in POLLUTANT_FEATURES:
        scale = max(float(s[col].std()), 1e-9)
        s[col] = s[col] + rng.normal(0, level * scale, len(s))
    for col in IMAGE_FEATURES:
        scale = max(float(f[col].std()), 1e-9)
        f[col] = f[col] + rng.normal(0, level * scale, len(f))
    return s, f


def compute_anomaly_scores(sensor_df: pd.DataFrame, feature_df: pd.DataFrame, config: ModelConfig) -> pd.DataFrame:
    config.validate()
    df = sensor_df.merge(feature_df, on=["image_file", "simulated_event"], how="left")
    xp = StandardScaler().fit_transform(df[POLLUTANT_FEATURES])
    pm = IsolationForest(n_estimators=config.n_estimators, contamination=config.contamination, random_state=config.pollutant_seed)
    p_if = normalize_score(-pm.fit(xp).decision_function(xp))
    p_rule = ((df.pm25 > 55)*0.35 + (df.pm10 > 100)*0.25 + (df.co2 > 550)*0.15 +
              (df.no2 > 35)*0.10 + (df.so2 > 12)*0.10 + (df.voc > 0.55)*0.05).astype(float)
    df["pollutant_if_score"] = p_if; df["pollutant_rule_score"] = p_rule
    df["pollutant_score"] = np.clip(config.pollutant_if_weight*p_if + (1-config.pollutant_if_weight)*p_rule, 0, 1)
    xi = StandardScaler().fit_transform(df[IMAGE_FEATURES])
    im = IsolationForest(n_estimators=config.n_estimators, contamination=config.contamination, random_state=config.image_seed)
    i_if = normalize_score(-im.fit(xi).decision_function(xi))
    i_rule = np.clip(df.red_index*3.2 + df.smoke_index*1.8 + (1-df.brightness)*0.15 + df.contrast*0.5, 0, 1)
    df["image_if_score"] = i_if; df["image_rule_score"] = i_rule
    df["image_score"] = np.clip(config.image_if_weight*i_if + (1-config.image_if_weight)*i_rule, 0, 1)
    df["ensemble_score"] = np.clip(config.alpha*df.pollutant_score + (1-config.alpha)*df.image_score, 0, 1)
    df["status"] = df.ensemble_score.map(lambda x: status_from_score(float(x), config.watch, config.warning, config.critical))
    df["recommendation"] = df.status.map(recommendation_from_status)
    df["ground_truth_anomaly"] = (df.simulated_event != "normal").astype(int)
    df["predicted_anomaly"] = (df.status != "Normal").astype(int)
    return df


def status_from_score(score: float, watch: float, warning: float, critical: float) -> str:
    if score >= critical: return "Critical"
    if score >= warning: return "Warning"
    if score >= watch: return "Watch"
    return "Normal"


def recommendation_from_status(status: str) -> str:
    return {"Normal":"Continue monitoring.", "Watch":"Increase sampling density and inspect nearby frames.",
            "Warning":"Request human validation and prepare response option.",
            "Critical":"Escalate to operator, validate field evidence, and prioritize response."}[status]


def evaluate_scores(df: pd.DataFrame, score_col: str = "ensemble_score", threshold: float | None = None) -> Dict[str, float]:
    y = df.ground_truth_anomaly.astype(int).to_numpy()
    pred = df.predicted_anomaly.astype(int).to_numpy() if threshold is None else (df[score_col].to_numpy() >= threshold).astype(int)
    out = {
        "accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0),
    }
    try: out["roc_auc"] = roc_auc_score(y, df[score_col])
    except ValueError: out["roc_auc"] = float("nan")
    try: out["pr_auc"] = average_precision_score(y, df[score_col])
    except ValueError: out["pr_auc"] = float("nan")
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    out.update(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))
    return out


def scenario_detection(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby("simulated_event", as_index=False)
            .agg(frames=("mission_minute","count"), mean_score=("ensemble_score","mean"),
                 max_score=("ensemble_score","max"), detected=("predicted_anomaly","sum"))
            .assign(detection_rate=lambda x: x.detected/x.frames))


def run_experiment_grid(sensor_df: pd.DataFrame, feature_df: pd.DataFrame, base: ModelConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    alpha_rows=[]
    for a in [0.2,0.4,0.5,0.6,0.8]:
        cfg = ModelConfig(**{**base.__dict__, "alpha": a})
        d = compute_anomaly_scores(sensor_df, feature_df, cfg); m=evaluate_scores(d)
        alpha_rows.append({"alpha":a, **m, "non_normal_frames":int(d.predicted_anomaly.sum()), "mean_score":d.ensemble_score.mean()})
    ablation=[]
    for name,a in [("Pollutant only",1.0),("Image only",0.0),("Ensemble",base.alpha)]:
        cfg=ModelConfig(**{**base.__dict__, "alpha":a}); d=compute_anomaly_scores(sensor_df,feature_df,cfg); m=evaluate_scores(d)
        ablation.append({"configuration":name,"alpha":a,**m})
    noise=[]
    for level in [0.0,0.05,0.10,0.20]:
        s,f=add_noise(sensor_df,feature_df,level,seed=123); d=compute_anomaly_scores(s,f,base); m=evaluate_scores(d)
        noise.append({"noise_level":level,**m,"mean_score":d.ensemble_score.mean(),"status_agreement":float((d.status==compute_anomaly_scores(sensor_df,feature_df,base).status).mean())})
    return pd.DataFrame(alpha_rows), pd.DataFrame(ablation), pd.DataFrame(noise)
