from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


@dataclass
class YoloSmokeResult:
    annotated_rgb: np.ndarray
    detections: pd.DataFrame
    candidate_count: int
    accepted_count: int
    max_confidence: float
    mean_confidence: float
    candidate_bbox_coverage_ratio: float
    accepted_bbox_coverage_ratio: float
    visual_score: float


def _load_yolo(weights_path: Path):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is not installed. Run: pip install -r requirements.txt"
        ) from exc
    return YOLO(str(weights_path))


def _class_name(names: Any, cls_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(cls_id, cls_id))
    if isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
        return str(names[cls_id])
    return str(cls_id)


def _clip_box(
    box: Sequence[float], width: int, height: int
) -> Tuple[float, float, float, float] | None:
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = min(max(x1, 0.0), float(width))
    x2 = min(max(x2, 0.0), float(width))
    y1 = min(max(y1, 0.0), float(height))
    y2 = min(max(y2, 0.0), float(height))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _rectangle_union_area(
    boxes: Sequence[Sequence[float]], width: int, height: int
) -> float:
    """Exact union area for axis-aligned boxes using an x-sweep."""
    clipped = []
    for box in boxes:
        c = _clip_box(box, width, height)
        if c is not None:
            clipped.append(c)
    if not clipped:
        return 0.0

    xs = sorted({b[0] for b in clipped} | {b[2] for b in clipped})
    area = 0.0
    for xa, xb in zip(xs[:-1], xs[1:]):
        if xb <= xa:
            continue
        intervals = []
        for x1, y1, x2, y2 in clipped:
            if x1 < xb and x2 > xa:
                intervals.append((y1, y2))
        if not intervals:
            continue
        intervals.sort()
        cur_start, cur_end = intervals[0]
        merged_y = 0.0
        for start, end in intervals[1:]:
            if start <= cur_end:
                cur_end = max(cur_end, end)
            else:
                merged_y += cur_end - cur_start
                cur_start, cur_end = start, end
        merged_y += cur_end - cur_start
        area += (xb - xa) * merged_y
    return area


def _coverage_ratio(
    detections: pd.DataFrame,
    width: int,
    height: int,
    accepted_only: bool | None = None,
) -> float:
    if detections.empty:
        return 0.0
    selected = detections
    if accepted_only is True:
        selected = detections[detections["accepted"]]
    elif accepted_only is False:
        selected = detections[~detections["accepted"]]
    if selected.empty:
        return 0.0
    boxes = selected[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float).tolist()
    union = _rectangle_union_area(boxes, width, height)
    return float(min(max(union / max(float(width * height), 1.0), 0.0), 1.0))


def run_smoke_inference(
    image: Image.Image,
    weights_path: Path,
    candidate_confidence_threshold: float = 0.25,
    accepted_confidence_threshold: float = 0.45,
    iou_threshold: float = 0.45,
) -> YoloSmokeResult:
    """Run a frozen YOLO smoke detector on one single-drone image.

    Candidate and accepted thresholds are intentionally separated.
    The visual score is maximum candidate smoke confidence.
    Bounding-box coverage is reported separately as union coverage.
    """
    if not weights_path.exists():
        raise FileNotFoundError(f"YOLO weights not found: {weights_path}")
    if accepted_confidence_threshold < candidate_confidence_threshold:
        raise ValueError(
            "Accepted smoke threshold must be greater than or equal to the candidate threshold."
        )

    model = _load_yolo(weights_path)
    rgb = np.asarray(image.convert("RGB"))
    results = model.predict(
        source=rgb,
        conf=float(candidate_confidence_threshold),
        iou=float(iou_threshold),
        verbose=False,
    )
    if not results:
        return YoloSmokeResult(
            rgb, pd.DataFrame(), 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0
        )

    result = results[0]
    names = getattr(result, "names", getattr(model, "names", {}))
    boxes = getattr(result, "boxes", None)
    rows: List[Dict[str, float | int | str | bool]] = []
    h, w = rgb.shape[:2]

    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.detach().cpu().numpy()
        confs = boxes.conf.detach().cpu().numpy()
        classes = boxes.cls.detach().cpu().numpy().astype(int)
        class_names = [_class_name(names, int(c)) for c in classes]
        smoke_mask = np.array(["smoke" in name.lower() for name in class_names], dtype=bool)

        unique_names = {name.lower() for name in class_names}
        if not smoke_mask.any() and len(set(classes.tolist())) == 1 and len(unique_names) == 1:
            smoke_mask[:] = True

        for box, conf, cls_id, name, is_smoke in zip(
            xyxy, confs, classes, class_names, smoke_mask
        ):
            if not is_smoke:
                continue
            clipped = _clip_box(box, w, h)
            if clipped is None:
                continue
            x1, y1, x2, y2 = clipped
            rows.append(
                {
                    "class_id": int(cls_id),
                    "class_name": name,
                    "confidence": float(conf),
                    "accepted": bool(float(conf) >= float(accepted_confidence_threshold)),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "box_area_ratio": float(
                        max(0.0, (x2 - x1) * (y2 - y1))
                        / max(float(w * h), 1.0)
                    ),
                }
            )

    detections = pd.DataFrame(rows)
    if detections.empty:
        candidate_count = accepted_count = 0
        max_conf = mean_conf = 0.0
        candidate_coverage = accepted_coverage = 0.0
    else:
        candidate_count = int(len(detections))
        accepted_count = int(detections["accepted"].sum())
        max_conf = float(detections["confidence"].max())
        mean_conf = float(detections["confidence"].mean())
        candidate_coverage = _coverage_ratio(detections, w, h, accepted_only=None)
        accepted_coverage = _coverage_ratio(detections, w, h, accepted_only=True)

    visual_score = max_conf

    try:
        annotated_rgb = _draw_detections(rgb, detections)
    except Exception:
        annotated_rgb = rgb

    return YoloSmokeResult(
        annotated_rgb=annotated_rgb,
        detections=detections,
        candidate_count=candidate_count,
        accepted_count=accepted_count,
        max_confidence=max_conf,
        mean_confidence=mean_conf,
        candidate_bbox_coverage_ratio=candidate_coverage,
        accepted_bbox_coverage_ratio=accepted_coverage,
        visual_score=visual_score,
    )




def _draw_detections(rgb: np.ndarray, detections: pd.DataFrame) -> np.ndarray:
    """Draw accepted and candidate boxes with different colors for interpretability."""
    if detections.empty:
        return rgb

    img = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    accepted_color = (34, 197, 94)   # green
    candidate_color = (245, 158, 11) # orange
    text_color = (255, 255, 255)

    # draw candidate first, then accepted on top
    ordered = detections.sort_values(["accepted", "confidence"], ascending=[True, False])
    for _, row in ordered.iterrows():
        x1, y1, x2, y2 = [float(row[c]) for c in ["x1", "y1", "x2", "y2"]]
        accepted = bool(row["accepted"])
        conf = float(row["confidence"])
        color = accepted_color if accepted else candidate_color
        prefix = "accepted smoke" if accepted else "candidate smoke"
        label = f"{prefix} {conf:.2f}"

        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        if font is not None:
            try:
                bbox = draw.textbbox((x1, y1), label, font=font)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
            except Exception:
                tw = int(len(label) * 7)
                th = 12
        else:
            tw = int(len(label) * 7)
            th = 12

        tx = max(0, int(x1))
        ty = max(0, int(y1) - th - 6)
        if ty <= 0:
            ty = max(0, int(y1) + 2)
        draw.rectangle([tx, ty, tx + tw + 8, ty + th + 4], fill=color)
        draw.text((tx + 4, ty + 2), label, fill=text_color, font=font)

    return np.asarray(img)

def visual_status(score: float, watch: float, warning: float, critical: float) -> str:
    if score >= critical:
        return "Critical"
    if score >= warning:
        return "Warning"
    if score >= watch:
        return "Watch"
    return "Normal"


def append_inference_log(csv_path: Path, row: Dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    record = dict(row)
    record.setdefault("inference_timestamp_utc", datetime.now(timezone.utc).isoformat())
    new = pd.DataFrame([record])
    old = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()
    pd.concat([old, new], ignore_index=True).to_csv(csv_path, index=False)
