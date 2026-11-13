"""
export_results.py
-----------------
Run both models on the openly licensed example photos and write everything the
teaching app (and a reader of this repo) needs into ``results/``:

    results/detections.json      every box: model, image, class, confidence, pixels
    results/predictions/<model>/  the photos with boxes drawn (confidence >= 0.25)
    results/training/             fine-tuning metrics (results.csv) and the
                                  photo-free plots (curves, confusion matrix)

Why a separate export: the training images were collected for a classroom
exercise and are not published, so the fine-tuning run itself cannot be
reproduced from this repo alone. Its *outputs* — metrics, curves, and the
fine-tuned model applied to openly licensed photos — can be published, and the
teaching app reads them from here without installing torch or YOLO.

Run:  uv run python export_results.py
(needs yolo11n.pt and custom/fine-tune/weights/best.pt locally; *.pt is git-ignored)
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import torch
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
IMAGES = RESULTS / "images"
FINE_TUNE = ROOT / "custom" / "fine-tune"

# Keep every detection down to this confidence, so a viewer can explore the
# threshold afterwards. The drawn images use YOLO's default of 0.25.
EXPORT_CONFIDENCE = 0.05
DRAW_CONFIDENCE = 0.25

MODELS = {
    # name: (weights, which image folders it is run on)
    "pretrained": (ROOT / "yolo11n.pt", ["everyday", "spaghetti"]),
    "fine_tuned": (FINE_TUNE / "weights" / "best.pt", ["spaghetti"]),
}

# Training plots that contain no training photos — safe to publish.
PHOTO_FREE_PLOTS = [
    "results.png", "PR_curve.png", "F1_curve.png", "P_curve.png", "R_curve.png",
    "confusion_matrix.png", "confusion_matrix_normalized.png",
]


def detect(model: YOLO, image: Path, confidence: float) -> dict:
    """One image → its size and every box as plain data (pixels, top-left origin)."""
    result = model(str(image), conf=confidence, verbose=False)[0]
    boxes = []
    for xyxy, cls, conf in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist()):
        boxes.append({
            "class_id": int(cls),
            "label": result.names[int(cls)],
            "confidence": round(float(conf), 4),
            "box": [round(v, 1) for v in xyxy],
        })
    height, width = result.orig_shape
    return {"width": width, "height": height, "detections": boxes}


def draw(model: YOLO, image: Path, out_dir: Path) -> None:
    """Save the photo with boxes drawn, as YOLO renders them."""
    out_dir.mkdir(parents=True, exist_ok=True)
    result = model(str(image), conf=DRAW_CONFIDENCE, verbose=False)[0]
    result.save(filename=str(out_dir / image.name))


def export_training() -> dict:
    """Copy metrics and photo-free plots; return a small summary for the JSON."""
    out = RESULTS / "training"
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(FINE_TUNE / "results.csv", out / "results.csv")
    for plot in PHOTO_FREE_PLOTS:
        if (FINE_TUNE / plot).exists():
            shutil.copy(FINE_TUNE / plot, out / plot)
    # Only the settings worth teaching — args.yaml also holds the local folder layout.
    args = yaml.safe_load((FINE_TUNE / "args.yaml").read_text())
    metrics = pd.read_csv(out / "results.csv")
    metrics.columns = metrics.columns.str.strip()
    last = metrics.iloc[-1]
    # YOLO keeps the BEST round as best.pt, not the last one, and best.pt is the
    # model used everywhere else — so report its scores (stored in the checkpoint)
    # and find which round it was.
    best = torch.load(FINE_TUNE / "weights" / "best.pt", map_location="cpu", weights_only=False)["train_metrics"]
    scores = ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]
    match = (metrics[scores] - pd.Series({k: best[k] for k in scores})).abs().sum(axis=1)
    return {
        "epochs": int(last["epoch"]),
        "best_epoch": int(metrics.loc[match.idxmin(), "epoch"]),
        "train_images": len(list((ROOT / "custom" / "images" / "train").glob("*"))),
        "val_images": len(list((ROOT / "custom" / "images" / "val").glob("*"))),
        "best": {
            "precision": float(best["metrics/precision(B)"]),
            "recall": float(best["metrics/recall(B)"]),
            "mAP50": float(best["metrics/mAP50(B)"]),
            "mAP50-95": float(best["metrics/mAP50-95(B)"]),
        },
        "training_minutes": round(float(last["time"]) / 60, 1),
        "settings": {key: args[key] for key in ("model", "epochs", "batch", "imgsz", "device")},
    }


def main() -> None:
    export = {"export_confidence": EXPORT_CONFIDENCE, "draw_confidence": DRAW_CONFIDENCE, "models": {}}
    for name, (weights, folders) in MODELS.items():
        model = YOLO(str(weights))
        entry = {"weights": weights.name, "classes": len(model.names), "images": {}}
        for folder in folders:
            for image in sorted((IMAGES / folder).glob("*")):
                key = f"{folder}/{image.name}"
                entry["images"][key] = detect(model, image, EXPORT_CONFIDENCE)
                draw(model, image, RESULTS / "predictions" / name / folder)
                print(f"{name:<11} {key:<40} {len(entry['images'][key]['detections'])} boxes")
        export["models"][name] = entry
    export["training"] = export_training()
    (RESULTS / "detections.json").write_text(json.dumps(export, indent=1) + "\n")
    print(f"\nwrote {RESULTS / 'detections.json'}")


if __name__ == "__main__":
    main()
