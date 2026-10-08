from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path=None):
    config_path = Path(path) if path else ROOT / "config.yaml"
    with config_path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def draw_results(frame, results):
    for (x, y, w, h), name, score in results:
        known = name != "Desconhecido"
        color = (80, 180, 90) if known else (70, 70, 210)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        label = f"{name} {score:.2f}"
        cv2.putText(
            frame,
            label,
            (x, max(24, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    return frame
