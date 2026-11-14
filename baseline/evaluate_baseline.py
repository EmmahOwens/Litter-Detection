"""Baseline Single-Pass Evaluation Script on Sample Street Images."""
from pathlib import Path
import sys
import time
import cv2
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from proposed.conditional_sahi import LitterSegmentationEngine, run_litter_pipeline


def main() -> None:
    engine = LitterSegmentationEngine(ROOT_DIR / "models")
    sample_imgs = sorted((ROOT_DIR / "sample_data" / "taco_test_samples").glob("*.jpg"))
    rows = []
    for idx, img_path in enumerate(sample_imgs):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        out = run_litter_pipeline(
            engine,
            img_bgr,
            backend="pytorch",
            pipeline_mode="baseline",
            scene_idx=idx,
            image_name=img_path.name,
        )
        rep = out["ulsi_report"]
        tel = out["telemetry"]
        rows.append(
            {
                "image": img_path.name,
                "mode": "baseline_single_pass",
                "total_items": rep["total_items"],
                "small_objects": rep["small_objects_count"],
                "mask_area_pct": rep["total_mask_coverage_pct"],
                "ulsi_score": rep["ulsi_severity_score"],
                "latency_ms": tel["total_latency_ms"],
                "fps": tel["throughput_fps"],
            }
        )
    df = pd.DataFrame(rows)
    print("=== BASELINE SINGLE-PASS EVALUATION SUMMARY ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
