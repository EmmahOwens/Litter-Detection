"""Automated PC Deployment Benchmark & Visual Evidence Generator for Capstone D3.

Measures on the target user-end PC/laptop:
1. Hardware, OS, and Python runtime environment specifications (Question 3h).
2. PyTorch (`best.pt`) vs. ONNX Runtime (`best.onnx`) CPU inference latency & FPS (Question 3f/3h).
3. Baseline (Single-Pass) vs. Proposed D3 (DIP + Uncertainty-Gated 2x2 SAHI + Calibrated)
   vs. Unconditional 2x2 SAHI (5 Passes) across TACO and external pLitterStreet scenes.
4. Generates visual 3-stage pipeline and Baseline-vs-Proposed figures in `outputs/`.
"""
import json
import platform
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import pandas as pd
import psutil
import torch
import ultralytics

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from proposed.conditional_sahi import LitterSegmentationEngine, run_litter_pipeline
from proposed.ulsi_triage import build_kcca_gis_artifacts


def collect_pc_specs() -> Dict[str, Any]:
    """Gather detailed hardware and software specifications of the user-end PC."""
    cpu_model = platform.processor() or platform.machine()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if "model name" in line.lower():
                cpu_model = line.split(":", 1)[1].strip()
                break
    except Exception:
        pass

    vm = psutil.virtual_memory()
    return {
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "architecture": platform.machine(),
        "processor": cpu_model,
        "physical_cores": psutil.cpu_count(logical=False) or 2,
        "logical_threads": psutil.cpu_count(logical=True) or 4,
        "total_ram_gib": round(vm.total / (1024 ** 3), 2),
        "available_ram_gib": round(vm.available / (1024 ** 3), 2),
        "gpu_accelerated": torch.cuda.is_available(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "onnxruntime_version": ort.__version__,
        "onnxruntime_providers": ort.get_available_providers(),
        "ultralytics_version": ultralytics.__version__,
        "opencv_version": cv2.__version__,
        "pytorch_model_mb": round((ROOT_DIR / "models" / "best.pt").stat().st_size / (1024 ** 2), 2),
        "onnx_model_mb": round((ROOT_DIR / "models" / "best.onnx").stat().st_size / (1024 ** 2), 2),
    }


def main() -> None:
    out_dir = ROOT_DIR / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = collect_pc_specs()
    (out_dir / "pc_environment_specs.json").write_text(json.dumps(specs, indent=2))
    print("=== TARGET USER-END PC ENVIRONMENT SPECIFICATIONS ===")
    for k, v in specs.items():
        print(f"  {k:24s}: {v}")

    engine = LitterSegmentationEngine(ROOT_DIR / "models")
    taco_imgs = sorted((ROOT_DIR / "sample_data" / "taco_test_samples").glob("*.jpg"))
    plitter_imgs = sorted((ROOT_DIR / "sample_data" / "plitterstreet_samples").glob("*"))
    all_samples = [("TACO", p) for p in taco_imgs] + [("pLitterStreet", p) for p in plitter_imgs]

    # Warm up both backends once so initialization overhead is excluded from steady-state timing
    warm_bgr = cv2.imread(str(taco_imgs[0]))
    _ = run_litter_pipeline(engine, warm_bgr, backend="pytorch", pipeline_mode="baseline")
    _ = run_litter_pipeline(engine, warm_bgr, backend="onnx", pipeline_mode="baseline")

    configs = [
        ("PyTorch_Baseline_SinglePass", "pytorch", "baseline"),
        ("ONNX_Baseline_SinglePass", "onnx", "baseline"),
        ("PyTorch_Proposed_Conditional_SAHI", "pytorch", "proposed_conditional"),
        ("ONNX_Proposed_Conditional_SAHI", "onnx", "proposed_conditional"),
        ("ONNX_Unconditional_SAHI_5Pass", "onnx", "unconditional_sahi"),
    ]

    summary_rows: List[Dict[str, Any]] = []
    per_scene_rows: List[Dict[str, Any]] = []
    gis_records: List[Dict[str, Any]] = []

    for cfg_name, backend, mode in configs:
        latencies, pre_ms_list, inf_ms_list, post_ms_list = [], [], [], []
        passes_list, total_dets, small_dets, tile_dets = [], [], [], []
        ulsi_list = []

        for idx, (dataset_tag, img_path) in enumerate(all_samples):
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue
            res = run_litter_pipeline(
                engine,
                img_bgr,
                backend=backend,
                pipeline_mode=mode,
                scene_idx=idx,
                image_name=img_path.name,
            )
            tel = res["telemetry"]
            rep = res["ulsi_report"]

            latencies.append(tel["total_latency_ms"])
            pre_ms_list.append(tel["preprocess_ms"])
            inf_ms_list.append(tel["inference_ms"])
            post_ms_list.append(tel["postprocess_ms"])
            passes_list.append(tel["passes_executed"])
            total_dets.append(rep["total_items"])
            small_dets.append(rep["small_objects_count"])
            tile_dets.append(tel["tile_recovered_count"])
            ulsi_list.append(rep["ulsi_severity_score"])

            if cfg_name in ("ONNX_Baseline_SinglePass", "ONNX_Proposed_Conditional_SAHI"):
                per_scene_rows.append(
                    {
                        "Config": cfg_name,
                        "Dataset": dataset_tag,
                        "Image": img_path.name,
                        "Total_Items": rep["total_items"],
                        "Small_Items": rep["small_objects_count"],
                        "Tile_Recovered": tel["tile_recovered_count"],
                        "Mask_Area_Pct": rep["total_mask_coverage_pct"],
                        "ULSI_Score": rep["ulsi_severity_score"],
                        "Priority": rep["cleanup_priority"],
                        "Triage": rep["triage_status"],
                        "Passes": tel["passes_executed"],
                        "Latency_ms": tel["total_latency_ms"],
                    }
                )
            if cfg_name == "ONNX_Proposed_Conditional_SAHI":
                gis_records.append(rep)

        mean_lat = float(np.mean(latencies))
        summary_rows.append(
            {
                "Configuration": cfg_name,
                "Backend": "ONNX Runtime" if backend == "onnx" else "PyTorch FP32",
                "Pipeline_Mode": mode,
                "Avg_Passes_Per_Img": round(float(np.mean(passes_list)), 2),
                "Avg_Preprocess_ms": round(float(np.mean(pre_ms_list)), 2),
                "Avg_Inference_ms": round(float(np.mean(inf_ms_list)), 2),
                "Avg_Postprocess_ms": round(float(np.mean(post_ms_list)), 2),
                "Avg_Total_Latency_ms": round(mean_lat, 2),
                "Throughput_FPS": round(1000.0 / max(mean_lat, 1e-3), 2),
                "Total_Detections_11_Scenes": int(sum(total_dets)),
                "Small_Objects_Detected": int(sum(small_dets)),
                "SAHI_Tile_Recovered": int(sum(tile_dets)),
                "Mean_ULSI_Score": round(float(np.mean(ulsi_list)), 1),
            }
        )

    bench_df = pd.DataFrame(summary_rows)
    scene_df = pd.DataFrame(per_scene_rows)
    bench_df.to_csv(out_dir / "pc_deployment_benchmark.csv", index=False)
    scene_df.to_csv(out_dir / "sample_scene_comparison.csv", index=False)

    print("\n=== PC DEPLOYMENT BENCHMARK SUMMARY (11 SCENES: 6 TACO + 5 pLitterStreet) ===")
    print(bench_df.to_string(index=False))

    # Build KCCA GIS artifacts in outputs/
    build_kcca_gis_artifacts(gis_records, out_dir)

    # Save visual demonstration figures for README and D3 Report
    hero_path = ROOT_DIR / "sample_data" / "taco_test_samples" / "batch_10_000010.jpg"
    if not hero_path.exists():
        hero_path = taco_imgs[0]
    hero_bgr = cv2.imread(str(hero_path))

    base_hero = run_litter_pipeline(
        engine, hero_bgr, backend="onnx", pipeline_mode="baseline", image_name=hero_path.name
    )
    prop_hero = run_litter_pipeline(
        engine,
        hero_bgr,
        backend="onnx",
        pipeline_mode="proposed_conditional",
        image_name=hero_path.name,
    )

    # Figure 1: 3-Stage Processing Pipeline Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8))
    axes[0].imshow(prop_hero["raw_rgb"])
    axes[0].set_title(f"Stage 1: Raw Input\n({hero_path.name})", fontweight="bold", fontsize=11)
    axes[0].axis("off")

    dip_gain = prop_hero["telemetry"]["dip_diagnostics"]["contrast_gain_pct"]
    axes[1].imshow(prop_hero["preprocessed_rgb"])
    axes[1].set_title(
        f"Stage 2: CIE LAB CLAHE + Shadow Gamma\n(+{dip_gain:.1f}% Local Edge Contrast)",
        fontweight="bold",
        fontsize=11,
    )
    axes[1].axis("off")

    axes[2].imshow(prop_hero["annotated_rgb"])
    axes[2].set_title(
        f"Stage 3: Proposed D3 Segmented Output\n(ULSI={prop_hero['ulsi_report']['ulsi_severity_score']}, Items={prop_hero['ulsi_report']['total_items']})",
        fontweight="bold",
        fontsize=11,
    )
    axes[2].axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "demo_3stage_pipeline.png", dpi=180)
    plt.close(fig)

    # Figure 2: Baseline vs. Proposed D3 Side-by-Side Comparison
    fig2, axes2 = plt.subplots(1, 2, figsize=(15, 6.2))
    axes2[0].imshow(base_hero["annotated_rgb"])
    axes2[0].set_title(
        f"Baseline (Single-Pass, No DIP)\nDetected: {base_hero['ulsi_report']['total_items']} items "
        f"(Small <32px: {base_hero['ulsi_report']['small_objects_count']})",
        fontweight="bold",
        fontsize=12,
    )
    axes2[0].axis("off")

    axes2[1].imshow(prop_hero["annotated_rgb"])
    axes2[1].set_title(
        f"Proposed D3 (LAB-CLAHE + Conditional SAHI + Calibrated)\n"
        f"Detected: {prop_hero['ulsi_report']['total_items']} items "
        f"(Small <32px: {prop_hero['ulsi_report']['small_objects_count']})",
        fontweight="bold",
        fontsize=12,
    )
    axes2[1].axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "demo_baseline_vs_proposed.png", dpi=180)
    plt.close(fig2)

    print(f"\nSaved visual evidence and benchmark artifacts to: {out_dir}")


if __name__ == "__main__":
    main()
