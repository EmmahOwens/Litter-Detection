"""Capstone Deliverable 3 — Upgraded Municipal Litter Inspection Workstation (PC Deployment).

Features a 4-Tab Executive Gradio Web Interface:
- Tab 1: Interactive 3-Stage Pipeline Workstation (Raw -> LAB-CLAHE -> Segmented Output)
         + Executive HTML KPI Cards + 4 Visual X-Ray Overlay Modes + Zoomed Instance Crop Gallery
         + Object-Level Detection Table + Interactive Human-in-the-Loop (HITL) Inspector Sign-Off
         & Printable HTML Inspection Certificate Exporter.
- Tab 2: Live Baseline vs. Proposed D3 Side-by-Side Comparison with Quantitative Improvement Delta.
- Tab 3: Batch Municipal Street Audit & Interactive KCCA Folium GIS Hotspot Exporter.
- Tab 4: Street Patrol Video / Dashcam Sequence Screener with ULSI Severity Timeline & Peak Hotspot Finder.
"""
from datetime import datetime
import json
import os
import platform
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from proposed.conditional_sahi import LitterSegmentationEngine, run_litter_pipeline
from proposed.ulsi_triage import (
    CALIBRATED_CLASS_THRESHOLDS,
    KCCA_ZONES,
    build_kcca_gis_artifacts,
)

ENGINE = LitterSegmentationEngine(ROOT_DIR / "models")
OUTPUTS_DIR = ROOT_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_TRAIL_CSV = OUTPUTS_DIR / "hitl_verified_audit_trail.csv"


def _discover_sample_images() -> Dict[str, Path]:
    """Return a mapping of human-friendly sample names to file paths."""
    samples: Dict[str, Path] = {}
    taco_dir = ROOT_DIR / "sample_data" / "taco_test_samples"
    plitter_dir = ROOT_DIR / "sample_data" / "plitterstreet_samples"

    if taco_dir.exists():
        for p in sorted(taco_dir.glob("*")):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                samples[f"[TACO Benchmark] {p.name}"] = p
    if plitter_dir.exists():
        for p in sorted(plitter_dir.glob("*")):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                samples[f"[pLitterStreet External] {p.name}"] = p
    return samples


SAMPLE_MAP = _discover_sample_images()
SAMPLE_CHOICES = list(SAMPLE_MAP.keys())

MATERIAL_LEGEND_HTML = """
<div style="display:flex; flex-wrap:wrap; gap:10px; align-items:center; background:#0f172a; color:#f8fafc; padding:10px 16px; border-radius:10px; margin:6px 0; font-size:13px;">
  <span style="font-weight:700; margin-right:6px;">5-Class Harmonized Material Legend & Calibrated Thresholds:</span>
  <span style="background:#1e73eb; color:#fff; padding:3px 10px; border-radius:999px; font-weight:600;">Plastic (th=0.31, w=1.3)</span>
  <span style="background:#f5821e; color:#fff; padding:3px 10px; border-radius:999px; font-weight:600;">Metal (th=0.34, w=1.2)</span>
  <span style="background:#d23cb4; color:#fff; padding:3px 10px; border-radius:999px; font-weight:600;">Glass Hazard (th=0.26, w=1.4)</span>
  <span style="background:#3cc332; color:#fff; padding:3px 10px; border-radius:999px; font-weight:600;">Paper (th=0.19, w=0.8)</span>
  <span style="background:#ebc828; color:#111; padding:3px 10px; border-radius:999px; font-weight:600;">Other / Butts (th=0.20, w=1.1)</span>
</div>
"""


def get_pc_environment_banner() -> str:
    """Return a concise hardware/software summary of the local PC deployment environment."""
    cpu_name = platform.processor() or platform.machine()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if "model name" in line.lower():
                cpu_name = line.split(":", 1)[1].strip()
                break
    except Exception:
        pass
    ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)
    os_str = f"{platform.system()} {platform.release()}"
    py_ver = platform.python_version()
    return (
        f"**Deployed User-End Device:** `{cpu_name}` | **RAM:** `{ram_gb} GiB` | "
        f"**OS:** `{os_str}` | **Python:** `{py_ver}` | "
        f"**Web Export Model:** `best.onnx` (`ONNX Runtime Opset 18`, 39.8 MB) & `best.pt` (`PyTorch FP32`, 22.3 MB)"
    )


def load_sample_to_pil(sample_label: str) -> Optional[Image.Image]:
    """Load a selected sample photograph into the Gradio image input widget."""
    if not sample_label or sample_label not in SAMPLE_MAP:
        return None
    img_path = SAMPLE_MAP[sample_label]
    if not img_path.exists():
        return None
    return Image.open(img_path).convert("RGB")


def _mode_code_from_ui(ui_mode: str) -> str:
    low = ui_mode.lower()
    if "baseline" in low:
        return "baseline"
    if "smart 2-tile" in low or "2-tile" in low:
        return "proposed_smart_2tile"
    if "unconditional" in low:
        return "unconditional_sahi"
    return "proposed_conditional"


def _overlay_code_from_ui(ui_overlay: str) -> str:
    low = ui_overlay.lower()
    if "x-ray" in low or "sahi" in low:
        return "sahi_grid_xray"
    if "spotlight" in low:
        return "mask_spotlight"
    if "heatmap" in low or "sobel" in low:
        return "dip_edge_heatmap"
    return "standard"


def build_kpi_cards_html(rep: Dict[str, Any], tel: Dict[str, Any]) -> str:
    """Render executive HTML KPI cards for ULSI score, priority, HITL triage, and speed."""
    ulsi = float(rep["ulsi_severity_score"])
    prio = rep["cleanup_priority"]
    triage = rep["triage_status"]

    prio_colors = {
        "Critical Dispatch": ("#dc2626", "#fef2f2"),
        "High Priority": ("#ea580c", "#fff7ed"),
        "Moderate Sweep": ("#2563eb", "#eff6ff"),
        "Low / Routine": ("#16a34a", "#f0fdf4"),
    }
    p_fg, p_bg = prio_colors.get(prio, ("#2563eb", "#eff6ff"))
    t_fg = "#b45309" if "Flagged" in triage else "#15803d"
    t_bg = "#fffbeb" if "Flagged" in triage else "#f0fdf4"

    dip_gain = tel["dip_diagnostics"]["contrast_gain_pct"]
    sahi_badge = (
        f"Triggered ({tel['passes_executed']} passes)"
        if tel["sahi_triggered"]
        else "Fast 1-Pass (Skipped)"
    )

    return f"""
    <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(210px, 1fr)); gap:12px; margin:8px 0;">
      <div style="background:{p_bg}; border-left:5px solid {p_fg}; padding:12px 14px; border-radius:8px; box-shadow:0 1px 3px rgba(0,0,0,0.08);">
        <div style="font-size:12px; color:#475569; font-weight:600; text-transform:uppercase;">ULSI Severity & Priority</div>
        <div style="font-size:24px; font-weight:800; color:{p_fg}; margin:4px 0;">{ulsi:.1f} <span style="font-size:14px; font-weight:600;">/ 100</span></div>
        <div style="background:#e2e8f0; height:7px; border-radius:4px; overflow:hidden; margin-bottom:6px;">
          <div style="background:{p_fg}; width:{min(100.0, ulsi):.1f}%; height:100%;"></div>
        </div>
        <div style="font-size:13px; font-weight:700; color:{p_fg};">{prio}</div>
      </div>

      <div style="background:{t_bg}; border-left:5px solid {t_fg}; padding:12px 14px; border-radius:8px; box-shadow:0 1px 3px rgba(0,0,0,0.08);">
        <div style="font-size:12px; color:#475569; font-weight:600; text-transform:uppercase;">Human-in-the-Loop Triage</div>
        <div style="font-size:16px; font-weight:800; color:{t_fg}; margin:6px 0;">{triage}</div>
        <div style="font-size:12px; color:#334155;">Uncertain/Hazard Items: <b>{rep['uncertain_items_count']}</b></div>
        <div style="font-size:12px; color:#15803d; font-weight:600;">Screening Time Saved: {rep['screening_time_saved_pct']:.1f}%</div>
      </div>

      <div style="background:#f8fafc; border-left:5px solid #0284c7; padding:12px 14px; border-radius:8px; box-shadow:0 1px 3px rgba(0,0,0,0.08);">
        <div style="font-size:12px; color:#475569; font-weight:600; text-transform:uppercase;">Litter & Small-Object Recovery</div>
        <div style="font-size:22px; font-weight:800; color:#0f172a; margin:4px 0;">{rep['total_items']} <span style="font-size:13px; font-weight:600; color:#0284c7;">Items ({rep['small_objects_count']} Small &lt;32px)</span></div>
        <div style="font-size:12px; color:#334155;">SAHI Tile-Recovered: <b>{tel['tile_recovered_count']}</b> | Mask Area: <b>{rep['total_mask_coverage_pct']:.3f}%</b></div>
      </div>

      <div style="background:#f8fafc; border-left:5px solid #7c3aed; padding:12px 14px; border-radius:8px; box-shadow:0 1px 3px rgba(0,0,0,0.08);">
        <div style="font-size:12px; color:#475569; font-weight:600; text-transform:uppercase;">PC Execution Telemetry</div>
        <div style="font-size:20px; font-weight:800; color:#6d28d9; margin:4px 0;">{tel['total_latency_ms']:.0f} ms <span style="font-size:13px; font-weight:600;">({tel['throughput_fps']:.2f} FPS)</span></div>
        <div style="font-size:12px; color:#334155;">SAHI Gate: <b>{sahi_badge}</b> | DIP Edge Contrast: <b>+{dip_gain:.1f}%</b></div>
      </div>
    </div>
    """


def run_tab1_inspection(
    pil_img: Optional[Image.Image],
    sample_label: str,
    backend_choice: str,
    pipeline_choice: str,
    overlay_choice: str,
    apply_dip: bool,
    use_calibrated: bool,
    conf_override_enabled: bool,
    conf_slider_val: float,
) -> Tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
    str,
    List[Tuple[np.ndarray, str]],
    pd.DataFrame,
    str,
    str,
    str,
]:
    """Execute Tab 1 inspection and populate all visual panels, gallery, table, and KPI cards."""
    img_name = "uploaded_street_photo.jpg"
    if pil_img is None and sample_label in SAMPLE_MAP:
        pil_img = Image.open(SAMPLE_MAP[sample_label]).convert("RGB")
        img_name = SAMPLE_MAP[sample_label].name
    elif sample_label in SAMPLE_MAP:
        img_name = SAMPLE_MAP[sample_label].name

    empty_df = pd.DataFrame(
        columns=[
            "ID",
            "Material",
            "Confidence",
            "Calibrated_Thresh",
            "COCO_Size_Tier",
            "Mask_Area_%",
            "Pass_Origin",
            "HITL_Flag",
        ]
    )
    if pil_img is None:
        return (
            None,
            None,
            "<p>Please upload a street photograph or select a sample image.</p>",
            [],
            empty_df,
            "Please select an image.",
            "{}",
            "Moderate Sweep",
        )

    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    mode_code = _mode_code_from_ui(pipeline_choice)
    overlay_code = _overlay_code_from_ui(overlay_choice)
    backend_code = "onnx" if "onnx" in backend_choice.lower() else "pytorch"
    conf_override = float(conf_slider_val) if conf_override_enabled else None

    result = run_litter_pipeline(
        ENGINE,
        img_bgr,
        backend=backend_code,
        pipeline_mode=mode_code,
        apply_dip_override=apply_dip,
        base_conf_override=conf_override,
        use_calibrated_thresholds=use_calibrated,
        overlay_style=overlay_code,
        scene_idx=0,
        image_name=img_name,
    )

    tel = result["telemetry"]
    rep = result["ulsi_report"]
    dip_d = tel["dip_diagnostics"]
    kpi_html = build_kpi_cards_html(rep, tel)

    det_rows = []
    for idx, d in enumerate(result["detections_table"], start=1):
        is_uncertain = d["confidence"] < 0.38 or d["material"] == "Glass"
        det_rows.append(
            {
                "ID": f"#{idx}",
                "Material": d["material"],
                "Confidence": d["confidence"],
                "Calibrated_Thresh": d["threshold_applied"],
                "COCO_Size_Tier": d["coco_size_tier"],
                "Mask_Area_%": d["mask_area_pct"],
                "Pass_Origin": "SAHI Tile" if d["source"] == "sahi_tile" else "Full Frame",
                "HITL_Flag": "Verify (Uncertain/Glass)" if is_uncertain else "Auto-Verified",
            }
        )
    det_df = pd.DataFrame(det_rows) if det_rows else empty_df

    mat_lines = []
    for m, cnt in sorted(rep["counts_by_material"].items()):
        area_m = rep["mask_area_pct_by_material"].get(m, 0.0)
        cal_th = CALIBRATED_CLASS_THRESHOLDS.get(m, 0.25)
        mat_lines.append(
            f"  - **{m}**: `{cnt}` item(s) | Mask Surface Area: `{area_m:.3f}%` (Calibrated Thresh: `{cal_th:.2f}`)"
        )
    if not mat_lines:
        mat_lines.append("  - *No litter items detected above threshold (Clean Street Scene).*")

    md_summary = f"""#### Stage-by-Stage Pipeline Telemetry (`{tel['backend']}`)
- **Stage 1 (Working Scale Normalization & Classical DIP):** `{tel['preprocess_ms']:.2f} ms` | Gamma: `{dip_d['applied_gamma']}` | Luminance: `{dip_d['raw_mean_l']} -> {dip_d['enh_mean_l']}` | Sobel Local Edge Contrast Gain: **`+{dip_d['contrast_gain_pct']:.2f}%`**
- **Stage 2 & 3 (Feature Extraction & Conditional SAHI Inference):** `{tel['inference_ms']:.2f} ms` | Passes Executed: `{tel['passes_executed']}` | **Gate Reason:** `{tel['sahi_gate_reason']}`
- **Stage 4 (Class-Aware NMS, Mask Stitching & F1 Calibration):** `{tel['postprocess_ms']:.2f} ms` | Total Latency: **`{tel['total_latency_ms']:.2f} ms`** (`{tel['throughput_fps']:.2f} FPS`)

#### Material Breakdown
{chr(10).join(mat_lines)}
"""

    export_payload = {
        "telemetry": tel,
        "municipal_decision_support": rep,
    }
    return (
        result["preprocessed_rgb"],
        result["annotated_rgb"],
        kpi_html,
        result["instance_crops"],
        det_df,
        md_summary,
        json.dumps(export_payload, indent=2),
        rep["cleanup_priority"],
    )


def submit_inspector_signoff(
    inspector_name: str,
    kcca_zone: str,
    approved_priority: str,
    inspector_decision: str,
    inspector_notes: str,
    latest_json_str: str,
) -> Tuple[str, Optional[str], pd.DataFrame]:
    """Log the Human-in-the-Loop inspector verification and generate a printable HTML certificate."""
    try:
        payload = json.loads(latest_json_str) if latest_json_str else {}
    except Exception:
        payload = {}

    rep = payload.get("municipal_decision_support", {})
    tel = payload.get("telemetry", {})
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    insp_id = inspector_name.strip() or "KCCA-INSP-01"

    record = {
        "Timestamp": timestamp,
        "Inspector": insp_id,
        "Image": rep.get("image", "street_scene.jpg"),
        "KCCA_Zone": kcca_zone,
        "Total_Items": rep.get("total_items", 0),
        "Small_Items": rep.get("small_objects_count", 0),
        "Mask_Area_%": rep.get("total_mask_coverage_pct", 0.0),
        "Model_ULSI_Score": rep.get("ulsi_severity_score", 0.0),
        "Model_Priority": rep.get("cleanup_priority", "N/A"),
        "Inspector_Approved_Priority": approved_priority,
        "HITL_Action": inspector_decision,
        "Notes": inspector_notes.strip() or "Verified via D3 PC Workstation.",
    }

    if AUDIT_TRAIL_CSV.exists():
        existing_df = pd.read_csv(AUDIT_TRAIL_CSV)
        updated_df = pd.concat([existing_df, pd.DataFrame([record])], ignore_index=True)
    else:
        updated_df = pd.DataFrame([record])
    updated_df.to_csv(AUDIT_TRAIL_CSV, index=False)

    cert_path = OUTPUTS_DIR / "latest_inspection_certificate.html"
    cert_html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>KCCA Municipal Litter Inspection Certificate — {record['Image']}</title>
  <style>
    body {{ font-family: Arial, sans-serif; max-width: 780px; margin: 24px auto; color: #0f172a; line-height: 1.5; }}
    .header {{ background: #0f172a; color: #fff; padding: 18px 24px; border-radius: 8px; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin: 16px 0; }}
    .box {{ border: 1px solid #cbd5e1; border-radius: 8px; padding: 12px 16px; background: #f8fafc; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
    th, td {{ border: 1px solid #cbd5e1; padding: 8px 10px; text-align: left; font-size: 13px; }}
    th {{ background: #e2e8f0; }}
  </style>
</head>
<body>
  <div class="header">
    <h2 style="margin:0;">Municipal Litter Inspection & HITL Triage Certificate</h2>
    <p style="margin:4px 0 0 0; font-size:13px;">Soroti University Capstone D3 — Urban Litter Instance Segmentation System</p>
  </div>
  <div class="grid">
    <div class="box">
      <b>Timestamp:</b> {timestamp}<br/>
      <b>Verified By Inspector:</b> {insp_id}<br/>
      <b>Assigned KCCA Zone:</b> {kcca_zone}<br/>
      <b>HITL Decision:</b> {inspector_decision}
    </div>
    <div class="box">
      <b>Inspected Image:</b> {record['Image']}<br/>
      <b>ULSI Severity Score:</b> {record['Model_ULSI_Score']} / 100<br/>
      <b>Approved Dispatch Priority:</b> <span style="color:#dc2626;font-weight:bold;">{approved_priority}</span><br/>
      <b>Model Runtime:</b> {tel.get('backend', 'ONNX Runtime')} ({tel.get('total_latency_ms', 0)} ms)
    </div>
  </div>
  <div class="box">
    <b>Inspector Field Notes:</b> {record['Notes']}
  </div>
</body>
</html>"""
    cert_path.write_text(cert_html)

    status_msg = (
        f"✅ **Logged HITL Verification at `{timestamp}`** by `{insp_id}` | "
        f"Approved Priority: **`{approved_priority}`** | Saved to `outputs/hitl_verified_audit_trail.csv`"
    )
    return status_msg, str(cert_path), updated_df


def run_tab2_comparison(
    pil_img: Optional[Image.Image],
    sample_label: str,
    backend_choice: str,
    proposed_variant: str,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    """Run Baseline vs. Proposed D3 side-by-side on the same image and quantify improvements."""
    img_name = "comparison_photo.jpg"
    if pil_img is None and sample_label in SAMPLE_MAP:
        pil_img = Image.open(SAMPLE_MAP[sample_label]).convert("RGB")
        img_name = SAMPLE_MAP[sample_label].name
    elif sample_label in SAMPLE_MAP:
        img_name = SAMPLE_MAP[sample_label].name

    if pil_img is None:
        return None, None, "Please select a sample image or upload a photograph to compare."

    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    backend_code = "onnx" if "onnx" in backend_choice.lower() else "pytorch"
    prop_mode = (
        "proposed_smart_2tile"
        if "2-tile" in proposed_variant.lower()
        else "proposed_conditional"
    )

    base_out = run_litter_pipeline(
        ENGINE,
        img_bgr,
        backend=backend_code,
        pipeline_mode="baseline",
        apply_dip_override=False,
        use_calibrated_thresholds=False,
        scene_idx=0,
        image_name=img_name,
    )
    prop_out = run_litter_pipeline(
        ENGINE,
        img_bgr,
        backend=backend_code,
        pipeline_mode=prop_mode,
        apply_dip_override=True,
        use_calibrated_thresholds=True,
        scene_idx=0,
        image_name=img_name,
    )

    b_rep, p_rep = base_out["ulsi_report"], prop_out["ulsi_report"]
    b_tel, p_tel = base_out["telemetry"], prop_out["telemetry"]

    delta_items = p_rep["total_items"] - b_rep["total_items"]
    delta_small = p_rep["small_objects_count"] - b_rep["small_objects_count"]
    delta_area = round(p_rep["total_mask_coverage_pct"] - b_rep["total_mask_coverage_pct"], 3)
    delta_ulsi = round(p_rep["ulsi_severity_score"] - b_rep["ulsi_severity_score"], 1)

    comparison_md = f"""### Controlled Side-by-Side Comparison (`{p_tel['backend']}` on `{img_name}`)

| Metric / Component | Baseline System (D2 Single-Pass) | Proposed D3 System (`{p_tel['pipeline_mode']}`) | Measured Improvement / Delta |
| :--- | :--- | :--- | :--- |
| **Classical DIP Preprocessing** | Disabled (Raw BGR) | **CIE LAB CLAHE + Adaptive Gamma (`+{p_tel['dip_diagnostics']['contrast_gain_pct']:.1f}%` edge contrast)** | Enhances shadowed/translucent litter edges |
| **Inference Strategy** | Single-Pass 640px (`1 pass`) | **Uncertainty-Gated SAHI (`{p_tel['passes_executed']} pass(es)`)** | `{p_tel['sahi_gate_reason']}` |
| **Confidence Filtering** | Uniform (`conf >= 0.25`) | **Validation F1-Calibrated Per-Class Thresholds** | Recovers minority `Glass` (`0.26`), `Paper` (`0.19`), `Other` (`0.20`) |
| **Total Litter Detected** | `{b_rep['total_items']}` item(s) | **`{p_rep['total_items']}` item(s)** | **`{delta_items:+d}` item(s)** |
| **Small Objects (<32x32 px)** | `{b_rep['small_objects_count']}` item(s) | **`{p_rep['small_objects_count']}` item(s)** | **`{delta_small:+d}` small object(s) (`{p_tel['tile_recovered_count']}` via SAHI tiles)** |
| **Segmented Mask Area (%)** | `{b_rep['total_mask_coverage_pct']:.3f}%` | **`{p_rep['total_mask_coverage_pct']:.3f}%`** | **`{delta_area:+.3f}%` pixel coverage** |
| **ULSI Severity Score (0–100)** | `{b_rep['ulsi_severity_score']}` (`{b_rep['cleanup_priority']}`) | **`{p_rep['ulsi_severity_score']}` (`{p_rep['cleanup_priority']}`)** | **`{delta_ulsi:+.1f}` severity points** |
| **Human-in-the-Loop Triage** | `{b_rep['triage_status']}` | **`{p_rep['triage_status']}`** | `{p_rep['screening_time_saved_pct']:.1f}%` manual screening time saved |
| **End-to-End PC Latency** | `{b_tel['total_latency_ms']:.1f} ms` (`{b_tel['throughput_fps']:.2f} FPS`) | **`{p_tel['total_latency_ms']:.1f} ms` (`{p_tel['throughput_fps']:.2f} FPS`)** | Adaptive speed–accuracy trade-off |
"""
    return base_out["annotated_rgb"], prop_out["annotated_rgb"], comparison_md


def run_tab3_batch_audit(
    backend_choice: str,
    pipeline_choice: str,
) -> Tuple[str, pd.DataFrame, str, str]:
    """Run batch inspection across all bundled sample street photos and generate KCCA GIS outputs."""
    backend_code = "onnx" if "onnx" in backend_choice.lower() else "pytorch"
    mode_code = _mode_code_from_ui(pipeline_choice)

    records: List[Dict[str, Any]] = []
    for idx, (_, img_path) in enumerate(SAMPLE_MAP.items()):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        out = run_litter_pipeline(
            ENGINE,
            img_bgr,
            backend=backend_code,
            pipeline_mode=mode_code,
            scene_idx=idx,
            image_name=img_path.name,
        )
        rep = out["ulsi_report"]
        tel = out["telemetry"]
        rep["sahi_triggered"] = tel["sahi_triggered"]
        rep["passes_executed"] = tel["passes_executed"]
        rep["latency_ms"] = tel["total_latency_ms"]
        records.append(rep)

    map_html, csv_path, geojson_path = build_kcca_gis_artifacts(records, OUTPUTS_DIR)

    df_display = pd.DataFrame(
        [
            {
                "Image": r["image"],
                "KCCA Division": r["kcca_division"],
                "Total Items": r["total_items"],
                "Small (<32px)": r["small_objects_count"],
                "Mask Area (%)": r["total_mask_coverage_pct"],
                "ULSI Score": r["ulsi_severity_score"],
                "Priority": r["cleanup_priority"],
                "HITL Triage": r["triage_status"],
                "SAHI Passes": f"{r.get('passes_executed', 1)} pass(es)",
                "Latency (ms)": r.get("latency_ms", 0.0),
            }
            for r in records
        ]
    )

    iframe_html = (
        '<div style="border:1px solid #cbd5e1; border-radius:8px; overflow:hidden;">'
        f"{map_html}</div>"
    )
    return iframe_html, df_display, str(csv_path), str(geojson_path)


def run_tab4_patrol_screener(
    video_path: Optional[str],
    backend_choice: str,
    pipeline_choice: str,
    frame_stride: int,
    max_keyframes: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], pd.DataFrame, str]:
    """Screen a street patrol video or simulated patrol sequence and plot the ULSI timeline."""
    backend_code = "onnx" if "onnx" in backend_choice.lower() else "pytorch"
    mode_code = _mode_code_from_ui(pipeline_choice)

    frames_to_process: List[Tuple[str, np.ndarray]] = []
    if video_path and os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        f_idx = 0
        while cap.isOpened() and len(frames_to_process) < int(max_keyframes):
            ret, frame = cap.read()
            if not ret:
                break
            if f_idx % max(1, int(frame_stride)) == 0:
                frames_to_process.append((f"Frame_{f_idx:04d}", frame))
            f_idx += 1
        cap.release()
    else:
        # Use bundled street sequence to simulate a vehicle/foot patrol sweep
        for _, p in list(SAMPLE_MAP.items())[: int(max_keyframes)]:
            im = cv2.imread(str(p))
            if im is not None:
                frames_to_process.append((p.name, im))

    if not frames_to_process:
        return None, None, pd.DataFrame(), "No frames available to process."

    timeline_rows = []
    peak_ulsi = -1.0
    peak_annotated_rgb = None
    peak_label = ""

    for idx, (label, f_bgr) in enumerate(frames_to_process):
        res = run_litter_pipeline(
            ENGINE,
            f_bgr,
            backend=backend_code,
            pipeline_mode=mode_code,
            scene_idx=idx,
            image_name=label,
        )
        rep = res["ulsi_report"]
        tel = res["telemetry"]
        ulsi_val = float(rep["ulsi_severity_score"])

        if ulsi_val > peak_ulsi:
            peak_ulsi = ulsi_val
            peak_annotated_rgb = res["annotated_rgb"]
            peak_label = label

        timeline_rows.append(
            {
                "Step": idx + 1,
                "Frame / Scene": label,
                "Total_Items": rep["total_items"],
                "Small_Items": rep["small_objects_count"],
                "Mask_Area_%": rep["total_mask_coverage_pct"],
                "ULSI_Score": ulsi_val,
                "Priority": rep["cleanup_priority"],
                "HITL_Triage": rep["triage_status"],
                "Passes": tel["passes_executed"],
                "Latency_ms": tel["total_latency_ms"],
            }
        )

    df_patrol = pd.DataFrame(timeline_rows)

    # Render ULSI Severity Timeline Plot
    fig, ax = plt.subplots(figsize=(10, 4.2))
    steps = df_patrol["Step"].values
    scores = df_patrol["ULSI_Score"].values
    ax.plot(steps, scores, marker="o", linewidth=2.5, color="#1d4ed8", label="ULSI Severity (0-100)")
    ax.fill_between(steps, scores, color="#93c5fd", alpha=0.30)
    ax.axhline(70, color="#dc2626", linestyle="--", linewidth=1.2, label="Critical Dispatch (70)")
    ax.axhline(45, color="#ea580c", linestyle="--", linewidth=1.2, label="High Priority (45)")
    ax.axhline(20, color="#16a34a", linestyle="--", linewidth=1.2, label="Moderate Sweep (20)")
    ax.set_xticks(steps)
    ax.set_xticklabels(
        [str(x)[:14] for x in df_patrol["Frame / Scene"]], rotation=25, ha="right", fontsize=8
    )
    ax.set_ylim(0, 105)
    ax.set_ylabel("ULSI Severity Score")
    ax.set_title("Street Patrol ULSI Litter Severity Timeline & Dispatch Thresholds", fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)

    summary_md = (
        f"### Patrol Sequence Screening Summary\n"
        f"- **Keyframes Screened:** `{len(df_patrol)}` | **Mean ULSI Score:** `{df_patrol['ULSI_Score'].mean():.1f} / 100`\n"
        f"- **Peak Litter Hotspot Identified:** `{peak_label}` (**ULSI = `{peak_ulsi:.1f}`**)\n"
        f"- **Critical / High Priority Stops:** `{int((df_patrol['ULSI_Score'] >= 45).sum())}` of `{len(df_patrol)}` stops"
    )
    return buf, peak_annotated_rgb, df_patrol, summary_md


def build_gradio_app() -> gr.Blocks:
    """Construct the 4-Tab Municipal Litter Workstation for Capstone Deliverable 3."""
    default_sample = SAMPLE_CHOICES[0] if SAMPLE_CHOICES else None
    kcca_zone_names = [z[0] for z in KCCA_ZONES]

    with gr.Blocks(
        title="Urban Litter Detection & Municipal Severity Workstation — Capstone D3"
    ) as demo:
        gr.Markdown(
            "# Urban Litter Instance Segmentation & Municipal Decision-Support Workstation (Capstone D3)\n"
            "**Proposed Improvement:** Classical DIP (`CIE LAB CLAHE + Adaptive Gamma`) + "
            "`YOLO26s-seg` (`STAL + C2PSA + Train-Only Copy-Paste`) + "
            "**Uncertainty-Gated Conditional SAHI Tiling** + **Per-Class F1 Threshold Calibration**\n\n"
            + get_pc_environment_banner()
        )
        gr.HTML(MATERIAL_LEGEND_HTML)

        with gr.Tabs():
            # ==================== TAB 1: PIPELINE WORKSTATION ====================
            with gr.TabItem("1. Interactive D3 Pipeline Workstation (3-Stage + X-Ray + HITL)"):
                with gr.Row():
                    sample_dd_t1 = gr.Dropdown(
                        choices=SAMPLE_CHOICES,
                        value=default_sample,
                        label="Pre-Loaded Benchmark Sample (TACO & pLitterStreet)",
                        scale=2,
                    )
                    backend_dd_t1 = gr.Dropdown(
                        choices=[
                            "ONNX Runtime (best.onnx — Web Deployment Format)",
                            "PyTorch FP32 (best.pt — Native Checkpoint)",
                        ],
                        value="ONNX Runtime (best.onnx — Web Deployment Format)",
                        label="Model Deployment Format (Question 3f)",
                        scale=2,
                    )
                    mode_dd_t1 = gr.Dropdown(
                        choices=[
                            "Proposed D3 (LAB-CLAHE + Uncertainty-Gated 2x2 SAHI + Calibrated)",
                            "Proposed D3 Fast (LAB-CLAHE + Smart 2-Tile Ground SAHI + Calibrated)",
                            "Unconditional 2x2 SAHI (Always 5 Passes — D2 Exp4)",
                            "Baseline (Single-Pass, No DIP, Uniform Conf=0.25)",
                        ],
                        value="Proposed D3 (LAB-CLAHE + Uncertainty-Gated 2x2 SAHI + Calibrated)",
                        label="Pipeline Operating Mode",
                        scale=3,
                    )
                    overlay_dd_t1 = gr.Dropdown(
                        choices=[
                            "Standard Annotated (Masks + Boxes + Labels)",
                            "SAHI Tile Grid & Small-Object X-Ray (Green=Tile, Cyan=Full)",
                            "Pixel Mask Spotlight (Dimmed Grayscale Background)",
                            "DIP Edge-Contrast Enhancement Heatmap (Sobel Delta)",
                        ],
                        value="Standard Annotated (Masks + Boxes + Labels)",
                        label="Visual Diagnostic Overlay Mode",
                        scale=2,
                    )

                with gr.Row():
                    dip_chk_t1 = gr.Checkbox(
                        value=True,
                        label="Enable Classical DIP Pre-Enhancement (CIE LAB CLAHE + Shadow Gamma)",
                    )
                    cal_chk_t1 = gr.Checkbox(
                        value=True,
                        label="Use Validation F1-Optimal Per-Class Thresholds",
                    )
                    conf_ovr_chk_t1 = gr.Checkbox(
                        value=False,
                        label="Override Confidence Threshold Manually",
                    )
                    conf_slider_t1 = gr.Slider(
                        0.10, 0.80, value=0.22, step=0.02, label="Manual Confidence Threshold"
                    )

                run_btn_t1 = gr.Button(
                    "Run 3-Stage Municipal Litter Inspection", variant="primary"
                )

                kpi_cards_t1 = gr.HTML(label="Executive KPI Cards")

                with gr.Row():
                    inp_img_t1 = gr.Image(
                        type="pil",
                        value=load_sample_to_pil(default_sample) if default_sample else None,
                        label="Stage 1: Raw Input Street Photograph",
                    )
                    dip_img_t1 = gr.Image(
                        label="Stage 2: Preprocessed Image (CIE LAB CLAHE + Shadow Recovery)"
                    )
                    out_img_t1 = gr.Image(
                        label="Stage 3: Instance Segmentation Masks, Material Classes & Diagnostic Overlay"
                    )

                with gr.Row():
                    crop_gallery_t1 = gr.Gallery(
                        label="Zoomed-In Detected Litter Instances (Cropped for Inspector Review)",
                        columns=4,
                        height=240,
                    )
                    det_table_t1 = gr.Dataframe(
                        label="Object-Level Detection Log (Every Segmented Instance)",
                        interactive=False,
                    )

                with gr.Row():
                    summary_md_t1 = gr.Markdown(
                        label="Pipeline Telemetry & Material Breakdown"
                    )
                    json_code_t1 = gr.Code(
                        label="Machine-Readable GIS / JSON Output", language="json"
                    )

                gr.Markdown("### Human-in-the-Loop (HITL) Inspector Verification & Sign-Off (Demo C.5)")
                with gr.Row():
                    insp_name_t1 = gr.Textbox(
                        value="Insp. Mbayo E. Owen (KCCA-04)",
                        label="Inspector Name / ID",
                    )
                    zone_dd_t1 = gr.Dropdown(
                        choices=kcca_zone_names,
                        value=kcca_zone_names[0],
                        label="Assigned KCCA Division / GPS Sector",
                    )
                    prio_dd_t1 = gr.Dropdown(
                        choices=[
                            "Critical Dispatch",
                            "High Priority",
                            "Moderate Sweep",
                            "Low / Routine",
                        ],
                        value="Moderate Sweep",
                        label="Approved / Overridden Cleanup Priority",
                    )
                    decision_dd_t1 = gr.Dropdown(
                        choices=[
                            "Confirmed Model Detections & Dispatched",
                            "Adjusted Priority After Visual Review",
                            "Flagged False Positive / Re-scheduled",
                        ],
                        value="Confirmed Model Detections & Dispatched",
                        label="Inspector Triage Decision",
                    )
                with gr.Row():
                    notes_t1 = gr.Textbox(
                        value="Small roadside plastic items verified via SAHI tile zoom gallery.",
                        label="Inspector Field Notes",
                        scale=3,
                    )
                    signoff_btn_t1 = gr.Button(
                        "Confirm HITL Sign-Off & Export Printable Certificate",
                        variant="secondary",
                        scale=1,
                    )
                with gr.Row():
                    signoff_status_t1 = gr.Markdown()
                    cert_file_t1 = gr.File(
                        label="Download Printable HTML Inspection Certificate"
                    )
                audit_trail_df_t1 = gr.Dataframe(
                    label="Persistent Session HITL Verified Audit Trail"
                )

                sample_dd_t1.change(
                    fn=load_sample_to_pil, inputs=[sample_dd_t1], outputs=[inp_img_t1]
                )
                run_btn_t1.click(
                    fn=run_tab1_inspection,
                    inputs=[
                        inp_img_t1,
                        sample_dd_t1,
                        backend_dd_t1,
                        mode_dd_t1,
                        overlay_dd_t1,
                        dip_chk_t1,
                        cal_chk_t1,
                        conf_ovr_chk_t1,
                        conf_slider_t1,
                    ],
                    outputs=[
                        dip_img_t1,
                        out_img_t1,
                        kpi_cards_t1,
                        crop_gallery_t1,
                        det_table_t1,
                        summary_md_t1,
                        json_code_t1,
                        prio_dd_t1,
                    ],
                )
                signoff_btn_t1.click(
                    fn=submit_inspector_signoff,
                    inputs=[
                        insp_name_t1,
                        zone_dd_t1,
                        prio_dd_t1,
                        decision_dd_t1,
                        notes_t1,
                        json_code_t1,
                    ],
                    outputs=[signoff_status_t1, cert_file_t1, audit_trail_df_t1],
                )

            # ==================== TAB 2: BASELINE VS PROPOSED ====================
            with gr.TabItem("2. Baseline vs. Proposed D3 Side-by-Side Comparison"):
                gr.Markdown(
                    "### Controlled Visual & Quantitative Proof of Improvement (Questions 1, 2 & 3c)\n"
                    "Runs the **Baseline Single-Pass Implementation** and the **Proposed D3 Implementation** "
                    "side-by-side on the exact same input image to demonstrate small-object recovery (`<32x32 px`), "
                    "minority-class sensitivity, and adaptive latency."
                )
                with gr.Row():
                    sample_dd_t2 = gr.Dropdown(
                        choices=SAMPLE_CHOICES,
                        value=default_sample,
                        label="Select Benchmark Sample (TACO or pLitterStreet)",
                        scale=2,
                    )
                    backend_dd_t2 = gr.Dropdown(
                        choices=[
                            "ONNX Runtime (best.onnx — Web Deployment Format)",
                            "PyTorch FP32 (best.pt — Native Checkpoint)",
                        ],
                        value="ONNX Runtime (best.onnx — Web Deployment Format)",
                        label="Execution Backend",
                        scale=2,
                    )
                    variant_dd_t2 = gr.Dropdown(
                        choices=[
                            "Proposed D3 Full (2x2 Conditional SAHI — Max Recall)",
                            "Proposed D3 Fast (Smart 2-Tile Ground SAHI — 40% Faster on CPU)",
                        ],
                        value="Proposed D3 Full (2x2 Conditional SAHI — Max Recall)",
                        label="Proposed D3 Variant",
                        scale=2,
                    )
                    cmp_btn_t2 = gr.Button(
                        "Run Side-by-Side Baseline vs. Proposed D3 Comparison",
                        variant="primary",
                        scale=1,
                    )

                inp_img_t2 = gr.Image(
                    type="pil",
                    value=load_sample_to_pil(default_sample) if default_sample else None,
                    label="Input Street Scene (Upload or Choose Sample Above)",
                    height=240,
                )

                with gr.Row():
                    base_out_img_t2 = gr.Image(
                        label="Baseline Output (Single-Pass, No DIP, Uniform Conf=0.25)"
                    )
                    prop_out_img_t2 = gr.Image(
                        label="Proposed D3 Output (LAB-CLAHE + Conditional SAHI + F1 Calibrated)"
                    )

                cmp_table_md_t2 = gr.Markdown(
                    label="Quantitative Comparison & Improvement Breakdown"
                )

                sample_dd_t2.change(
                    fn=load_sample_to_pil, inputs=[sample_dd_t2], outputs=[inp_img_t2]
                )
                cmp_btn_t2.click(
                    fn=run_tab2_comparison,
                    inputs=[inp_img_t2, sample_dd_t2, backend_dd_t2, variant_dd_t2],
                    outputs=[base_out_img_t2, prop_out_img_t2, cmp_table_md_t2],
                )

            # ==================== TAB 3: BATCH AUDIT & GIS MAP ====================
            with gr.TabItem("3. Batch Municipal Audit & KCCA GIS Hotspot Map"):
                gr.Markdown(
                    "### Municipal Sanitation Decision-Support & GIS Export Workflow (Question 1d & Demo C.5)\n"
                    "Processes all bundled **TACO** and **pLitterStreet** validation scenes, computes the "
                    "**Pixel-Level Mask-Area Urban Litter Severity Index (ULSI, 0–100)**, flags uncertain/hazardous "
                    "scenes for **Human-in-the-Loop (HITL) Inspector Review**, and exports an interactive "
                    "**Folium GIS Map**, `CSV`, and `GeoJSON`."
                )
                with gr.Row():
                    backend_dd_t3 = gr.Dropdown(
                        choices=[
                            "ONNX Runtime (best.onnx — Web Deployment Format)",
                            "PyTorch FP32 (best.pt — Native Checkpoint)",
                        ],
                        value="ONNX Runtime (best.onnx — Web Deployment Format)",
                        label="Model Format",
                    )
                    mode_dd_t3 = gr.Dropdown(
                        choices=[
                            "Proposed D3 (LAB-CLAHE + Uncertainty-Gated 2x2 SAHI + Calibrated)",
                            "Proposed D3 Fast (LAB-CLAHE + Smart 2-Tile Ground SAHI + Calibrated)",
                            "Baseline (Single-Pass, No DIP, Uniform Conf=0.25)",
                        ],
                        value="Proposed D3 Fast (LAB-CLAHE + Smart 2-Tile Ground SAHI + Calibrated)",
                        label="Audit Pipeline Mode",
                    )
                    batch_btn_t3 = gr.Button(
                        "Execute Batch Street Audit & Generate GIS Map", variant="primary"
                    )

                gis_html_t3 = gr.HTML(label="Interactive KCCA Litter Severity Hotspot Map")
                batch_df_t3 = gr.Dataframe(label="Municipal Street Inspection Log")
                with gr.Row():
                    csv_file_t3 = gr.File(label="Download CSV Inspection Log")
                    geojson_file_t3 = gr.File(label="Download GeoJSON GIS Layer")

                batch_btn_t3.click(
                    fn=run_tab3_batch_audit,
                    inputs=[backend_dd_t3, mode_dd_t3],
                    outputs=[gis_html_t3, batch_df_t3, csv_file_t3, geojson_file_t3],
                )

            # ==================== TAB 4: PATROL VIDEO / SEQUENCE SCREENER ====================
            with gr.TabItem("4. Street Patrol Video / Dashcam Sequence Screener"):
                gr.Markdown(
                    "### Frame-by-Frame Patrol Video & Street Sweep Timeline Screener (Question 3g)\n"
                    "Upload a dashcam/smartphone video clip (or leave empty to run a **Simulated KCCA Patrol Sweep** "
                    "across the bundled street sequence). Automatically tracks the **ULSI Severity Curve** across stops "
                    "and isolates the **Peak Litter Hotspot Frame** for immediate crew dispatch."
                )
                with gr.Row():
                    vid_inp_t4 = gr.Video(
                        label="Optional Patrol Video Upload (Leave empty to run Simulated KCCA Street Sweep)"
                    )
                    with gr.Column():
                        backend_dd_t4 = gr.Dropdown(
                            choices=[
                                "ONNX Runtime (best.onnx — Web Deployment Format)",
                                "PyTorch FP32 (best.pt — Native Checkpoint)",
                            ],
                            value="PyTorch FP32 (best.pt — Native Checkpoint)",
                            label="Execution Backend",
                        )
                        mode_dd_t4 = gr.Dropdown(
                            choices=[
                                "Proposed D3 Fast (LAB-CLAHE + Smart 2-Tile Ground SAHI + Calibrated)",
                                "Proposed D3 (LAB-CLAHE + Uncertainty-Gated 2x2 SAHI + Calibrated)",
                                "Baseline (Single-Pass, No DIP, Uniform Conf=0.25)",
                            ],
                            value="Proposed D3 Fast (LAB-CLAHE + Smart 2-Tile Ground SAHI + Calibrated)",
                            label="Patrol Screening Mode",
                        )
                        stride_slider_t4 = gr.Slider(
                            1, 60, value=15, step=1, label="Video Frame Sampling Stride (Every Nth Frame)"
                        )
                        max_frames_slider_t4 = gr.Slider(
                            3, 12, value=6, step=1, label="Max Keyframes to Screen"
                        )
                        patrol_btn_t4 = gr.Button(
                            "Run Patrol Sequence Screening & Plot ULSI Timeline",
                            variant="primary",
                        )

                patrol_md_t4 = gr.Markdown()
                with gr.Row():
                    timeline_img_t4 = gr.Image(
                        label="Patrol ULSI Severity Timeline & Dispatch Thresholds"
                    )
                    peak_img_t4 = gr.Image(
                        label="Automatically Isolated Peak Litter Hotspot Frame"
                    )
                patrol_df_t4 = gr.Dataframe(label="Patrol Stop-by-Stop Screening Log")

                patrol_btn_t4.click(
                    fn=run_tab4_patrol_screener,
                    inputs=[
                        vid_inp_t4,
                        backend_dd_t4,
                        mode_dd_t4,
                        stride_slider_t4,
                        max_frames_slider_t4,
                    ],
                    outputs=[timeline_img_t4, peak_img_t4, patrol_df_t4, patrol_md_t4],
                )

    return demo


if __name__ == "__main__":
    app = build_gradio_app()
    port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    app.launch(server_name="127.0.0.1", server_port=port)
