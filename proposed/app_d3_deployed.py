"""Capstone Deliverable 3 — Deployed PC Web Application for Urban Litter Inspection.

Provides a 3-Tab Gradio Web Interface designed for PC/Laptop deployment (Question 3e-3h & Demo 3.C):
- Tab 1: Interactive 3-Stage Pipeline Visualizer (Raw -> LAB-CLAHE Preprocessed -> Segmented Output)
         with ONNX Runtime (`best.onnx`) / PyTorch (`best.pt`) selection and Conditional 2x2 SAHI.
- Tab 2: Live Baseline vs. Proposed D3 Side-by-Side Comparison with Quantitative Improvement Delta.
- Tab 3: Batch Municipal Street Audit & Interactive KCCA Folium GIS Hotspot Exporter.
"""
import json
import os
import platform
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import gradio as gr
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
    build_kcca_gis_artifacts,
)

ENGINE = LitterSegmentationEngine(ROOT_DIR / "models")
OUTPUTS_DIR = ROOT_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


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
        f"**Models:** `best.onnx` (Web ONNX Runtime, 39.8 MB) & `best.pt` (PyTorch FP32, 22.3 MB)"
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
    if "unconditional" in low:
        return "unconditional_sahi"
    return "proposed_conditional"


def run_tab1_inspection(
    pil_img: Optional[Image.Image],
    sample_label: str,
    backend_choice: str,
    pipeline_choice: str,
    apply_dip: bool,
    use_calibrated: bool,
    conf_override_enabled: bool,
    conf_slider_val: float,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str, str]:
    """Execute Tab 1 inspection and return Preprocessed RGB, Annotated RGB, Summary, and JSON."""
    img_name = "uploaded_street_photo.jpg"
    if pil_img is None and sample_label in SAMPLE_MAP:
        pil_img = Image.open(SAMPLE_MAP[sample_label]).convert("RGB")
        img_name = SAMPLE_MAP[sample_label].name
    elif sample_label in SAMPLE_MAP:
        img_name = SAMPLE_MAP[sample_label].name

    if pil_img is None:
        return None, None, "Please upload a street photograph or select a sample image.", "{}"

    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    mode_code = _mode_code_from_ui(pipeline_choice)
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
        scene_idx=0,
        image_name=img_name,
    )

    tel = result["telemetry"]
    rep = result["ulsi_report"]
    dip_d = tel["dip_diagnostics"]

    mat_lines = []
    for m, cnt in sorted(rep["counts_by_material"].items()):
        area_m = rep["mask_area_pct_by_material"].get(m, 0.0)
        cal_th = CALIBRATED_CLASS_THRESHOLDS.get(m, 0.25)
        mat_lines.append(
            f"  - **{m}**: `{cnt}` item(s) | Mask Surface Area: `{area_m:.3f}%` (Calibrated Thresh: `{cal_th:.2f}`)"
        )
    if not mat_lines:
        mat_lines.append("  - *No litter items detected above threshold (Clean Street Scene).*")

    md_summary = f"""### Municipal Inspection & Pipeline Execution Summary
- **ULSI Severity Score:** `{rep['ulsi_severity_score']} / 100` (**Priority:** `{rep['cleanup_priority']}`)
- **Human-in-the-Loop Triage:** `{rep['triage_status']}` (`{rep['uncertain_items_count']}` uncertain/hazard item(s))
- **Total Detected Litter:** `{rep['total_items']}` item(s) | **Small Objects (<32x32 px):** `{rep['small_objects_count']}` | **Tile-Recovered:** `{tel['tile_recovered_count']}` | **Total Mask Coverage:** `{rep['total_mask_coverage_pct']:.3f}%`
- **Screening Time Proxy:** Manual `{rep['manual_screening_proxy_sec']:.2f}s` vs. Assisted `{rep['assisted_screening_proxy_sec']:.2f}s` (**`{rep['screening_time_saved_pct']:.1f}%` time saved**)

#### Pipeline Stage Telemetry (`{tel['backend']}`)
- **Stage 1 (Classical DIP Pre-Enhancement):** `{tel['preprocess_ms']:.2f} ms` | Enabled: `{tel['dip_enabled']}` | Gamma: `{dip_d['applied_gamma']}` | Luminance: `{dip_d['raw_mean_l']} -> {dip_d['enh_mean_l']}` | Local RMS Contrast Gain: **`+{dip_d['contrast_gain_pct']:.2f}%`**
- **Stage 2 & 3 (Feature Extraction & Inference):** `{tel['inference_ms']:.2f} ms` | Passes Executed: `{tel['passes_executed']}` | **SAHI Gate:** `{tel['sahi_gate_reason']}`
- **Stage 4 (Class-Aware NMS & Threshold Calibration):** `{tel['postprocess_ms']:.2f} ms` | Total Latency: **`{tel['total_latency_ms']:.2f} ms`** (`{tel['throughput_fps']:.2f} FPS`)

#### Detected Material Breakdown
{chr(10).join(mat_lines)}
"""

    export_payload = {
        "telemetry": tel,
        "municipal_decision_support": rep,
    }
    return (
        result["preprocessed_rgb"],
        result["annotated_rgb"],
        md_summary,
        json.dumps(export_payload, indent=2),
    )


def run_tab2_comparison(
    pil_img: Optional[Image.Image],
    sample_label: str,
    backend_choice: str,
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
        pipeline_mode="proposed_conditional",
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

| Metric / Component | Baseline System (D2 Single-Pass) | Proposed D3 System (DIP + Conditional SAHI + Calibrated) | Measured Improvement / Delta |
| :--- | :--- | :--- | :--- |
| **Classical DIP Preprocessing** | Disabled (Raw BGR) | **CIE LAB CLAHE + Adaptive Gamma (`+{p_tel['dip_diagnostics']['contrast_gain_pct']:.1f}%` contrast)** | Enhances shadowed/translucent litter edges |
| **Inference Strategy** | Single-Pass 640px (`1 pass`) | **Uncertainty-Gated 2x2 SAHI (`{p_tel['passes_executed']} pass(es)`)** | `{p_tel['sahi_gate_reason']}` |
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
    for idx, (label, img_path) in enumerate(SAMPLE_MAP.items()):
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
                "SAHI Gate": "Triggered" if r.get("sahi_triggered") else "Fast 1-Pass",
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


def build_gradio_app() -> gr.Blocks:
    """Construct the multi-tab Gradio interface for Capstone Deliverable 3."""
    default_sample = SAMPLE_CHOICES[0] if SAMPLE_CHOICES else None

    with gr.Blocks(
        title="Urban Litter Detection & Municipal Severity System — Capstone D3 PC Deployment"
    ) as demo:
        gr.Markdown(
            "# Urban Litter Instance Segmentation & Municipal Decision-Support System (Capstone D3)\n"
            "**Proposed Improvement:** Classical DIP (`CIE LAB CLAHE + Adaptive Gamma`) + "
            "`YOLO26s-seg` (`STAL + C2PSA + Train-Only Copy-Paste`) + "
            "**Uncertainty-Gated Conditional 2x2 SAHI Tiling** + **Per-Class F1 Threshold Calibration**\n\n"
            + get_pc_environment_banner()
        )

        with gr.Tabs():
            # ==================== TAB 1: PIPELINE INSPECTOR ====================
            with gr.TabItem("1. Interactive D3 Pipeline Inspector (3-Stage View)"):
                gr.Markdown(
                    "### End-to-End Processing Pipeline Demonstration (Questions 3a–3g & Demo C.1–C.5)\n"
                    "Select a benchmark sample from **TACO** or **pLitterStreet** (or upload your own photo) "
                    "to inspect **Stage 1 (Raw Input)** $\\rightarrow$ **Stage 2 (CIE LAB CLAHE Preprocessed)** "
                    "$\\rightarrow$ **Stage 3 (Segmented Material Masks, ULSI Score & HITL Triage)**."
                )
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
                            "Unconditional 2x2 SAHI (Always 5 Passes — D2 Exp4)",
                            "Baseline (Single-Pass, No DIP, Uniform Conf=0.25)",
                        ],
                        value="Proposed D3 (LAB-CLAHE + Uncertainty-Gated 2x2 SAHI + Calibrated)",
                        label="Pipeline Operating Mode",
                        scale=3,
                    )

                with gr.Row():
                    dip_chk_t1 = gr.Checkbox(
                        value=True,
                        label="Enable Classical DIP Pre-Enhancement (CIE LAB CLAHE + Shadow Gamma)",
                    )
                    cal_chk_t1 = gr.Checkbox(
                        value=True,
                        label="Use Validation F1-Optimal Per-Class Thresholds (Plastic=0.31, Metal=0.34, Glass=0.26, Paper=0.19, Other=0.20)",
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
                        label="Stage 3: Instance Segmentation Masks, Material Classes & SAHI Recovery"
                    )

                with gr.Row():
                    summary_md_t1 = gr.Markdown(
                        label="Pipeline Telemetry & ULSI Decision-Support Summary"
                    )
                    json_code_t1 = gr.Code(
                        label="Machine-Readable GIS / JSON Output", language="json"
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
                        dip_chk_t1,
                        cal_chk_t1,
                        conf_ovr_chk_t1,
                        conf_slider_t1,
                    ],
                    outputs=[dip_img_t1, out_img_t1, summary_md_t1, json_code_t1],
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
                    cmp_btn_t2 = gr.Button(
                        "Run Side-by-Side Baseline vs. Proposed D3 Comparison",
                        variant="primary",
                        scale=1,
                    )

                inp_img_t2 = gr.Image(
                    type="pil",
                    value=load_sample_to_pil(default_sample) if default_sample else None,
                    label="Input Street Scene (Upload or Choose Sample Above)",
                    height=260,
                )

                with gr.Row():
                    base_out_img_t2 = gr.Image(
                        label="Baseline Output (Single-Pass, No DIP, Uniform Conf=0.25)"
                    )
                    prop_out_img_t2 = gr.Image(
                        label="Proposed D3 Output (LAB-CLAHE + Conditional 2x2 SAHI + F1 Calibrated)"
                    )

                cmp_table_md_t2 = gr.Markdown(
                    label="Quantitative Comparison & Improvement Breakdown"
                )

                sample_dd_t2.change(
                    fn=load_sample_to_pil, inputs=[sample_dd_t2], outputs=[inp_img_t2]
                )
                cmp_btn_t2.click(
                    fn=run_tab2_comparison,
                    inputs=[inp_img_t2, sample_dd_t2, backend_dd_t2],
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
                            "Baseline (Single-Pass, No DIP, Uniform Conf=0.25)",
                        ],
                        value="Proposed D3 (LAB-CLAHE + Uncertainty-Gated 2x2 SAHI + Calibrated)",
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

    return demo


if __name__ == "__main__":
    app = build_gradio_app()
    port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    app.launch(server_name="127.0.0.1", server_port=port)
