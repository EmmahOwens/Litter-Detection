"""Baseline Single-Pass Litter Detection Web Interface (Deliverable 2 Baseline Reference).

Placed in `baseline/` per Capstone D3 Section 3.D requirements:
"Provide a Git repository containing the baseline and proposed implementations
in clearly identifiable locations."
"""
from collections import Counter
import json
import os
from pathlib import Path
import cv2
import gradio as gr
import numpy as np
from PIL import Image
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT_DIR / "models" / "best.pt"
MODEL_PATH = os.getenv("MODEL_PATH", str(DEFAULT_MODEL))
model = YOLO(MODEL_PATH if os.path.exists(MODEL_PATH) else "yolo26s-seg.pt")


def inspect_baseline(pil_img, conf_thresh: float = 0.25):
    """Run un-intervened single-pass baseline inference without SAHI tiling or DIP."""
    if pil_img is None:
        return None, "Upload an image to run the baseline model.", "{}"
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    res = model.predict(img_bgr, imgsz=640, conf=conf_thresh, verbose=False)[0]
    annotated = Image.fromarray(res.plot()[:, :, ::-1])

    counts = Counter()
    if res.boxes is not None and len(res.boxes) > 0:
        cids = res.boxes.cls.cpu().numpy().astype(int)
        for cid in cids:
            counts[res.names[int(cid)]] += 1

    summary = (
        f"Mode: Baseline Single-Pass (640px, Uniform Conf={conf_thresh:.2f})\n"
        f"Total Litter Items Detected: {sum(counts.values())}\n"
        + "\n".join(f"- {m}: {n} item(s)" for m, n in sorted(counts.items()))
    )
    export_json = json.dumps(
        {"mode": "baseline_single_pass", "conf_thresh": conf_thresh, "counts": dict(counts)},
        indent=2,
    )
    return annotated, summary, export_json


with gr.Blocks(title="Baseline Urban Litter Detector (Single-Pass D2 Reference)") as demo:
    gr.Markdown("# Baseline Urban Litter Detection (Single-Pass Reference)")
    with gr.Row():
        inp = gr.Image(type="pil", label="Input Street Photograph")
        out_img = gr.Image(label="Baseline Single-Pass Output")
    conf_slider = gr.Slider(0.10, 0.85, value=0.25, step=0.05, label="Uniform Confidence Threshold")
    btn = gr.Button("Run Baseline Single-Pass", variant="secondary")
    with gr.Row():
        out_txt = gr.Textbox(label="Baseline Detection Summary", lines=5)
        out_json = gr.Code(label="JSON Output", language="json")
    btn.click(inspect_baseline, [inp, conf_slider], [out_img, out_txt, out_json])

if __name__ == "__main__":
    demo.launch()
