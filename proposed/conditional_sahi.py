"""Uncertainty-Gated Conditional SAHI Tiling, Visual X-Ray Overlays & Multi-Backend Engine (D3).

Supports:
1. Web-Ready ONNX Runtime (`models/best.onnx`) and PyTorch (`models/best.pt`) execution.
2. Four selectable pipeline modes for controlled ablation and live PC demonstration:
   - `baseline`: Single-pass inference at 640px, no DIP preprocessing, uniform conf=0.25.
   - `proposed_conditional`: CIE LAB CLAHE + Adaptive Gamma -> Fast Full-Frame Pass ->
     Uncertainty/Clutter Gate -> Conditional 2x2 Overlapping SAHI Tiles (up to 5 passes) ->
     Class-Aware NMS -> Validation F1-Optimal Per-Class Confidence Calibration.
   - `proposed_smart_2tile`: Ground-focused 2-tile conditional SAHI (1 full pass + 2 lower
     ground-plane 65% tiles = 3 passes max), cutting CPU latency by ~40% on street scenes.
   - `unconditional_sahi`: Always executes 1 Full-Frame Pass + 4 SAHI Tiles (5 passes total).
3. Four diagnostic visualization modes:
   - `standard`: Instance masks + bounding boxes + material confidence tags.
   - `sahi_grid_xray`: Visualizes the overlapping SAHI tile boundaries and color-codes
     Full-Frame vs. Tile-Recovered small objects.
   - `mask_spotlight`: Dims the background to grayscale so segmented litter masks pop in vivid color.
   - `dip_edge_heatmap`: Visualizes the Sobel local edge-contrast boost from LAB-CLAHE.
4. Automatic zoomed-in instance crop gallery (`extract_instance_crops`) for inspector review.
"""
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from ultralytics import YOLO

from proposed.dip_preprocessing import dip_enhance_street_image
from proposed.ulsi_triage import (
    CALIBRATED_CLASS_THRESHOLDS,
    MATERIALS,
    compute_ulsi_and_triage,
)

# Distinct BGR colors for the 5 harmonized material classes
MATERIAL_COLORS_BGR: Dict[str, Tuple[int, int, int]] = {
    "Plastic": (235, 115, 30),   # Vivid Blue-Cyan
    "Metal": (30, 130, 245),     # Vivid Orange
    "Glass": (180, 60, 210),     # Vivid Magenta-Purple
    "Paper": (50, 195, 60),      # Vivid Green
    "Other": (40, 205, 235),     # Vivid Gold-Yellow
}


class LitterSegmentationEngine:
    """Multi-backend (ONNX / PyTorch) instance segmentation engine with Conditional SAHI."""

    def __init__(self, models_dir: Optional[Path] = None) -> None:
        root = Path(__file__).resolve().parent.parent
        self.models_dir = models_dir or (root / "models")
        self._loaded_models: Dict[str, YOLO] = {}

    def get_model(self, backend: str = "onnx") -> Tuple[YOLO, str]:
        """Load and cache either the exported ONNX model (`best.onnx`) or PyTorch (`best.pt`)."""
        backend_key = "onnx" if "onnx" in backend.lower() else "pytorch"
        if backend_key not in self._loaded_models:
            onnx_path = self.models_dir / "best.onnx"
            pt_path = self.models_dir / "best.pt"
            if backend_key == "onnx" and onnx_path.exists():
                self._loaded_models[backend_key] = YOLO(str(onnx_path), task="segment")
            elif pt_path.exists():
                self._loaded_models[backend_key] = YOLO(str(pt_path), task="segment")
            else:
                raise FileNotFoundError(
                    f"Neither {onnx_path} nor {pt_path} found in {self.models_dir}"
                )
        model_path_str = (
            str(self.models_dir / "best.onnx")
            if backend_key == "onnx"
            else str(self.models_dir / "best.pt")
        )
        return self._loaded_models[backend_key], model_path_str


def compute_box_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute Intersection-over-Union (IoU) between two [x1, y1, x2, y2] boxes."""
    xA = max(float(box1[0]), float(box2[0]))
    yA = max(float(box1[1]), float(box2[1]))
    xB = min(float(box1[2]), float(box2[2]))
    yB = min(float(box1[3]), float(box2[3]))
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    area1 = max(0.0, float(box1[2] - box1[0])) * max(0.0, float(box1[3] - box1[1]))
    area2 = max(0.0, float(box2[2] - box2[0])) * max(0.0, float(box2[3] - box2[1]))
    return inter / (area1 + area2 - inter + 1e-8)


def nms_candidates_per_class(
    candidates: List[Dict[str, Any]], iou_thresh: float = 0.50
) -> List[Dict[str, Any]]:
    """Class-aware Non-Maximum Suppression preserving polygons and source pass metadata."""
    if not candidates:
        return []
    kept: List[Dict[str, Any]] = []
    by_class: Dict[int, List[Dict[str, Any]]] = {}
    for cand in candidates:
        by_class.setdefault(int(cand["cls_id"]), []).append(cand)

    for _, cls_cands in by_class.items():
        cls_sorted = sorted(cls_cands, key=lambda d: float(d["score"]), reverse=True)
        while cls_sorted:
            best = cls_sorted.pop(0)
            kept.append(best)
            cls_sorted = [
                other
                for other in cls_sorted
                if compute_box_iou(best["box"], other["box"]) <= iou_thresh
            ]
    return kept


def _extract_candidates_from_result(
    res: Any,
    offset_x: int = 0,
    offset_y: int = 0,
    score_weight: float = 1.0,
    source_tag: str = "full_frame",
) -> List[Dict[str, Any]]:
    """Extract bounding boxes, classes, scores, and shifted mask polygons from a YOLO result."""
    out: List[Dict[str, Any]] = []
    if res.boxes is None or len(res.boxes) == 0:
        return out

    boxes_xyxy = res.boxes.xyxy.cpu().numpy()
    scores = res.boxes.conf.cpu().numpy()
    cls_ids = res.boxes.cls.cpu().numpy().astype(int)
    polys_xy = res.masks.xy if (res.masks is not None and hasattr(res.masks, "xy")) else None

    for idx, (box, sc, cid) in enumerate(zip(boxes_xyxy, scores, cls_ids)):
        shifted_box = np.array(
            [
                float(box[0]) + offset_x,
                float(box[1]) + offset_y,
                float(box[2]) + offset_x,
                float(box[3]) + offset_y,
            ],
            dtype=np.float32,
        )
        poly = None
        if polys_xy is not None and idx < len(polys_xy):
            raw_poly = polys_xy[idx]
            if raw_poly is not None and len(raw_poly) >= 3:
                poly = raw_poly.astype(np.float32).copy()
                poly[:, 0] += float(offset_x)
                poly[:, 1] += float(offset_y)

        out.append(
            {
                "box": shifted_box,
                "score": float(sc) * score_weight,
                "cls_id": int(cid),
                "material": MATERIALS[int(cid)] if 0 <= int(cid) < len(MATERIALS) else "Other",
                "polygon": poly,
                "source": source_tag,
            }
        )
    return out


def should_trigger_conditional_sahi(
    full_candidates: List[Dict[str, Any]],
    img_w: int,
    img_h: int,
    imgsz: int = 640,
) -> Tuple[bool, str]:
    """Evaluate the Uncertainty & Small-Object Gate after the initial Full-Frame pass."""
    valid_cands = [c for c in full_candidates if float(c["score"]) >= 0.17]
    if not valid_cands:
        return False, "Skipped (Clean scene — 0 initial proposals >= 0.17)"

    scale = float(imgsz) / float(max(img_w, img_h))
    uncertain_count = 0
    small_count = 0
    rare_count = 0

    for cand in valid_cands:
        sc = float(cand["score"])
        b = cand["box"]
        bw, bh = max(0.0, float(b[2] - b[0])), max(0.0, float(b[3] - b[1]))
        scaled_area = (bw * bh) * (scale ** 2)

        if 0.17 <= sc < 0.35:
            uncertain_count += 1
        if scaled_area < (32.0 ** 2):
            small_count += 1
        if cand["material"] in ("Glass", "Other") and sc < 0.42:
            rare_count += 1

    if uncertain_count > 0:
        return True, f"Triggered ({uncertain_count} borderline proposal(s) in [0.17, 0.35))"
    if len(valid_cands) >= 3:
        return True, f"Triggered (Cluttered scene with {len(valid_cands)} initial items)"
    if small_count >= 2:
        return True, f"Triggered ({small_count} small-object candidates < 32x32 px)"
    if rare_count > 0:
        return True, f"Triggered ({rare_count} tentative Glass/Other proposal(s))"

    return False, f"Skipped ({len(valid_cands)} high-confidence item(s) — single pass sufficient)"


def render_segmented_annotations(
    img_bgr: np.ndarray,
    detections: List[Dict[str, Any]],
    mode_label: str = "Proposed D3",
    sahi_status: str = "Auto",
    latency_ms: float = 0.0,
    overlay_style: str = "standard",
    tile_windows: Optional[List[Tuple[int, int, int, int]]] = None,
    raw_bgr_for_delta: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Draw instance masks, boxes, material labels, diagnostic overlays, and top telemetry bar."""
    H, W = img_bgr.shape[:2]

    if overlay_style == "dip_edge_heatmap" and raw_bgr_for_delta is not None:
        l_raw = cv2.cvtColor(raw_bgr_for_delta, cv2.COLOR_BGR2LAB)[:, :, 0]
        l_enh = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)[:, :, 0]
        g_raw = cv2.magnitude(
            cv2.Sobel(l_raw, cv2.CV_32F, 1, 0, ksize=3),
            cv2.Sobel(l_raw, cv2.CV_32F, 0, 1, ksize=3),
        )
        g_enh = cv2.magnitude(
            cv2.Sobel(l_enh, cv2.CV_32F, 1, 0, ksize=3),
            cv2.Sobel(l_enh, cv2.CV_32F, 0, 1, ksize=3),
        )
        diff = np.clip((g_enh - g_raw) * 3.2, 0, 255).astype(np.uint8)
        heat = cv2.applyColorMap(diff, cv2.COLORMAP_TURBO)
        canvas = cv2.addWeighted(img_bgr, 0.55, heat, 0.45, 0)
    elif overlay_style == "mask_spotlight":
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        canvas = (cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) * 0.45).astype(np.uint8)
    else:
        canvas = img_bgr.copy()

    mask_overlay = canvas.copy()
    font_scale = max(0.48, min(0.85, max(H, W) / 1400.0))
    thickness = max(2, int(max(H, W) / 600))

    # Draw SAHI Tile Windows if in X-Ray mode
    if overlay_style == "sahi_grid_xray" and tile_windows:
        tile_colors = [(255, 220, 0), (0, 220, 255), (255, 100, 255), (100, 255, 100)]
        for t_idx, (ox, oy, tw, th) in enumerate(tile_windows):
            tc = tile_colors[t_idx % len(tile_colors)]
            cv2.rectangle(canvas, (ox + 2, oy + 2), (ox + tw - 2, oy + th - 2), tc, 2)
            cv2.putText(
                canvas,
                f"SAHI Tile #{t_idx + 1} ({tw}x{th})",
                (ox + 10, oy + 26 + (t_idx % 2) * 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                max(0.45, font_scale * 0.8),
                tc,
                2,
                cv2.LINE_AA,
            )
        mask_overlay = canvas.copy()

    for det in detections:
        mat = det["material"]
        if overlay_style == "sahi_grid_xray":
            # Color-code by detection origin: Neon Green = SAHI Tile Recovered, Cyan = Full-Frame
            color = (40, 245, 80) if det.get("source") == "sahi_tile" else (245, 190, 40)
        else:
            color = MATERIAL_COLORS_BGR.get(mat, (0, 200, 255))
        poly = det.get("polygon")
        x1, y1, x2, y2 = [int(round(v)) for v in det["box_xyxy"]]

        if poly is not None and len(poly) >= 3:
            pts = np.clip(poly, [0, 0], [W - 1, H - 1]).astype(np.int32)
            cv2.fillPoly(mask_overlay, [pts], color)
            cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=thickness)
        else:
            cv2.rectangle(mask_overlay, (x1, y1), (x2, y2), color, -1)

    alpha_fill = 0.58 if overlay_style == "mask_spotlight" else 0.38
    canvas = cv2.addWeighted(mask_overlay, alpha_fill, canvas, 1.0 - alpha_fill, 0)

    # Draw bounding boxes and readable tags
    for idx, det in enumerate(detections, start=1):
        mat = det["material"]
        cf = det["confidence"]
        src_tag = " [Tile]" if det.get("source") == "sahi_tile" else ""
        if overlay_style == "sahi_grid_xray":
            color = (30, 200, 60) if det.get("source") == "sahi_tile" else (210, 140, 20)
        else:
            color = MATERIAL_COLORS_BGR.get(mat, (0, 200, 255))
        x1, y1, x2, y2 = [int(round(v)) for v in det["box_xyxy"]]

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        label = f"#{idx} {mat} {cf:.2f}{src_tag}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        ty1 = max(0, y1 - th - baseline - 6)
        ty2 = ty1 + th + baseline + 6
        tx2 = min(W - 1, x1 + tw + 8)
        cv2.rectangle(canvas, (x1, ty1), (tx2, ty2), color, -1)
        cv2.putText(
            canvas,
            label,
            (x1 + 4, ty2 - baseline - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            max(1, thickness - 1),
            cv2.LINE_AA,
        )

    # Non-destructive top status header bar (padded above image so it never covers detections)
    banner_h = max(36, int(H * 0.048))
    header = np.full((banner_h, W, 3), (22, 27, 34), dtype=np.uint8)
    hud_text = (
        f"{mode_label} | Detections: {len(detections)} | "
        f"SAHI Gate: {sahi_status} | {latency_ms:.1f} ms"
    )
    cv2.putText(
        header,
        hud_text,
        (12, int(banner_h * 0.68)),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.45, font_scale * 0.85),
        (240, 246, 252),
        max(1, thickness - 1),
        cv2.LINE_AA,
    )
    return np.vstack([header, canvas])


def extract_instance_crops(
    work_bgr: np.ndarray, detections: List[Dict[str, Any]], pad_px: int = 28
) -> List[Tuple[np.ndarray, str]]:
    """Extract zoomed-in RGB thumbnails for each detected litter instance for gallery review."""
    H, W = work_bgr.shape[:2]
    gallery_items: List[Tuple[np.ndarray, str]] = []

    for idx, det in enumerate(detections, start=1):
        x1, y1, x2, y2 = [int(round(v)) for v in det["box_xyxy"]]
        cx1, cy1 = max(0, x1 - pad_px), max(0, y1 - pad_px)
        cx2, cy2 = min(W, x2 + pad_px), min(H, y2 + pad_px)
        if cx2 <= cx1 or cy2 <= cy1:
            continue

        crop = work_bgr[cy1:cy2, cx1:cx2].copy()
        mat = det["material"]
        color = MATERIAL_COLORS_BGR.get(mat, (0, 200, 255))

        poly = det.get("polygon")
        if poly is not None and len(poly) >= 3:
            pts = (poly - np.array([cx1, cy1], dtype=np.float32)).astype(np.int32)
            overlay = crop.copy()
            cv2.fillPoly(overlay, [pts], color)
            crop = cv2.addWeighted(overlay, 0.35, crop, 0.65, 0)
            cv2.polylines(crop, [pts], isClosed=True, color=color, thickness=2)

        cv2.rectangle(crop, (x1 - cx1, y1 - cy1), (x2 - cx1, y2 - cy1), color, 2)
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        src_badge = "SAHI Tile" if det.get("source") == "sahi_tile" else "Full Frame"
        caption = (
            f"#{idx} {mat} (conf={det['confidence']:.2f}) | "
            f"{det['coco_size_tier'].split(' ')[0]} | {src_badge}"
        )
        gallery_items.append((crop_rgb, caption))

    return gallery_items


def run_litter_pipeline(
    engine: LitterSegmentationEngine,
    img_bgr: np.ndarray,
    backend: str = "onnx",
    pipeline_mode: str = "proposed_conditional",
    apply_dip_override: Optional[bool] = None,
    base_conf_override: Optional[float] = None,
    use_calibrated_thresholds: bool = True,
    overlay_style: str = "standard",
    imgsz: int = 640,
    scene_idx: int = 0,
    image_name: str = "sample.jpg",
) -> Dict[str, Any]:
    """Execute the complete litter segmentation pipeline and return images, crops & telemetry."""
    model, model_path = engine.get_model(backend=backend)
    orig_H, orig_W = img_bgr.shape[:2]

    is_baseline = pipeline_mode.lower().startswith("baseline")
    do_dip = (not is_baseline) if apply_dip_override is None else apply_dip_override

    # --- Stage 1: Classical DIP Preprocessing & Working Resolution Normalization ---
    t0_dip = time.perf_counter()
    max_work_dim = 1280  # 2x 640px preserves full resolution for both full-frame & 2x2 SAHI tiles
    if max(orig_H, orig_W) > max_work_dim:
        scale_w = max_work_dim / float(max(orig_H, orig_W))
        W = int(round(orig_W * scale_w))
        H = int(round(orig_H * scale_w))
        input_bgr = cv2.resize(img_bgr, (W, H), interpolation=cv2.INTER_AREA)
    else:
        input_bgr = img_bgr
        H, W = orig_H, orig_W

    if do_dip:
        work_bgr, dip_diag = dip_enhance_street_image(input_bgr)
    else:
        work_bgr = input_bgr.copy()
        l_raw = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2LAB)[:, :, 0]
        dip_diag = {
            "raw_mean_l": round(float(np.mean(l_raw)), 2),
            "enh_mean_l": round(float(np.mean(l_raw)), 2),
            "raw_rms_contrast": round(float(np.std(l_raw)), 2),
            "enh_rms_contrast": round(float(np.std(l_raw)), 2),
            "contrast_gain_pct": 0.0,
            "applied_gamma": 1.0,
        }
    dip_ms = (time.perf_counter() - t0_dip) * 1000.0

    # --- Stage 2 & 3: Feature Extraction & Model Inference ---
    t0_inf = time.perf_counter()
    passes_executed = 1
    initial_conf = (
        (base_conf_override if base_conf_override is not None else 0.25)
        if is_baseline
        else 0.16
    )

    res_full = model.predict(work_bgr, imgsz=imgsz, conf=initial_conf, verbose=False)[0]
    candidates = _extract_candidates_from_result(
        res_full, offset_x=0, offset_y=0, score_weight=1.0, source_tag="full_frame"
    )

    sahi_triggered = False
    sahi_reason = "Disabled (Baseline Single-Pass)"
    tile_windows: List[Tuple[int, int, int, int]] = []

    if pipeline_mode == "unconditional_sahi":
        sahi_triggered = True
        sahi_reason = "Forced On (Unconditional 2x2 SAHI — 5 Passes)"
    elif pipeline_mode in ("proposed_conditional", "proposed_smart_2tile"):
        sahi_triggered, sahi_reason = should_trigger_conditional_sahi(
            candidates, img_w=W, img_h=H, imgsz=imgsz
        )

    if sahi_triggered:
        tw, th = max(64, int(W * 0.60)), max(64, int(H * 0.60))
        if pipeline_mode == "proposed_smart_2tile":
            # Ground-Plane 2-Tile Mode: focuses on bottom-left and bottom-right road surface
            gh = max(64, int(H * 0.68))
            tile_offsets = [(0, H - gh), (W - tw, H - gh)]
            for ox, oy in tile_offsets:
                tile_windows.append((ox, oy, tw, gh))
                tile_bgr = work_bgr[oy : oy + gh, ox : ox + tw]
                res_tile = model.predict(tile_bgr, imgsz=imgsz, conf=0.16, verbose=False)[0]
                passes_executed += 1
                candidates.extend(
                    _extract_candidates_from_result(
                        res_tile,
                        offset_x=ox,
                        offset_y=oy,
                        score_weight=0.95,
                        source_tag="sahi_tile",
                    )
                )
        else:
            tile_offsets = [(0, 0), (W - tw, 0), (0, H - th), (W - tw, H - th)]
            for ox, oy in tile_offsets:
                tile_windows.append((ox, oy, tw, th))
                tile_bgr = work_bgr[oy : oy + th, ox : ox + tw]
                res_tile = model.predict(tile_bgr, imgsz=imgsz, conf=0.16, verbose=False)[0]
                passes_executed += 1
                candidates.extend(
                    _extract_candidates_from_result(
                        res_tile,
                        offset_x=ox,
                        offset_y=oy,
                        score_weight=0.95,
                        source_tag="sahi_tile",
                    )
                )

    infer_ms = (time.perf_counter() - t0_inf) * 1000.0

    # --- Stage 4: Post-Processing (Class-Aware NMS, Mask Reconstruction & Calibration) ---
    t0_post = time.perf_counter()
    merged_cands = nms_candidates_per_class(candidates, iou_thresh=0.50)

    scale_to_640 = float(imgsz) / float(max(W, H))
    final_detections: List[Dict[str, Any]] = []

    grid_scale = min(1.0, 320.0 / float(max(W, H)))
    gW, gH = max(16, int(round(W * grid_scale))), max(16, int(round(H * grid_scale)))

    for cand in merged_cands:
        mat = cand["material"]
        sc = float(cand["score"])

        if is_baseline or not use_calibrated_thresholds:
            thresh = base_conf_override if base_conf_override is not None else 0.25
        else:
            cal_th = CALIBRATED_CLASS_THRESHOLDS.get(mat, 0.22)
            thresh = (
                max(0.12, min(cal_th, base_conf_override))
                if base_conf_override is not None
                else cal_th
            )

        if sc < thresh:
            continue

        box = cand["box"]
        cy = 0.5 * (float(box[1]) + float(box[3]))
        if cand["source"] == "sahi_tile" and cy < 0.22 * H and sc < 0.42:
            continue

        bw = max(0.0, float(box[2] - box[0]))
        bh = max(0.0, float(box[3] - box[1]))
        scaled_area = (bw * bh) * (scale_to_640 ** 2)
        if scaled_area < (32.0 ** 2):
            size_tier = "Small (<32x32 px)"
        elif scaled_area < (96.0 ** 2):
            size_tier = "Medium (32x32 to 96x96 px)"
        else:
            size_tier = "Large (>=96x96 px)"

        poly = cand.get("polygon")
        if poly is not None and len(poly) >= 3:
            m_grid = np.zeros((gH, gW), dtype=np.uint8)
            pts_grid = np.clip(poly * grid_scale, [0, 0], [gW - 1, gH - 1]).astype(np.int32)
            cv2.fillPoly(m_grid, [pts_grid], 1)
            mask_area_pct = (float(m_grid.sum()) / float(gW * gH)) * 100.0
        else:
            mask_area_pct = ((bw * bh) / float(W * H)) * 100.0 * 0.65

        final_detections.append(
            {
                "material": mat,
                "cls_id": cand["cls_id"],
                "confidence": round(sc, 3),
                "threshold_applied": round(float(thresh), 2),
                "mask_area_pct": round(mask_area_pct, 3),
                "coco_size_tier": size_tier,
                "source": cand["source"],
                "box_xyxy": [round(float(v), 1) for v in box],
                "polygon": poly,
            }
        )

    post_ms = (time.perf_counter() - t0_post) * 1000.0
    total_ms = dip_ms + infer_ms + post_ms

    # --- Stage 5: Output Generation (ULSI Severity, HITL Triage, Crops & Visualization) ---
    clean_dets_for_json = [
        {k: v for k, v in d.items() if k != "polygon"} for d in final_detections
    ]
    ulsi_report = compute_ulsi_and_triage(
        clean_dets_for_json,
        infer_sec=total_ms / 1000.0,
        scene_idx=scene_idx,
        image_name=image_name,
    )

    mode_display = {
        "baseline": "Baseline (Single-Pass, No DIP)",
        "proposed_conditional": "Proposed D3 (DIP + Conditional 2x2 SAHI)",
        "proposed_smart_2tile": "Proposed D3 Fast (DIP + Smart 2-Tile Ground SAHI)",
        "unconditional_sahi": "Unconditional 2x2 SAHI (5 Passes)",
    }.get(pipeline_mode, pipeline_mode)

    annotated_bgr = render_segmented_annotations(
        work_bgr,
        final_detections,
        mode_label=mode_display,
        sahi_status=f"Triggered ({passes_executed} passes)" if sahi_triggered else "Fast Pass (1 pass)",
        latency_ms=total_ms,
        overlay_style=overlay_style,
        tile_windows=tile_windows,
        raw_bgr_for_delta=input_bgr,
    )

    instance_crops = extract_instance_crops(work_bgr, final_detections)

    telemetry = {
        "backend": "ONNX Runtime (best.onnx)" if "onnx" in backend.lower() else "PyTorch (best.pt)",
        "model_file": Path(model_path).name,
        "pipeline_mode": mode_display,
        "overlay_style": overlay_style,
        "image_resolution": f"{W}x{H}",
        "dip_enabled": do_dip,
        "dip_diagnostics": dip_diag,
        "sahi_triggered": sahi_triggered,
        "sahi_gate_reason": sahi_reason,
        "passes_executed": passes_executed,
        "preprocess_ms": round(dip_ms, 2),
        "inference_ms": round(infer_ms, 2),
        "postprocess_ms": round(post_ms, 2),
        "total_latency_ms": round(total_ms, 2),
        "throughput_fps": round(1000.0 / max(total_ms, 1e-3), 2),
        "tile_recovered_count": sum(1 for d in final_detections if d["source"] == "sahi_tile"),
    }

    return {
        "raw_rgb": cv2.cvtColor(input_bgr, cv2.COLOR_BGR2RGB),
        "preprocessed_rgb": cv2.cvtColor(work_bgr, cv2.COLOR_BGR2RGB),
        "annotated_rgb": cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB),
        "instance_crops": instance_crops,
        "detections_table": clean_dets_for_json,
        "telemetry": telemetry,
        "ulsi_report": ulsi_report,
    }
