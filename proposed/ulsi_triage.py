"""Urban Litter Severity Index (ULSI), Human-in-the-Loop Triage & KCCA GIS Exporter.

Implements the customer-oriented decision support layer defined in Deliverables 1–3:
- 5-class harmonized material taxonomy & hazard weights.
- Validation F1-optimal per-class confidence thresholds.
- Pixel-level mask-area ULSI severity scoring (0–100) and dispatch priority tiers.
- Human-in-the-Loop (HITL) inspector triage flagging & screening-time savings proxy.
- Interactive Folium GIS hotspot map and GeoJSON/CSV export for KCCA workflows.
"""
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple
import json
import folium
import pandas as pd

MATERIALS: List[str] = ["Plastic", "Metal", "Glass", "Paper", "Other"]

MATERIAL_HAZARD_WEIGHTS: Dict[str, float] = {
    "Glass": 1.4,
    "Plastic": 1.3,
    "Metal": 1.2,
    "Other": 1.1,
    "Paper": 0.8,
}

# Validation F1-Optimal Per-Class Confidence Thresholds (from D2/D3 Validation Calibration)
CALIBRATED_CLASS_THRESHOLDS: Dict[str, float] = {
    "Plastic": 0.31,
    "Metal": 0.34,
    "Glass": 0.26,
    "Paper": 0.19,
    "Other": 0.20,
}

# Simulated Kampala / KCCA division coordinates for GIS decision-support demonstration
KCCA_ZONES: List[Tuple[str, float, float]] = [
    ("Central Division (Nakasero - Demo GIS)", 0.3163, 32.5822),
    ("Kawempe Division (Bwaise - Demo GIS)", 0.3541, 32.5634),
    ("Makindye Division (Katwe - Demo GIS)", 0.2934, 32.5789),
    ("Nakawa Division (Bugolobi - Demo GIS)", 0.3198, 32.6152),
    ("Rubaga Division (Ndeeba - Demo GIS)", 0.3015, 32.5539),
]


def compute_ulsi_and_triage(
    detections: List[Dict[str, Any]],
    infer_sec: float = 0.15,
    scene_idx: int = 0,
    image_name: str = "uploaded_image.jpg",
) -> Dict[str, Any]:
    """Compute ULSI severity score (0-100), dispatch priority, and HITL triage status."""
    mat_counts: Counter = Counter()
    mat_area_pct: Counter = Counter()
    uncertain_dets = 0
    small_obj_count = 0

    for det in detections:
        mat = det["material"]
        cf = float(det["confidence"])
        area_pct = float(det.get("mask_area_pct", 0.0))
        size_tier = det.get("coco_size_tier", "Medium")

        mat_counts[mat] += 1
        mat_area_pct[mat] += area_pct

        if size_tier == "Small (<32x32 px)":
            small_obj_count += 1
        if cf < 0.38 or mat == "Glass":
            uncertain_dets += 1

    total_items = sum(mat_counts.values())
    total_area_pct = sum(mat_area_pct.values())

    weighted_count_score = sum(
        cnt * MATERIAL_HAZARD_WEIGHTS.get(m, 1.0) * 7.5 for m, cnt in mat_counts.items()
    )
    weighted_area_score = sum(
        area * MATERIAL_HAZARD_WEIGHTS.get(m, 1.0) * 4.5 for m, area in mat_area_pct.items()
    )
    ulsi_score = round(min(100.0, weighted_count_score + weighted_area_score), 1)

    if ulsi_score >= 70.0:
        priority = "Critical Dispatch"
        marker_color = "red"
    elif ulsi_score >= 45.0:
        priority = "High Priority"
        marker_color = "orange"
    elif ulsi_score >= 20.0:
        priority = "Moderate Sweep"
        marker_color = "blue"
    else:
        priority = "Low / Routine"
        marker_color = "green"

    needs_human_review = (uncertain_dets > 0) or (total_items >= 6)
    triage_status = (
        "Flagged for Inspector Review"
        if needs_human_review
        else "Auto-Verified (High Confidence)"
    )

    div_name, base_lat, base_lon = KCCA_ZONES[scene_idx % len(KCCA_ZONES)]
    lat = round(base_lat + ((scene_idx * 17) % 100 - 50) * 0.00025, 6)
    lon = round(base_lon + ((scene_idx * 31) % 100 - 50) * 0.00025, 6)

    # Human vs. Assisted Screening Time Proxy (Proposal Section 10)
    est_manual_sec = round(12.0 + 3.5 * total_items, 2)
    est_assisted_sec = round(infer_sec + (5.0 if needs_human_review else 1.5), 2)
    time_saved_pct = round(
        max(0.0, ((est_manual_sec - est_assisted_sec) / (est_manual_sec + 1e-6)) * 100.0),
        1,
    )

    return {
        "image": image_name,
        "kcca_division": div_name,
        "latitude": lat,
        "longitude": lon,
        "total_items": total_items,
        "small_objects_count": small_obj_count,
        "total_mask_coverage_pct": round(total_area_pct, 3),
        "ulsi_severity_score": ulsi_score,
        "cleanup_priority": priority,
        "marker_color": marker_color,
        "triage_status": triage_status,
        "uncertain_items_count": uncertain_dets,
        "counts_by_material": {m: mat_counts[m] for m in MATERIALS if mat_counts[m] > 0},
        "mask_area_pct_by_material": {
            m: round(mat_area_pct[m], 3) for m in MATERIALS if mat_counts[m] > 0
        },
        "plastic_count": mat_counts["Plastic"],
        "metal_count": mat_counts["Metal"],
        "glass_count": mat_counts["Glass"],
        "paper_count": mat_counts["Paper"],
        "other_count": mat_counts["Other"],
        "inference_sec": round(infer_sec, 3),
        "manual_screening_proxy_sec": est_manual_sec,
        "assisted_screening_proxy_sec": est_assisted_sec,
        "screening_time_saved_pct": time_saved_pct,
        "detections": detections,
    }


def build_kcca_gis_artifacts(
    records: List[Dict[str, Any]],
    output_dir: Path,
) -> Tuple[str, Path, Path]:
    """Generate an interactive Folium HTML hotspot map, GeoJSON file, and CSV log."""
    output_dir.mkdir(parents=True, exist_ok=True)
    kcca_map = folium.Map(location=[0.3163, 32.5822], zoom_start=12, tiles="OpenStreetMap")
    geojson_features = []
    flat_rows = []

    for rec in records:
        flat_rec = {
            k: v
            for k, v in rec.items()
            if k not in ("detections", "counts_by_material", "mask_area_pct_by_material")
        }
        flat_rows.append(flat_rec)

        popup_html = (
            f"<b>Image:</b> {rec['image']}<br>"
            f"<b>Zone:</b> {rec['kcca_division']}<br>"
            f"<b>ULSI Severity:</b> {rec['ulsi_severity_score']}/100 ({rec['cleanup_priority']})<br>"
            f"<b>Total Items:</b> {rec['total_items']} (Small: {rec['small_objects_count']}) | "
            f"<b>Mask Area:</b> {rec['total_mask_coverage_pct']}%<br>"
            f"<b>Materials:</b> Plastic={rec['plastic_count']}, Metal={rec['metal_count']}, "
            f"Glass={rec['glass_count']}, Paper={rec['paper_count']}, Other={rec['other_count']}<br>"
            f"<b>HITL Triage:</b> {rec['triage_status']}"
        )
        folium.CircleMarker(
            location=[rec["latitude"], rec["longitude"]],
            radius=max(6, min(16, int(rec["ulsi_severity_score"] / 6) + 5)),
            color=rec["marker_color"],
            fill=True,
            fill_opacity=0.78,
            popup=folium.Popup(popup_html, max_width=340),
        ).add_to(kcca_map)

        geojson_features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [rec["longitude"], rec["latitude"]],
                },
                "properties": flat_rec,
            }
        )

    csv_path = output_dir / "municipal_inspection_log.csv"
    geojson_path = output_dir / "kcca_litter_hotspots.geojson"
    html_path = output_dir / "kcca_hotspot_map.html"

    pd.DataFrame(flat_rows).to_csv(csv_path, index=False)
    geojson_path.write_text(
        json.dumps({"type": "FeatureCollection", "features": geojson_features}, indent=2)
    )
    kcca_map.save(str(html_path))

    return kcca_map._repr_html_(), csv_path, geojson_path
