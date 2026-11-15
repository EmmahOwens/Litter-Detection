# How to Run the Urban Litter Detection Model & PC App (Deliverable 3)

This guide explains how to set up and run the **Deliverable 3 Urban Litter Instance-Segmentation & Municipal Severity System** on a standard CPU-only laptop or desktop.

---

## 1. Prerequisites & Required Files

- **Python:** 3.10 – 3.12
- **Model weights** (already included in `models/`):
  - `models/best.onnx` (recommended for CPU deployment via ONNX Runtime)
  - `models/best.pt` (PyTorch checkpoint)
  - Optional: `models/calibrated_thresholds.json` (auto-loaded by `proposed/ulsi_triage.py` if present)

---

## 2. First-Time Installation

Clone the repository (or open a terminal inside the extracted project folder):
```bash
git clone https://github.com/EmmahOwens/Litter-Detection.git
cd Litter-Detection
```

### Linux / macOS
```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install CPU-only PyTorch first (avoids downloading 2+ GB CUDA packages)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 3. Install project dependencies
pip install -r requirements.txt
```

### Windows (PowerShell)
```powershell
cd litter-detection-capstone

# 1. Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install CPU-only PyTorch first
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 3. Install project dependencies
pip install -r requirements.txt
```

---

## 3. Launch the Deliverable 3 Web Application

### Quick Launch (Linux / macOS)
If `.venv` is already set up, run the one-command launcher:
```bash
./run_pc_app.sh
```

### Manual Launch (Linux / macOS / Windows)
With your virtual environment activated:
```bash
python proposed/app_d3_deployed.py
```

Once the server starts, open your browser and go to:
**`http://127.0.0.1:7860`**

*(If port `7860` is already in use, set `GRADIO_SERVER_PORT=7861` before launching.)*

---

## 4. Using the Application (4 Tabs)

1. **Tab 1 — `1. Interactive D3 Pipeline Workstation (3-Stage + X-Ray + HITL)`**
   - Pick a built-in TACO/pLitterStreet sample or upload your own street photo.
   - Select **`ONNX Runtime (best.onnx)`** (or `PyTorch (best.pt)`) and **`proposed_conditional`** mode.
   - Click **Run** to inspect the 3-stage output (Raw $\rightarrow$ CIE LAB CLAHE $\rightarrow$ Segmented Masks), switch between the 4 diagnostic overlays (`Standard`, `SAHI Grid X-Ray`, `Mask Spotlight`, `DIP Edge Heatmap`), review zoomed item crops, and sign off the Human-in-the-Loop (HITL) priority / download the HTML inspection certificate.

2. **Tab 2 — `2. Baseline vs. Proposed D3 Side-by-Side Comparison`**
   - Run the same street image through **Baseline (Single-Pass, No DIP)** and **Proposed D3 (LAB-CLAHE + Conditional SAHI)** side by side to compare item counts, small-object recovery ($<32\text{ px}$), and ULSI severity scores.

3. **Tab 3 — `3. Batch Municipal Audit & KCCA GIS Hotspot Map`**
   - Batch-process multiple street images, generate the interactive Folium hotspot map (`outputs/kcca_hotspot_map.html`), and export GeoJSON/CSV inspection logs.

4. **Tab 4 — `4. Street Patrol Video / Dashcam Sequence Screener`**
   - Upload a short street patrol video to sample frames and screen litter severity across a route.

---

## 5. Optional Commands

- **Run the standalone Deliverable 2 Baseline app:**
  ```bash
  python baseline/app_baseline.py
  ```
- **Re-run the automated PC deployment benchmarks & regenerate report figures (`outputs/`):**
  ```bash
  python benchmarks/run_pc_benchmarks.py
  ```

---

## 6. Troubleshooting

- **`best.onnx` or `best.pt` not found:** Ensure both files are placed inside `litter-detection-capstone/models/`.
- **Slow first inference:** The first image loads the ONNX/PyTorch runtime into memory; subsequent images run faster.
- **Matplotlib cache warning on Linux:** `run_pc_app.sh` automatically sets `MPLCONFIGDIR` to `/tmp/matplotlib-cache`. If running manually, you can run `export MPLCONFIGDIR=/tmp/matplotlib-cache`.
