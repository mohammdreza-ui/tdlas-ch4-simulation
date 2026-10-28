# TDLAS Interactive Visual Dashboard & Scientific Comparison Cockpit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a zero-dependency, local interactive Web Dashboard and publication figure exporter enabling a research supervisor to visually inspect, interact with, and compare TDLAS methane spectroscopy results across physics (Levenberg-Marquardt with analytical Jacobians), machine learning (Deep Ensembles), Allan-Werle stability, and harmonic lock-in demodulation.

**Architecture:** A lightweight Python standard library HTTP/REST backend (`src/tdlas/dashboard_backend.py`) serving a modern responsive dark-mode web cockpit (`src/tdlas/static/`) with real-time sliders, 4 analytical tabs, and 4 one-click scientific comparison presets. Accompanied by a publication-grade 300 DPI Matplotlib figure exporter and standalone HTML report generator (`scripts/export_figures.py`).

**Tech Stack:** Python 3.12, NumPy, SciPy, PyTorch (Deep Ensemble), HTML5, CSS3 (Glassmorphic dark design), Vanilla JS, Matplotlib (for 300 DPI publication figures).

**Spec:** Established through interactive `/grill-me` session.

---

## Global Constraints

- **100% Offline Execution:** Zero external cloud or remote CDN requirements; all backend logic calls the local `tdlas` package.
- **Zero Extra Dependencies:** Backend must run using Python's standard `http.server` without forcing external framework installations (no required Flask/FastAPI/Streamlit), ensuring immediate compatibility in `.venv`.
- **Physical Naming & Typings:** Strict adherence to physical units (`nu_cm1`, `T_K`, `P_atm`, `L_cm`, `x_ppm`).
- **Complete Test Coverage:** Every module and endpoint must have automated pytest coverage before commit.
- **Zero Placeholders:** Full, working implementations only.

---

### Task 1: Dashboard REST Engine (`src/tdlas/dashboard_backend.py`)

**Files:**
- Create: `src/tdlas/dashboard_backend.py`
- Test: `tests/test_dashboard_backend.py`

**Interfaces:**
- Consumes:
  - `tdlas.instrument.simulate_sensor_scan(x_ppm, T_K, P_atm, dev, lines_data)`
  - `tdlas.fit.levenberg_marquardt_fit(V_measured, nu_axis_cm1, t_norm, T_K, P_atm, L_cm, lines_data)`
  - `tdlas.uncertainty.compute_covariance_and_crb(J, sigma_noise)`
  - `tdlas.allan.compute_allan_deviation(y, fs_hz)`
  - `hardware.lockin.SoftwareLockIn`
  - `tdlas.ml.model.DeepEnsembleTDLAS`
- Produces:
  - `run_server(host, port, blocking)`
  - `create_dashboard_handler(static_dir)`
  - REST endpoints: `/api/simulate`, `/api/fit`, `/api/compare`, `/api/allan`, `/api/lockin`, `/api/presets`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dashboard_backend.py
import json
import urllib.request
import threading
import time
import pytest
from tdlas.dashboard_backend import start_test_server

@pytest.fixture(scope="module")
def server_url():
    server, port = start_test_server()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.1)
    yield f"http://127.0.0.1:{port}"
    server.shutdown()

def test_api_presets(server_url):
    req = urllib.request.Request(f"{server_url}/api/presets")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert "nominal" in data
        assert "detection_limit" in data
        assert "etalon_interference" in data

def test_api_simulate_and_fit(server_url):
    payload = json.dumps({
        "x_ppm": 25.0,
        "T_K": 296.0,
        "P_atm": 1.0,
        "noise_std_v": 0.001,
        "etalon_amplitude": 0.0
    }).encode("utf-8")
    req = urllib.request.Request(f"{server_url}/api/fit", data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert "fitted_x_ppm" in data
        assert abs(data["fitted_x_ppm"] - 25.0) < 2.0
        assert "crb_std_x" in data
        assert "residuals" in data
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_dashboard_backend.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'tdlas.dashboard_backend'`

- [ ] **Step 3: Implement dashboard backend handler and server**

Create `src/tdlas/dashboard_backend.py` with:
- `DashboardRequestHandler(http.server.SimpleHTTPRequestHandler)` routing GET/POST requests.
- JSON response helper with CORS headers.
- Endpoints processing simulation, Levenberg-Marquardt fitting, CRB estimation, Deep Ensemble inference, and Allan variance calculation.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_dashboard_backend.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/tdlas/dashboard_backend.py tests/test_dashboard_backend.py
git commit -m "feat(dashboard): add REST backend engine for interactive visualization (Task 1)"
```

---

### Task 2: Interactive Web Cockpit UI (`src/tdlas/static/`)

**Files:**
- Create: `src/tdlas/static/index.html`
- Create: `src/tdlas/static/style.css`
- Create: `src/tdlas/static/app.js`
- Test: `tests/test_dashboard_ui.py`

**Interfaces:**
- Consumes: REST API from `src/tdlas/dashboard_backend.py`
- Produces: Web interface with:
  - 4 one-click scientific presets (`Nominal 50 ppm`, `Detection Limit 1 ppm`, `Etalon Shift`, `High T/P 350K`).
  - Interactive sliders: Concentration, Temperature, Pressure, Etalon Amplitude, White Noise.
  - Tab 1: Spectroscopy & Voigt Fit (Transmission, Baseline, Absorbance, Fit, Residuals).
  - Tab 2: Side-by-Side Comparison: Physics (LM) vs Machine Learning (Deep Ensemble).
  - Tab 3: Metrology & Allan Variance ($\sigma_A(\tau)$ curve, $\tau_{\text{opt}}$, CRB vs Empirical error).
  - Tab 4: Harmonic Lock-In Demodulation ($1f, 2f$ WMS, phase and amplitude).
  - Single-click "Export Publication Figures" action trigger.

- [ ] **Step 1: Write failing test**

```python
# tests/test_dashboard_ui.py
from pathlib import Path

def test_static_files_exist_and_contain_elements():
    static_dir = Path("src/tdlas/static")
    html_file = static_dir / "index.html"
    css_file = static_dir / "style.css"
    js_file = static_dir / "app.js"
    
    assert html_file.exists()
    assert css_file.exists()
    assert js_file.exists()
    
    html_text = html_file.read_text(encoding="utf-8")
    assert "id=\"tab-spectroscopy\"" in html_text
    assert "id=\"tab-comparison\"" in html_text
    assert "id=\"tab-allan\"" in html_text
    assert "id=\"tab-lockin\"" in html_text
    assert "id=\"preset-nominal\"" in html_text
    assert "id=\"slider-concentration\"" in html_text
    assert "id=\"btn-export-figures\"" in html_text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_dashboard_ui.py -v`  
Expected: FAIL (files missing)

- [ ] **Step 3: Implement `index.html`, `style.css`, and `app.js`**

Implement modern UI featuring:
- SVG-based lightweight responsive charts rendering absorption, Voigt profile, residuals, Allan variance, and lock-in harmonics.
- Asynchronous API fetching from `/api/fit`, `/api/compare`, `/api/allan`, `/api/lockin`.
- Dynamic DOM updating for tables, badges, error bars, and latency counters.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_dashboard_ui.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/tdlas/static/ tests/test_dashboard_ui.py
git commit -m "feat(dashboard): implement modern interactive web cockpit and controls (Task 2)"
```

---

### Task 3: Standalone Figures Exporter & Static HTML Report (`scripts/export_figures.py`)

**Files:**
- Create: `scripts/export_figures.py`
- Test: `tests/test_export_figures.py`

**Interfaces:**
- Produces:
  - `docs/figures/fig1_spectroscopy_voigt_fit.png` & `.svg`
  - `docs/figures/fig2_physics_vs_ml_comparison.png` & `.svg`
  - `docs/figures/fig3_allan_variance_stability.png` & `.svg`
  - `docs/figures/fig4_wms_lockin_harmonics.png` & `.svg`
  - `docs/interactive_report.html`

- [ ] **Step 1: Write failing test**

```python
# tests/test_export_figures.py
from pathlib import Path
import subprocess

def test_export_figures_script():
    cmd = [".venv/bin/python3", "scripts/export_figures.py", "--fast"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert "Exporting publication figures" in result.stdout or "complete" in result.stdout.lower()
    
    figures_dir = Path("docs/figures")
    assert (figures_dir / "fig1_spectroscopy_voigt_fit.png").exists()
    assert (figures_dir / "fig2_physics_vs_ml_comparison.png").exists()
    assert (figures_dir / "fig3_allan_variance_stability.png").exists()
    assert (figures_dir / "fig4_wms_lockin_harmonics.png").exists()
    assert Path("docs/interactive_report.html").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_export_figures.py -v`  
Expected: FAIL with `FileNotFoundError: No such file 'scripts/export_figures.py'`

- [ ] **Step 3: Implement `scripts/export_figures.py`**

- Uses Matplotlib with clean academic styling (LaTeX fonts, tight bounding box, 300 DPI raster and vector SVG).
- Generates all 4 multi-panel figures covering spectroscopy, physics vs ML comparison, Allan variance with $-0.5$ slope, and $1f/2f$ harmonic demodulation.
- Generates `docs/interactive_report.html` with embedded base64/SVG figures and summary tables.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_export_figures.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/export_figures.py tests/test_export_figures.py docs/figures/ docs/interactive_report.html
git commit -m "feat(export): add 300 DPI publication figure exporter and interactive report (Task 3)"
```

---

### Task 4: Master CLI Launch Script (`scripts/run_dashboard.py`)

**Files:**
- Create: `scripts/run_dashboard.py`
- Test: `tests/test_run_dashboard.py`

**Interfaces:**
- Produces: CLI interface `python scripts/run_dashboard.py [--port PORT] [--no-browser]`

- [ ] **Step 1: Write failing test**

```python
# tests/test_run_dashboard.py
import subprocess
import time
import urllib.request
import signal

def test_run_dashboard_cli():
    # Start dashboard on specific port with --no-browser
    proc = subprocess.Popen(
        [".venv/bin/python3", "scripts/run_dashboard.py", "--port", "8899", "--no-browser"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    try:
        time.sleep(1.0)
        # Check that root index is served
        with urllib.request.urlopen("http://127.0.0.1:8899/") as resp:
            assert resp.status == 200
            content = resp.read().decode("utf-8")
            assert "TDLAS" in content
    finally:
        proc.terminate()
        proc.wait(timeout=2.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_run_dashboard.py -v`  
Expected: FAIL (file missing)

- [ ] **Step 3: Implement `scripts/run_dashboard.py`**

Add argument parsing (`--port`, `--no-browser`, `--host`), automatic port discovery if requested port is occupied, browser launch via `webbrowser.open`, and graceful `KeyboardInterrupt` handling.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_run_dashboard.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/run_dashboard.py tests/test_run_dashboard.py
git commit -m "feat(cli): add one-command dashboard launcher script (Task 4)"
```

---

### Task 5: End-to-End Verification & Documentation Update

**Files:**
- Modify: `README.md`
- Modify: `docs/technical_report.md`
- Modify: `tests/test_docs_integrity.py`

- [ ] **Step 1: Write failing test update**

Update `tests/test_docs_integrity.py` to assert that `README.md` contains the instructions for running the dashboard (`python scripts/run_dashboard.py`) and exporting publication figures (`python scripts/export_figures.py`).

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_docs_integrity.py -v`  
Expected: FAIL

- [ ] **Step 3: Update documentation**

Add dedicated section in `README.md` and `docs/technical_report.md` detailing:
- Launching the visual dashboard: `python scripts/run_dashboard.py`
- Generating publication figures: `python scripts/export_figures.py`
- Description of the 4 interactive panels and preset scenarios.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_docs_integrity.py -v`  
Expected: PASS

- [ ] **Step 5: Run full project test suite**

Run: `.venv/bin/pytest -v`  
Expected: All 45+ tests PASS across all modules.

- [ ] **Step 6: Commit**

```bash
git add README.md docs/technical_report.md tests/test_docs_integrity.py
git commit -m "docs: document interactive visual dashboard and publication figure exporter (Task 5)"
```

---

## Verification Plan

### Automated Tests
1. Unit and API tests:
   ```bash
   .venv/bin/pytest tests/test_dashboard_backend.py tests/test_dashboard_ui.py tests/test_export_figures.py tests/test_run_dashboard.py -v
   ```
2. Full suite regression check:
   ```bash
   .venv/bin/pytest -v
   ```
3. Exporter execution:
   ```bash
   .venv/bin/python3 scripts/export_figures.py
   ```

### Manual Verification
1. Start the visual dashboard:
   ```bash
   .venv/bin/python3 scripts/run_dashboard.py --port 8050
   ```
2. Open `http://localhost:8050` in browser.
3. Test the 4 preset buttons (`Nominal`, `Detection Limit`, `Etalon Shift`, `High T/P`).
4. Switch through all 4 tabs and verify real-time chart updates.
5. Click "Export Figures" and check that 300 DPI images are generated in `docs/figures/`.
