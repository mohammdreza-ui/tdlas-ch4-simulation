# Design of Experiments (DOE): Ground Truth Data Generation for Laser Spectroscopy

## 1. Objective & Scope
The objective is to establish a rigorous, reproducible experimental protocol for acquiring training, validation, and test datasets from a physical Tunable Diode Laser Absorption Spectroscopy (TDLAS) sensor detecting methane ($CH_4$) at $3313\text{ nm}$ ($3018\text{ cm}^{-1}$), avoiding statistical data leakage and accounting for instrument drift.

---

## 2. Experimental Factors & Parameter Space

| Factor | Type | Operational Range / Levels | Control Method |
| :--- | :--- | :--- | :--- |
| **Methane Concentration ($x$)** | Primary | $0, 5, 10, 25, 50\text{ ppm}$ | Mass flow controllers with certified $N_2$ balance gas |
| **Gas Cell Temperature ($T$)** | Environmental | $280\text{ K}$ to $315\text{ K}$ ($\pm 0.1\text{ K}$) | Thermoelectric Peltier cooler/heater on gas cell |
| **Gas Cell Pressure ($P$)** | Environmental | $0.85\text{ atm}$ to $1.15\text{ atm}$ | Barometric controller with precision vacuum pump |
| **Laser Diode Temperature** | Instrument | Fixed at $20.00^\circ\text{C}$ ($\pm 0.005^\circ\text{C}$) | High-precision TEC driver |
| **Optical Alignment / Vibration**| Disturbance | Baseline vs. Induced $10\text{ Hz}$ vibration | Shaker table & kinematic mirror mounts |

---

## 3. Randomization, Replication & Drift Control

### 3.1 Sample Size Justification
Sample size per concentration level is computed using Allan-Werle variance and Cramér-Rao Lower Bound (CRB):
$$N_{\text{reps}} \ge \left( \frac{z_{0.975} \cdot \sigma_{\text{CRB}}}{\text{Margin of Error}} \right)^2$$
For $\sigma_{\text{CRB}} \approx 0.15\text{ ppm}$ and a target margin of $\pm 0.02\text{ ppm}$, $N_{\text{reps}} \ge 220$ scans per level.

### 3.2 Drift Control via Zero-Air Interleaving
To prevent slow thermal random-walk drift from biasing the training distribution:
1. Every experimental block begins and ends with certified **Zero Air** ($< 0.1\text{ ppm}$ total hydrocarbons).
2. The zero-air scan establishes the unperturbed baseline polynomial $P_L(i)$ and optical etalon fringe phase $\phi$.
3. Run order across concentration levels is fully randomized within each block using Latin Hypercube Sampling (LHS).

---

## 4. Leakage-Free Dataset Partitioning
Data leakage represents the primary failure mode in machine learning applied to physical sensors.
- **Rule:** Scans must be partitioned at the **Sensor Device / Acquisition Session level**, never randomly across individual time scans.
- **Train / Validation / Test Split:**
  - **Train:** Sensor Units 1 through 6 (multiple days, varying temperatures $285 - 305\text{ K}$).
  - **Validation:** Sensor Unit 7 (unseen hardware instance, standard conditions).
  - **Test (OOD Scenarios):** Sensor Units 8 through 10, explicitly subjected to out-of-distribution pressure ($1.2\text{ atm}$), elevated $1/f$ noise, and misaligned etalons.

---

## 5. Sim-to-Real Transfer & Integrating Sphere Calibration
In diffuse integrating sphere gas cells (e.g. Hodgkinson et al., Cranfield University), the effective optical pathlength $L_{\text{eff}}$ deviates from the geometric diameter:
$$L_{\text{eff}} \approx \frac{4V}{A(1 - \rho)}$$
where $V$ is sphere volume, $A$ is internal surface area, and $\rho \approx 0.98$ is the wall reflectance at $3.3\ \mu\text{m}$.
1. **Calibrated Baseline:** $L_{\text{eff}}$ is calibrated prior to deployment using a certified $50\text{ ppm}$ span gas cell.
2. **Transfer Strategy:** The forward simulation pipeline models $L_{\text{eff}}$ as a Gaussian prior $\mathcal{N}(70.0\text{ cm}, 2.0\text{ cm})$ across device instances, allowing neural networks to learn invariant gas absorption features rather than memorizing pathlength geometry.
