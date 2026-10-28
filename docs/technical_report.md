# Technical Report: Methane Spectroscopy Signal Processing Pipeline

**Context:** Inspired by the Cranfield University TDLAS Methane Detection Architecture (Prof. Jane Hodgkinson Group, 2023)  
**Notice:** All optical detector signals in this work are generated via mathematical forward modeling; no physical laser laboratory access is claimed.

---

> The numerical results in sections 1, 3 and 4 come from `python scripts/run_validation.py` (stored in `docs/results/validation.json`); section 5 comes from `python scripts/run_ml_benchmark.py` (`docs/results/ml_benchmark.json`).

## 1. Physical Foundations & Beer-Lambert Law

Tunable Diode Laser Absorption Spectroscopy (TDLAS) targets the strong ro-vibrational absorption bands of methane ($CH_4$) in the mid-infrared region at $\lambda \approx 3313\text{ nm}$ ($\nu \approx 3018\text{ cm}^{-1}$, $\nu_3$ band P-branch).

### 1.1 Forward Model
The optical transmission through an effective pathlength $L$ is governed by the Beer-Lambert law:
$$I(\nu) = I_0(\nu) \exp(-A(\nu)), \qquad A(\nu) = L \sum_{i} S_i(T) N g_i(\nu - \nu_i)$$

Where:
- $N = \frac{x P}{k_B T}$ is molecular number density ($\text{molecules}\cdot\text{cm}^{-3}$).
- $x$ is the methane molar mixing ratio ($x_{\text{ppm}} \times 10^{-6}$).
- $S_i(T)$ is the temperature-adjusted line intensity. $Q(T)$ is a hand-entered cubic polynomial (giving $Q(296\ \text{K})=590.4$), not a TIPS table lookup, and the four line parameter sets in `src/tdlas/data/` were entered by hand and not checked against HITRAN:
  $$S_i(T) = S_i(T_0) \frac{Q(T_0)}{Q(T)} \exp\left( -c_2 E_i'' \left( \frac{1}{T} - \frac{1}{T_0} \right) \right) \left[ \frac{1 - e^{-c_2 \nu_{0,i}/T}}{1 - e^{-c_2 \nu_{0,i}/T_0}} \right]$$
- $\nu_{0,i}(P) = \nu_{0,i} + \delta_{\text{air},i} P$ accounts for pressure-induced line center shifting.

### 1.2 Lineshape Function (From-Scratch Faddeeva & Voigt)
Under atmospheric conditions ($P \approx 1\text{ atm}$, $T \approx 296\text{ K}$), line broadening exhibits comparable Doppler Gaussian broadening ($\gamma_D \approx 0.0046\text{ cm}^{-1}$) and Lorentzian pressure broadening ($\gamma_L \approx 0.062\text{ cm}^{-1}$). The resulting profile is a convolution represented by the normalized Voigt profile:
$$g(\nu - \nu_0) = \sqrt{\frac{\ln 2}{\pi}} \frac{1}{\gamma_D} \text{Re}(w(z)), \qquad z = \sqrt{\ln 2} \frac{(\nu - \nu_0) + i \gamma_L}{\gamma_D}$$

We implemented the complex Faddeeva function $w(z) = e^{-z^2}\text{erfc}(-iz)$ from scratch using J.A.C. Weideman's (1994) rational expansion. Against `scipy.special.wofz`, our implementation has a maximum relative error of $1.7 \times 10^{-14}$ on the grid $|x|\le 50$, $10^{-4}\le y\le 100$ ($9 \times 10^{-16}$ on the grid that covers the CH4 lines). A numerical trapezoid integral of the Voigt profile ($\gamma_D=0.015$, $\gamma_L=0.035\ \text{cm}^{-1}$, 236 cm$^{-1}$ window) gives $0.9998$; the finite window cuts the Lorentzian wings.

---

## 2. Sensor Simulation & Noise Sources

The detector voltage signal includes realistic instrument artifacts:
$$V(t) = G \cdot P_L(i(t)) \cdot [1 + \epsilon_{\text{et}}(t)] \cdot \exp(-A(\nu(t))) + n_{\text{white}}(t) + n_{1/f}(t) + n_{\text{drift}}(t)$$

1. **Laser Tuning & Baseline Power:** Current ramp produces quadratic wavenumber tuning $\nu(t) = \nu_c + a t + b t^2$ and quadratic power emission $P_L(t) = p_0 + p_1 t + p_2 t^2$.
2. **Optical Etalon Fringes:** Parasitic Fabry-Perot interference between transmissive optical elements creates periodic modulation:
   $$\epsilon_{\text{et}}(t) = m \cos\left( \frac{2\pi \nu(t)}{\text{FSR}} + \phi \right)$$
3. **Colored $1/f$ Flicker Noise:** Generated via frequency-domain filtering with spectral power decaying as $f^{-1}$.
4. **Quantization:** Sampled using a simulated 16-bit ADC over a $0 - 5\text{ V}$ full-scale range ($\text{LSB} \approx 76.3\ \mu\text{V}$).

---

## 3. Inversion via Levenberg-Marquardt with Analytical Jacobians

Concentration $x$ and baseline polynomial parameters $c = [c_0, \dots, c_K]^T$ are estimated simultaneously from measured voltage $V_{\text{meas}}$ by minimizing the non-linear sum of squares:
$$\chi^2(\theta) = \frac{1}{2} \sum_{i=1}^N \left[ V_{\text{model}}(t_i; \theta) - V_{\text{meas}}(t_i) \right]^2$$

### Analytical Jacobian
For parameter vector $\theta = [x, c_0, \dots, c_K]^T$:
$$\frac{\partial V}{\partial x} = P_L(t) e^{-A(\nu)} \cdot \left( -\alpha_{\text{unit}}(\nu) L \right)$$
$$\frac{\partial V}{\partial c_k} = t^k e^{-A(\nu)}$$

The Levenberg-Marquardt step solves:
$$(J^T J + \lambda \text{diag}(J^T J)) \Delta \theta = -J^T r$$

On noiseless synthetic signals (35 ppm, no etalon) the optimizer recovers the concentration with relative error $1.2 \times 10^{-5}$ in 3 iterations. On 30 noisy scans (15 ppm, one random device each) the custom LM estimate differs from `scipy.optimize.least_squares` by at most $1.2 \times 10^{-4}$ ppm. The Jacobian is analytical only for the concentration and the baseline polynomial coefficients; line centre and line width are not fitted. The fit is given the simulator's true wavenumber axis and the true $T$, $P$, $L$. The etalon fringe is not modelled in the fit and shows up as structured residuals.

---

## 4. Uncertainty Quantification & Allan-Werle Deviation

### 4.1 Cramér-Rao Lower Bound (CRB)
The parameter covariance is derived from the Fisher Information Matrix:
$$\text{Cov}(\hat{\theta}) \approx \sigma^2 (J^T J)^{-1}, \qquad \text{CRB}(x) = \sqrt{\left[ \sigma^2 (J^T J)^{-1} \right]_{00}}$$
Here $\sigma$ is taken as the standard deviation of the fit residuals. Results over 1000 simulated scans per row (fixed device with white noise 2.0 mV, 1/f 0.59 mV, etalon 0.35%, random-walk drift of about 5 mV per scan):

| Case | True $x$ | Std of estimate (ppm) | Mean CRB std (ppm) | 95% CI coverage |
| :--- | :--- | :--- | :--- | :--- |
| White noise only | 5 | 0.095 | 0.098 | 95.3% |
| White noise only | 25 | 0.096 | 0.100 | 95.2% |
| White noise only | 50 | 0.099 | 0.102 | 95.3% |
| White + 1/f + drift + etalon | 5 | 0.448 | 0.276 | 77.6% |
| White + 1/f + drift + etalon | 25 | 0.461 | 0.279 | 76.3% |
| White + 1/f + drift + etalon | 50 | 0.481 | 0.283 | 75.6% |

With white noise only the CRB matches the empirical spread (ratio 1.04) and coverage is about 95%. At 0 ppm the non-negativity clamp on $x$ distorts the statistics (mean 0.038, coverage 97.3%). With the default noise model the CRB underestimates the spread by about 40% and coverage falls to about 76%, so the stated 95% interval is not calibrated for this instrument model.

### 4.2 Allan-Werle Variance
Stability is assessed via Allan variance computed on non-overlapping block averages:
$$\sigma_A^2(\tau) = \frac{1}{2(M-1)} \sum_{k=1}^{M-1} (\bar{x}_{k+1} - \bar{x}_k)^2$$
On pure white noise (20000 unit-variance samples at 100 Hz) the fitted log-log slope over the first 12 points is $-0.503$. Figure 3 uses a hand-built synthetic series (100 Hz, 10000 samples: white noise 2 mV, pink noise 0.8 mV rms, linear drift 0.3 mV/s), not a series of fitted concentrations. Its slope over the first 8 points is $-0.46$, its minimum is 0.41 mV at $\tau \approx 0.9\text{ s}$, and it rises afterwards because of the added drift. Because the axis is in volts, no detection limit in ppm is derived, and the noise-budget table (switching white, 1/f, etalon and drift on and off) has not been produced.

---

## 5. Classical Inversion vs 1D CNN and Deep Ensemble Under Distribution Shifts

**Protocol** (`scripts/run_ml_benchmark.py`, results in `docs/results/ml_benchmark.json`). 100 simulated devices with 100 scans each are split by device into 60 train, 20 validation and 20 test devices, so no device appears in two splits. The 1D CNN outputs a mean and a log-variance and is trained with the Gaussian NLL, $\mathcal{L} = \tfrac12\left[e^{-s}(y-\mu)^2 + s\right]$ with $s = \log\sigma^2$, on targets divided by 50. Inputs are scans normalised as $v/\bar v - 1$ (times 10). Validation devices drive early stopping, checkpoint choice and a three-value learning-rate search (best $3\times10^{-4}$, at the edge of the grid). The ensemble has 5 members (different seeds); its mean is the average of the member means and its variance is the mean aleatoric variance plus the variance of the member means. The test devices are used only for the final evaluation (400 scans per scenario, concentration 5 to 45 ppm). The classical fit is given the true $T$, $P$, $L$ and, except in the axis-error scenario, the true wavenumber axis. Training is seeded and deterministic; two full runs gave identical numbers on this machine.

| Scenario | LM fit RMSE (ppm) / 95% cov | CNN RMSE / cov | Ensemble RMSE / cov | LM bias | Ensemble bias |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Unseen device | 0.92 / 80.2% | 2.01 / 93.2% | 2.06 / 96.8% | +0.25 | $-0.41$ |
| $T$ 330 to 350 K, $P$ 1.2 to 1.4 atm | 0.91 / 88.2% | 4.79 / 45.2% | 5.07 / 52.8% | +0.28 | $-4.39$ |
| 4x noise | 1.15 / 76.8% | 2.81 / 81.5% | 2.08 / 99.5% | +0.25 | +0.10 |
| Etalon amplitude 4 to 6% | 5.35 / 85.2% | 18.19 / 3.8% | 12.57 / 50.7% | +1.93 | +10.83 |
| Axis error ($\pm 0.03\ \text{cm}^{-1}$, $\pm 5\%$ span) | 2.36 / 58.2% | 2.98 / 88.0% | 3.06 / 93.2% | $-1.22$ | $-0.75$ |
| Unknown interferent (3 placeholder lines) | 6.57 / 0.5% | 7.95 / 7.2% | 8.51 / 7.0% | $-6.21$ | $-8.05$ |

Validation RMSE of the ensemble is 2.05 ppm. CPU batch-1 latency is 0.35 ms for the CNN and 1.64 ms for the ensemble, against about 3 to 3.6 ms for the LM fit.

**Findings.**

- The classical fit has the lower RMSE in all six scenarios, and the ensemble's RMSE is about twice as large even on unseen devices within the training distribution.
- The ensemble's intervals are better calibrated where the data stay inside the training distribution (coverage 93 to 99.5% for unseen device, 4x noise and axis error), whereas the CRB intervals of the classical fit cover 58 to 80% there because they ignore etalon, drift and 1/f noise. The ensemble also has the lower NLL in the axis-error and interferent scenarios; the classical fit has the lower NLL in unseen device, temperature/pressure and strong etalon.
- Out of range (temperature/pressure, strong etalon) the networks degrade and become over-confident (coverage 4 to 53%); gross errors above 5 ppm affect 85 to 95% of strong-etalon scans for the networks and 25% for the classical fit.
- An unmodelled interfering absorber breaks both methods (coverage below 8%).
- Axis error stays inside the axis variation present in the training devices, so the networks are affected much less than in the other shifts, while the classical fit using the nominal axis loses accuracy.

**Caveats.** The CNN is small, trained on 6000 simulated scans with three learning rates tried, so these results do not bound what a better-tuned or hybrid (network plus physical fit) model could do. The interferent lines are assumed placeholders, not real spectroscopic data. The result is from one seed and one set of simulated devices. The classical fit is given the true $T$, $P$, $L$ and axis, which favours it.

---

## 6. Interactive Visual Cockpit & Metrological Figures Export

To facilitate experimental inspection, parameter tuning, and research oversight, an interactive visual cockpit and high-resolution figure exporter were implemented:

1. **Local Interactive Cockpit (`scripts/run_dashboard.py`):**
   - Caveats: the "Physics vs. ML" tab runs the trained ensemble from section 5 (weights in `src/tdlas/data/ensemble_ch4.pt`) on one simulated scan per setting, so it illustrates single cases and is not a benchmark. The lock-in tab uses a test signal that has no 2f component. The Allan tab uses a hand-built noise series.
   - Serves an offline, responsive dark-mode cockpit (`http://127.0.0.1:8050/`) with real-time parameter sliders ($CH_4$ concentration, temperature, pressure, etalon amplitude, and detector noise).
   - Features 4 analytical tabs:
     - **Spectroscopy & Voigt Fit:** Real-time photodiode transmission $V(\nu)$ vs laser baseline, absorbance $A(\nu)$ vs Voigt fit, and $\pm 3\sigma$ fit residuals.
     - **Physics vs. ML Benchmark:** Side-by-side retrieved concentrations, 95% intervals and inference latency of the LM fit and the trained ensemble for one simulated scan, with a flag when the settings leave the ensemble's training range.
     - **Allan-Werle Variance:** Log-log stability curves displaying white-noise averaging limit and drift turnaround point $\tau_{\text{opt}}$.
     - **WMS Lock-In Demodulation:** $1f$ baseline tracking and $2f$ second harmonic peak extraction.
   - Provides 1-click scientific presets (*Nominal 50 ppm*, *Detection Limit 1 ppm*, *Etalon Interference*, and *Extreme T/P 350K*).

2. **300 DPI Publication Figure Exporter (`scripts/export_figures.py`):**
   - Automatically generates journal-quality vector SVGs and 300 DPI PNG figures saved to [`docs/figures/`](figures/):
     - `fig1_spectroscopy_voigt_fit.png` & `.svg`
     - `fig2_physics_vs_ml_comparison.png` & `.svg` (drawn from `docs/results/ml_benchmark.json`, the stored output of `scripts/run_ml_benchmark.py`)
     - `fig3_allan_variance_stability.png` & `.svg`
     - `fig4_wms_lockin_harmonics.png` & `.svg`
   - Generates a standalone interactive HTML summary report at [`docs/interactive_report.html`](interactive_report.html) for offline review.

