"""From-scratch Voigt line profile via Weideman rational approximation of the Faddeeva function.

References:
- J.A.C. Weideman, Computation of the Complex Error Function, SIAM J. Numer. Anal. 31(5), pp. 1497-1518, 1994.
- Armstrong, Spectrum Line Profiles: The Voigt Function, JQSRT 7(1), 1967.
"""
from __future__ import annotations
import numpy as np

def faddeeva_rational(z: complex | np.ndarray, N: int = 36) -> complex | np.ndarray:
    """Compute the Faddeeva function w(z) = exp(-z^2) * erfc(-i*z) from scratch.
    
    Implements J.A.C. Weideman's (1994) rational expansion with N poles.
    Achieves near-machine precision (relative error < 1e-12) across the upper complex half-plane.
    For points in the lower half-plane (Im(z) < 0), applies the reflection identity:
        w(-z) = 2*exp(-z^2) - w(z)
    """
    z_arr = np.asarray(z, dtype=np.complex128)
    orig_shape = z_arr.shape
    z_flat = z_arr.ravel()
    
    is_lower = np.imag(z_flat) < 0
    z_calc = np.where(is_lower, -z_flat, z_flat)
    
    M = 2 * N
    M2 = 2 * M
    k = np.arange(-M + 1, M, dtype=np.float64)
    L = np.sqrt(N / np.sqrt(2.0))
    theta = k * np.pi / M
    t = L * np.tan(theta / 2.0)
    
    f = np.exp(-t ** 2) * (L ** 2 + t ** 2)
    f = np.insert(f, 0, 0.0)
    
    a = np.real(np.fft.fft(np.fft.fftshift(f))) / M2
    a = a[1 : N + 1][::-1]
    
    Z = (L + 1j * z_calc) / (L - 1j * z_calc)
    
    # Horner's polynomial evaluation
    poly = np.zeros_like(z_calc, dtype=np.complex128)
    for n in range(N):
        poly = poly * Z + a[n]
        
    denom_sq = (L - 1j * z_calc) ** 2
    denom = L - 1j * z_calc
    
    w_upper = 2.0 * poly / denom_sq + (1.0 / (np.sqrt(np.pi) * denom))
    if np.any(is_lower):
        res = w_upper.copy()
        res[is_lower] = 2.0 * np.exp(-z_flat[is_lower] ** 2) - w_upper[is_lower]
    else:
        res = w_upper
    
    if orig_shape == ():
        return complex(res[0])
    return res.reshape(orig_shape)

def voigt_profile(
    nu_cm1: np.ndarray,
    nu0_cm1: float,
    gamma_D_cm1: float,
    gamma_L_cm1: float
) -> np.ndarray:
    """Normalized Voigt line shape function g(nu - nu0) [cm].
    
    Formula:
        g(nu - nu0) = (sqrt(ln 2 / pi) / gamma_D) * Re(w(z))
        where z = sqrt(ln 2) * [(nu - nu0) + i*gamma_L] / gamma_D
    
    Properties:
        Integral over (-inf, inf) equals 1.0.
        Approaches Gaussian as gamma_L -> 0.
        Approaches Lorentzian as gamma_D -> 0.
    """
    if gamma_D_cm1 <= 0:
        raise ValueError(f"gamma_D must be > 0, got {gamma_D_cm1}")
    if gamma_L_cm1 < 0:
        raise ValueError(f"gamma_L must be >= 0, got {gamma_L_cm1}")
        
    sqrt_ln2 = np.sqrt(np.log(2.0))
    inv_gamma_D = 1.0 / gamma_D_cm1
    
    z = sqrt_ln2 * ((nu_cm1 - nu0_cm1) + 1j * gamma_L_cm1) * inv_gamma_D
    w_z = faddeeva_rational(z)
    
    profile = (sqrt_ln2 / np.sqrt(np.pi)) * inv_gamma_D * np.real(w_z)
    return np.maximum(profile, 0.0)
