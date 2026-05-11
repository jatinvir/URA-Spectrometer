"""
SVD + simulated annealing speckle spectrum reconstruction.

Port of the MATLAB main script. Reconstructs a spectrum from a speckle
measurement using a truncated-SVD pseudo-inverse, then refines with SA.
"""

import time

import numpy as np
from scipy.io import loadmat
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from simulated_annealing import simulated_annealing


# ---------- Configuration ----------
CAL_MAT_FILE = "transmission_matrices/matrix_cal_10.mat"
PROBE_MAT_FILE = "transmission_matrices/matrix_cal_10_probe.mat"

MERGE_LAMBDA = 1                    # Column-merge factor for T
SVD_THRESHOLD = 5000                # Singular values <= this get zeroed
NUM_SA_ITERATIONS = 10_000_000      # SA iterations
T0 = 1e7                            # SA initial temperature
START_WAVELENGTH = 1550             # nm
STEP_SIZE = 0.01 * MERGE_LAMBDA     # nm per bin
PROBE_SCALE = 2.73                  # Scale factor applied to probe in MATLAB
PROBE_INDEX = 1400                  # Which probe column to test (m in MATLAB)


# ---------- Helpers ----------
def merge_columns(matrix: np.ndarray, merge_factor: int) -> np.ndarray:
    """Average groups of `merge_factor` consecutive columns."""
    m, n = matrix.shape
    reduced_n = n // merge_factor
    out = np.zeros((m, reduced_n))
    for i in range(reduced_n):
        out[:, i] = matrix[:, i * merge_factor:(i + 1) * merge_factor].mean(axis=1)
    return out


def truncated_svd_pseudoinverse(T: np.ndarray, threshold: float) -> np.ndarray:
    """Compute pseudo-inverse keeping only singular values above threshold."""
    U, s, Vt = np.linalg.svd(T, full_matrices=False)
    s_inv = np.where(s > threshold, 1.0 / s, 0.0)
    return Vt.T @ np.diag(s_inv) @ U.T


def lorentzian(x: np.ndarray, amp: float, center: float, width: float) -> np.ndarray:
    return amp / ((x - center) ** 2 + width ** 2)


# ---------- Main pipeline ----------
def reconstruct_spectrum(
    I: np.ndarray,
    T: np.ndarray,
    T_pinv: np.ndarray,
    num_iterations: int = NUM_SA_ITERATIONS,
    T0: float = T0,
    seed: int | None = None,
) -> dict:
    """Run SVD-only and SVD+SA reconstruction on a single measurement."""
    # SVD-only reconstruction
    t_start = time.perf_counter()
    S_guess = T_pinv @ I
    svd_time = time.perf_counter() - t_start

    # Normalize for SA initial guess (matches MATLAB: abs(S_guess / max(S_guess)))
    max_val = np.max(S_guess)
    if max_val == 0:
        max_val = 1.0
    initial_spectrum = np.abs(S_guess / max_val)

    # SVD + simulated annealing
    t_start = time.perf_counter()
    optimal_spectrum, optimal_energy = simulated_annealing(
        I, T, initial_spectrum,
        T0=T0, num_iterations=num_iterations, seed=seed,
    )
    sa_time = time.perf_counter() - t_start

    return {
        "svd_spectrum": S_guess,
        "svd_time": svd_time,
        "sa_spectrum": optimal_spectrum,
        "sa_energy": optimal_energy,
        "sa_time": sa_time,
    }


def fit_peak(spectrum: np.ndarray, xdata: np.ndarray) -> tuple[float, dict]:
    """Find dominant peak and fit a Lorentzian. Returns (peak_x, fit_info)."""
    peaks, _ = find_peaks(spectrum)
    if len(peaks) == 0:
        return float("nan"), {}
    peak_locs = peaks[np.argsort(spectrum[peaks])[::-1]]
    initial_peak = peak_locs[0]

    try:
        popt, _ = curve_fit(
            lorentzian, xdata, spectrum,
            p0=[1.0, float(initial_peak), 1.0],
            maxfev=2000,
        )
        fitted = lorentzian(xdata, *popt)
        idx_peak = int(np.argmax(fitted))
        peak_x = float(xdata[idx_peak])
        return peak_x, {"params": popt, "idx_peak": idx_peak, "fitted": fitted}
    except Exception as e:
        return float(initial_peak), {"error": str(e)}


def main():
    # Load calibration matrix
    cal_data = loadmat(CAL_MAT_FILE)
    T_full = cal_data["mat2"]
    T = merge_columns(T_full, MERGE_LAMBDA)
    print(f"Calibration matrix T: shape {T.shape}")

    # Build truncated-SVD pseudo-inverse
    T_pinv = truncated_svd_pseudoinverse(T, SVD_THRESHOLD)
    print(f"Pseudo-inverse: shape {T_pinv.shape}")

    # Load probe data
    probe_data = loadmat(PROBE_MAT_FILE)
    P = probe_data["mat2"]

    # Build measurement
    m = PROBE_INDEX
    I = P[:, m] * PROBE_SCALE
    peak1_input_nm = START_WAVELENGTH + m * STEP_SIZE
    print(f"\nTrue peak wavelength: {peak1_input_nm:.3f} nm")

    # Reconstruct
    result = reconstruct_spectrum(I, T, T_pinv)
    print(f"SVD-only time: {result['svd_time']*1000:.2f} ms")
    print(f"SVD+SA time:   {result['sa_time']:.2f} s")
    print(f"SA final energy: {result['sa_energy']:.4e}")

    # Peak fit on SA spectrum
    n_bins = T.shape[1]
    xdata = np.linspace(1, n_bins, n_bins)
    peak_idx, fit_info = fit_peak(result["sa_spectrum"], xdata)
    peak1_recovered_nm = START_WAVELENGTH + peak_idx * STEP_SIZE
    deviation_nm = peak1_input_nm - peak1_recovered_nm
    print(f"Recovered peak: {peak1_recovered_nm:.3f} nm")
    print(f"Deviation: {deviation_nm:.4f} nm")

    # Reconstruction error proxy (residual std, like MATLAB's spectrum_error)
    spectrum_error = np.std(result["svd_spectrum"] - result["sa_spectrum"])
    print(f"std(SVD - SA): {spectrum_error:.4e}")


if __name__ == "__main__":
    main()