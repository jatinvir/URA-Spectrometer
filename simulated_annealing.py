"""
Simulated annealing solver for speckle spectrum reconstruction.

Port of MATLAB simulated_annealing.m — finds spectrum S that minimizes
||I - T @ S||^2 starting from an initial guess.
"""

import numpy as np


def perturb_spectrum(spectrum: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Modify one random element by multiplying by a random factor in [0.5, 1.5)."""
    new_spectrum = spectrum.copy()
    index_to_modify = rng.integers(len(spectrum))
    random_factor = 0.5 + rng.random()
    new_spectrum[index_to_modify] *= random_factor
    return new_spectrum


def simulated_annealing(
    I: np.ndarray,
    T: np.ndarray,
    initial_spectrum: np.ndarray,
    T0: float = 1e7,
    num_iterations: int = 10_000_000,
    cooling_rate: float = 0.9,
    seed: int | None = None,
) -> tuple[np.ndarray, float]:
    """
    Args:
        I: Measured speckle signal, shape (m,)
        T: Calibration matrix, shape (m, n)
        initial_spectrum: Starting guess, shape (n,)
        T0: Initial temperature
        num_iterations: Number of SA iterations
        cooling_rate: Multiplicative cooling factor applied each iteration
        seed: Optional RNG seed for reproducibility

    Returns:
        optimal_spectrum: Best spectrum found, shape (n,)
        optimal_energy: Final residual ||I - T @ optimal_spectrum||^2
    """
    rng = np.random.default_rng(seed)

    def energy(S: np.ndarray) -> float:
        residual = I - T @ S
        return float(residual @ residual)

    current_spectrum = initial_spectrum.copy()
    current_energy = energy(current_spectrum)
    optimal_spectrum = current_spectrum.copy()
    optimal_energy = current_energy

    temperature = T0
    for _ in range(num_iterations):
        new_spectrum = perturb_spectrum(current_spectrum, rng)
        new_energy = energy(new_spectrum)
        delta_energy = new_energy - current_energy

        # Metropolis acceptance
        if rng.random() < np.exp(-delta_energy / temperature):
            current_spectrum = new_spectrum
            current_energy = new_energy
            if current_energy < optimal_energy:
                optimal_spectrum = current_spectrum.copy()
                optimal_energy = current_energy

        # Cool down (note: MATLAB cools every iteration, very aggressive)
        temperature *= cooling_rate

    return optimal_spectrum, optimal_energy