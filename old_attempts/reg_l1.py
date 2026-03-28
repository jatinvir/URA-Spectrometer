import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.io
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.optimize import nnls
import sys

plt.ioff()
wavelengths = np.linspace(1550, 1565, 1500)

def generate_sparse_spectrum(number_of_spikes):
    x = np.arange(1500)
    spectrum = np.zeros(1500)
    for _ in range(number_of_spikes):
        center = random.uniform(0, 1500)
        width = random.uniform(0.5, 2.0)
        intensity = random.random()
        spectrum += intensity * np.exp(-((x - center)**2) / (2 * width**2))
    return spectrum / np.max(spectrum)

def generate_narrow_absorption(number_of_spikes):
    x = np.arange(1500)
    spectrum = np.ones(1500)
    for _ in range(number_of_spikes):
        center = random.uniform(0, 1500)
        width = random.uniform(0.5, 2.0)
        intensity = random.random()
        spectrum -= intensity * np.exp(-((x - center)**2) / (2 * width**2))
    return np.clip(spectrum, 0, 1)

def broad_curve(x, x0, gamma, I_peak):
    return I_peak / (1 + ((x - x0) / gamma) ** 2)

def generate_broad_emission(number_of_peaks):
    spectrum = np.zeros(1500)
    spectrum_clean = np.arange(1500)
    for _ in range(number_of_peaks):
        center = random.uniform(0,1500)
        width = random.uniform(5,700)
        height = random.uniform(0.1,1.0)
        spectrum += broad_curve(spectrum_clean, center, width, height)
    return spectrum / np.max(spectrum)

def generate_broad_absorption(number_of_peaks):
    return 1.0 - generate_broad_emission(number_of_peaks)

# Load Matrices
calibration_matrix = scipy.io.loadmat("transmission_matrices/Matrix_calabration_Dis7.mat")['mat2']

def simulate_measurement(spectrum, T, snr_db=30):
    measurement = np.dot(T, spectrum)
    signal_power = np.mean(measurement ** 2)
    noise_level = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_level), measurement.shape)
    return measurement + noise

random.seed(42)
np.random.seed(42)
print("Generating 3000 synthetic samples...")
X_train_list, y_train_list = [], []
for _ in range(3000):
    spectrum_generator = random.choice([
        generate_sparse_spectrum, generate_broad_emission,
        generate_narrow_absorption, generate_broad_absorption
    ])
    spectrum = spectrum_generator(random.randint(1,10))
    snr_db = random.choice([30, 40, 50])
    measurement = simulate_measurement(spectrum, calibration_matrix, snr_db)
    
    X_train_list.append(measurement)
    y_train_list.append(spectrum)

X_train_np = np.array(X_train_list)
y_train_np = np.array(y_train_list)

print("Running Reg NNLS + L1 to generate rough drafts...")

# the difference operator
# want it to be smooth, and rough line
# means adjacent indices have huge absolute jumps
# this is matrix of -1 and 1 on diagonal, so $(x_{i+1} - x_i)$
# so dont solve Tx = y, optimize it via Ridge Regression, and want x >= 0
# $$\min_{x \ge 0} \left( ||Tx - y||_2^2 + \alpha ||Lx||_2^2 \right)$$
# \alpha ||Lx||_2^2 \right)$$ smoothness, penalize big jumps, alpha is how much we care about this
# this good for broad, for sparse, it spreads out large penalities
def build_L(n):
    L = np.zeros((n-1, n))
    for i in range(n-1):
        L[i, i] = -1
        L[i, i+1] = 1
    return L

# 50 means we care a lot
alpha = 500.0
T = calibration_matrix
L = build_L(1500)
# T = [T, a*L]^t, y = [y 0]^t
T_aug = np.vstack([T, np.sqrt(alpha) * L])

X_train_rough = []
for i in range(5):
    if i % 1 == 0:
        print(f"{i}/5 done...")
    y_aug = np.concatenate([X_train_np[i], np.zeros(1499)])
    x, _ = nnls(T_aug, y_aug)
    X_train_rough.append(x)

X_train_rough = np.array(X_train_rough)

# X_train_rough = X_rough_guess.detach().cpu().numpy()


sample_idx = 0
plt.figure(figsize=(12, 5))
plt.plot(wavelengths, y_train_np[sample_idx], label="Ground Truth", color="black", linewidth=2)
plt.plot(wavelengths, X_train_rough[sample_idx], label="Reg NNLS + L1 Output", color="blue", alpha=0.8)
plt.title(f"Solver Sanity Check: Reg NNLS + L1 Result (Sample {sample_idx})")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Normalized Intensity")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show(block=True)

