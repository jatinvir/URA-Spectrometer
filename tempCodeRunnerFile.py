import numpy as np
import matplotlib.pyplot as plt
import random

# sparse
# spectrum is entirely zero, with some thin sharp spikes
def generate_sparse_spectrum(number_of_spikes):
    spectrum = np.zeros(1500)
    random_spike_indices = random.sample(range(0,1500), number_of_spikes)
    random_spike_intensities = [random.random() for _ in range(number_of_spikes)]

    spectrum[random_spike_indices] = random_spike_intensities

    return spectrum









# narrow absorption
# full with some sharp drips
def generate_narrow_absorption(number_of_spikes):
    spectrum = np.ones(1500)
    random_spike_indices = random.sample(range(0,1500), number_of_spikes)
    random_spike_intensities = [1 - random.random() for _ in range(number_of_spikes)]

    spectrum[random_spike_indices] = random_spike_intensities

    return spectrum


def broad_curve(x, x0, gamma, I_peak):
    return I_peak / (1 + ((x - x0) / gamma) ** 2)

# broad emission
# bell wide smooth surves 
def generate_broad_emission(number_of_peaks):
    spectrum = np.zeros(1500)
    spectrum_clean = np.arange(1500)
    
    for _ in range(number_of_peaks):
        center = random.uniform(0,1500)
        width = random.uniform(5,50)
        height = random.uniform(0.1,1.0)

        spectrum += broad_curve(spectrum_clean, center, width, height)
    
    spectrum = spectrum / np.max(spectrum)

    return spectrum



# broad abosportion
# bell shape wide smooth dips
def generate_broad_absorption(number_of_peaks):
    spectrum = 1.0 - generate_broad_emission(number_of_peaks)
    return spectrum


import matplotlib.pyplot as plt

# Create a 2x2 grid
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Ground Truth Spectrum Generation (No Noise)', fontsize=16)

# 1. Sparse Emission
axs[0, 0].plot(generate_sparse_spectrum(5), color='tab:blue')
axs[0, 0].set_title('Type 1: Sparse Emission')
axs[0, 0].set_ylim(-0.1, 1.1)

# 2. Narrow Absorption
axs[0, 1].plot(generate_narrow_absorption(5), color='tab:orange')
axs[0, 1].set_title('Type 2: Narrow Absorption')
axs[0, 1].set_ylim(-0.1, 1.1)

# 3. Broad Emission
axs[1, 0].plot(generate_broad_emission(5), color='tab:green')
axs[1, 0].set_title('Type 3: Broad Emission')
axs[1, 0].set_ylim(-0.1, 1.1)

# 4. Broad Absorption
axs[1, 1].plot(generate_broad_absorption(5), color='tab:red')
axs[1, 1].set_title('Type 4: Broad Absorption')
axs[1, 1].set_ylim(-0.1, 1.1)

# Make it look nice
for ax in axs.flat:
    ax.set_xlabel('Wavelength Index')
    ax.set_ylabel('Intensity')

plt.tight_layout()
plt.show()

