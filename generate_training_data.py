import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.io

wavelengths = np.linspace(1550, 1565, 1500)

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
        width = random.uniform(5,700)
        height = random.uniform(0.1,1.0)

        spectrum += broad_curve(spectrum_clean, center, width, height)
    
    spectrum = spectrum / np.max(spectrum)

    return spectrum



# broad abosportion
# bell shape wide smooth dips
def generate_broad_absorption(number_of_peaks):
    spectrum = 1.0 - generate_broad_emission(number_of_peaks)
    return spectrum


# Create a 2x2 grid
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Ground Truth Spectrum Generation (No Noise)', fontsize=16)

# 1. Sparse Emission
axs[0, 0].plot(wavelengths, generate_sparse_spectrum(5), color='tab:blue')
axs[0, 0].set_title('Type 1: Sparse Emission')
axs[0, 0].set_ylim(-0.1, 1.1)

# 2. Narrow Absorption
axs[0, 1].plot(wavelengths, generate_narrow_absorption(5), color='tab:orange')
axs[0, 1].set_title('Type 2: Narrow Absorption')
axs[0, 1].set_ylim(-0.1, 1.1)

# 3. Broad Emission
axs[1, 0].plot(wavelengths, generate_broad_emission(5), color='tab:green')
axs[1, 0].set_title('Type 3: Broad Emission')
axs[1, 0].set_ylim(-0.1, 1.1)

# 4. Broad Absorption
axs[1, 1].plot(wavelengths, generate_broad_absorption(5), color='tab:red')
axs[1, 1].set_title('Type 4: Broad Absorption')
axs[1, 1].set_ylim(-0.1, 1.1)

# Make it look nice
for ax in axs.flat:
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Intensity')

# plt.tight_layout()
# plt.show()

rows, cols = 4, 4
fig, axs = plt.subplots(rows, cols, figsize=(16, 12))
plt.subplots_adjust(hspace=0.4, wspace=0.3)

types = ['Sparse', 'Narrow Abs', 'Broad Emis', 'Broad Abs']

for i in range(rows):
    for j in range(cols):
        # Cycle through your 4 types
        current_type = types[i]
        
        if current_type == 'Sparse':
            data = generate_sparse_spectrum(random.randint(2, 10))
            color = 'tab:blue'
        elif current_type == 'Narrow Abs':
            data = generate_narrow_absorption(random.randint(2, 10))
            color = 'tab:orange'
        elif current_type == 'Broad Emis':
            data = generate_broad_emission(random.randint(1, 4))
            color = 'tab:green'
        else:
            data = generate_broad_absorption(random.randint(1, 4))
            color = 'tab:red'
            
        axs[i, j].plot(wavelengths, data, color=color, linewidth=1)
        axs[i, j].set_title(f"{current_type} Sample {j+1}", fontsize=10)
        axs[i, j].set_ylim(-0.05, 1.05)
        axs[i, j].tick_params(axis='both', which='major', labelsize=8)

fig.suptitle('Batch Synthetic Spectra Generation (Ground Truths)', fontsize=18)
# plt.show()

def plot_spectrum_grid(rows=4, cols=4):
    fig, axs = plt.subplots(rows, cols, figsize=(16, 12))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    # Define our types and matching colors
    types = [
        ('Sparse Emission', 'tab:blue', generate_sparse_spectrum),
        ('Narrow Absorption', 'tab:orange', generate_narrow_absorption),
        ('Broad Emission', 'tab:green', generate_broad_emission),
        ('Broad Absorption', 'tab:red', generate_broad_absorption)
    ]

    for i in range(rows):
        # Pick the type for this entire row
        label, color, func = types[i]
        
        for j in range(cols):
            ax = axs[i, j]
            
            # Randomize the "intensity" or "complexity" for variety
            if 'Broad' in label:
                n = random.randint(1, 5) # 1 to 5 broad peaks
            else:
                n = random.randint(3, 15) # 3 to 15 sharp spikes
            
            data = func(n)
            
            ax.plot(data, color=color, linewidth=1)
            ax.set_title(f"{label} (n={n})", fontsize=10)
            ax.set_ylim(-0.05, 1.05)
            
            # Clean up the labels so it's not cluttered
            if j > 0: ax.set_yticklabels([])
            if i < rows - 1: ax.set_xticklabels([])

    fig.suptitle('Variation Gallery: 4 Types of Synthetic Spectra', fontsize=20, y=0.95)
    # plt.show()

# plot_spectrum_grid()


# load transmission matrix

# main matrix for training
calibration_matrix = scipy.io.loadmat("Matrix_calabration_Dis7.mat")['mat2']
#matrix for validation set
probe_matrix = scipy.io.loadmat("Matrix_probe_Dis7.mat")['mat2']

print(calibration_matrix.shape)
print(probe_matrix.shape)



# generate training set
X_train_list = [] # device measurements that get inputted to the NN
y_train_list = [] # spectra that are targets for the NN

def simulate_measurement(spectrum, T, snr_db = 30):
    measurement = np.dot(T, spectrum)
    signal_power = np.mean(measurement ** 2)
    noise_level = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_level), measurement.shape)
    return measurement + noise


for _ in range(3000):
    
    spectrum_generator = random.choice([
        generate_sparse_spectrum,
        generate_broad_emission,
        generate_narrow_absorption,
        generate_broad_absorption
    ])

    spectrum = spectrum_generator(random.randint(1,10))
    snr_db = random.choice([30, 40, 50])
    measurement = simulate_measurement(spectrum, calibration_matrix, snr_db)

    X_train_list.append(measurement)
    y_train_list.append(spectrum)
    
X_train = np.array(X_train_list)
y_train = np.array(y_train_list)

print(f"X_train shape: {X_train.shape}") # Should be (3000, 25)
# print(X_train[:5])
# print(f"y_train shape: {y_train.shape}") # Should be (3000, 1500)


# now perform the SVD part to get physically informed first draft
# T = U * Sigma * V^T
# (source) x = V * inverse(Sigma) * U^T * y

U, S, Vh = np.linalg.svd(calibration_matrix, full_matrices=False)

plt.figure(figsize=(10, 4))
plt.plot(Vh[0, :], label='Shape 1 (Strongest)')
plt.plot(Vh[1, :], label='Shape 2')
plt.plot(Vh[2, :], label='Shape 3')
plt.legend()
plt.title("The SVD Dictionary: The 3 Most Important Shapes our Chip Sees")
# plt.show()

print(S)

S_inv_matrix = np.diag(1 / S)
print(S_inv_matrix.shape)
# print(S_inv_matrix)

print(Vh.shape)
print(S_inv_matrix.shape)
print(U.T.shape)


inflation_matrix = Vh.T @ S_inv_matrix @ U.T
print(inflation_matrix.shape)
# final_matrix = inflation_matrix 
X_train_rough = X_train @ inflation_matrix.T
print(X_train_rough)

sample_idx = 42

plt.figure(figsize=(12, 5))
plt.plot(wavelengths, y_train[sample_idx], label="Ground Truth (The Target)", color='black', alpha=0.5)
plt.plot(wavelengths, X_train_rough[sample_idx], label="SVD Rough Sketch (The Input)", color='red')
plt.title(f"Sample {sample_idx}: Ground Truth vs. SVD Reconstruction")
plt.legend()
plt.show()