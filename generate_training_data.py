import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.io
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
plt.ioff()
wavelengths = np.linspace(1550, 1565, 1500)

# sparse
# spectrum is entirely zero, with some thin sharp spikes
def generate_sparse_spectrum(number_of_spikes):
    x = np.arange(1500)
    spectrum = np.zeros(1500)

    for _ in range(number_of_spikes):
        center = random.uniform(0, 1500)
        width = random.uniform(0.5, 2.0)
        intensity = random.random()
        spectrum += intensity * np.exp(-((x - center)**2) / (2 * width**2))
    return spectrum / np.max(spectrum)



# narrow absorption
# full with some sharp drips
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


# # Create a 2x2 grid
# fig, axs = plt.subplots(2, 2, figsize=(12, 8))
# fig.suptitle('Ground Truth Spectrum Generation (No Noise)', fontsize=16)

# # 1. Sparse Emission
# axs[0, 0].plot(wavelengths, generate_sparse_spectrum(5), color='tab:blue')
# axs[0, 0].set_title('Type 1: Sparse Emission')
# axs[0, 0].set_ylim(-0.1, 1.1)

# # 2. Narrow Absorption
# axs[0, 1].plot(wavelengths, generate_narrow_absorption(5), color='tab:orange')
# axs[0, 1].set_title('Type 2: Narrow Absorption')
# axs[0, 1].set_ylim(-0.1, 1.1)

# # 3. Broad Emission
# axs[1, 0].plot(wavelengths, generate_broad_emission(5), color='tab:green')
# axs[1, 0].set_title('Type 3: Broad Emission')
# axs[1, 0].set_ylim(-0.1, 1.1)

# # 4. Broad Absorption
# axs[1, 1].plot(wavelengths, generate_broad_absorption(5), color='tab:red')
# axs[1, 1].set_title('Type 4: Broad Absorption')
# axs[1, 1].set_ylim(-0.1, 1.1)

# # Make it look nice
# for ax in axs.flat:
#     ax.set_xlabel('Wavelength (nm)')
#     ax.set_ylabel('Intensity')

# plt.tight_layout()
# plt.show()

# rows, cols = 4, 4
# fig, axs = plt.subplots(rows, cols, figsize=(16, 12))
# plt.subplots_adjust(hspace=0.4, wspace=0.3)

# types = ['Sparse', 'Narrow Abs', 'Broad Emis', 'Broad Abs']

# for i in range(rows):
#     for j in range(cols):
#         # Cycle through your 4 types
#         current_type = types[i]
        
#         if current_type == 'Sparse':
#             data = generate_sparse_spectrum(random.randint(2, 10))
#             color = 'tab:blue'
#         elif current_type == 'Narrow Abs':
#             data = generate_narrow_absorption(random.randint(2, 10))
#             color = 'tab:orange'
#         elif current_type == 'Broad Emis':
#             data = generate_broad_emission(random.randint(1, 4))
#             color = 'tab:green'
#         else:
#             data = generate_broad_absorption(random.randint(1, 4))
#             color = 'tab:red'
            
#         # axs[i, j].plot(wavelengths, data, color=color, linewidth=1)
#         axs[i, j].set_title(f"{current_type} Sample {j+1}", fontsize=10)
#         axs[i, j].set_ylim(-0.05, 1.05)
#         axs[i, j].tick_params(axis='both', which='major', labelsize=8)

# # fig.suptitle('Batch Synthetic Spectra Generation (Ground Truths)', fontsize=18)
# # plt.show()

# def plot_spectrum_grid(rows=4, cols=4):
#     fig, axs = plt.subplots(rows, cols, figsize=(16, 12))
#     # plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
#     # Define our types and matching colors
#     types = [
#         ('Sparse Emission', 'tab:blue', generate_sparse_spectrum),
#         ('Narrow Absorption', 'tab:orange', generate_narrow_absorption),
#         ('Broad Emission', 'tab:green', generate_broad_emission),
#         ('Broad Absorption', 'tab:red', generate_broad_absorption)
#     ]

#     for i in range(rows):
#         # Pick the type for this entire row
#         label, color, func = types[i]
        
#         for j in range(cols):
#             ax = axs[i, j]
            
#             # Randomize the "intensity" or "complexity" for variety
#             if 'Broad' in label:
#                 n = random.randint(1, 5) # 1 to 5 broad peaks
#             else:
#                 n = random.randint(3, 15) # 3 to 15 sharp spikes
            
#             data = func(n)
            
#             # ax.plot(data, color=color, linewidth=1)
#             # ax.set_title(f"{label} (n={n})", fontsize=10)
#             # ax.set_ylim(-0.05, 1.05)
            
#             # Clean up the labels so it's not cluttered
#             if j > 0: ax.set_yticklabels([])
#             if i < rows - 1: ax.set_xticklabels([])

    # fig.suptitle('Variation Gallery: 4 Types of Synthetic Spectra', fontsize=20, y=0.95)
    # plt.show()

# plot_spectrum_grid()


# load transmission matrix
# main matrix for training
calibration_matrix = scipy.io.loadmat("Matrix_calabration_Dis7.mat")['mat2']
#matrix for validation set
probe_matrix = scipy.io.loadmat("Matrix_probe_Dis7.mat")['mat2']

print(calibration_matrix.shape)
print(probe_matrix.shape)

# this is to peanlize missing peaks
# right now it hallucinates to not miss peaks
def weighted_mse_loss(input, target, weight):
    sqaured_errors = (input - target) ** 2
    weighted_errors = weight * sqaured_errors
    return weighted_errors.sum() / weight.sum()

# generate training set
X_train_list = [] # device measurements that get inputted to the NN
y_train_list = [] # spectra that are targets for the NN

# forward model: y = T*x + error
def simulate_measurement(spectrum, T, snr_db = 30):
    # T * spectrum (dot product of them)
    measurement = np.dot(T, spectrum)
    # average energy of light hitting sensor
    signal_power = np.mean(measurement ** 2)
    # noise level using decibel formula
    noise_level = signal_power / (10 ** (snr_db / 10))
    # generate Gaussian noise
    noise = np.random.normal(0, np.sqrt(noise_level), measurement.shape)
    return measurement + noise

# 3000 samples
for _ in range(3000):
    
    # randomly pick light spectrum
    spectrum_generator = random.choice([
        generate_sparse_spectrum,
        generate_broad_emission,
        generate_narrow_absorption,
        generate_broad_absorption
    ])

    # generate spectrum based on the one that was picked
    spectrum = spectrum_generator(random.randint(1,10))
    # randomly pick snr (noise), not fit on one noise level
    snr_db = random.choice([30, 40, 50])
    # call function to generate the spectrum with noise
    measurement = simulate_measurement(spectrum, calibration_matrix, snr_db)

    # so X_train has the noisy spectrum images
    X_train_list.append(measurement)
    # y_train has the perfect spectrums
    y_train_list.append(spectrum)

# 25 point measurements
X_train = np.array(X_train_list)
y_train = np.array(y_train_list)

# print(f"X_train shape: {X_train.shape}") # Should be (3000, 25)
# print(X_train[:5])
# print(f"y_train shape: {y_train.shape}") # Should be (3000, 1500)





# now perform the SVD part to get physically informed first draft
# T = U Sigma V^T
U, S, Vh = np.linalg.svd(calibration_matrix, full_matrices=False)

# plt.figure(figsize=(10, 4))
# plt.plot(Vh[0, :], label='Shape 1 (Strongest)')
# plt.plot(Vh[1, :], label='Shape 2')
# plt.plot(Vh[2, :], label='Shape 3')
# plt.legend()
# plt.title("The SVD Dictionary: The 3 Most Important Shapes our Chip Sees")
# plt.show()

# print(S)
# dynamic threshold to filter out low values
ratio = 0.05
threshold = ratio * S[0]
# only invert singular values that are larger than the threshold
S_inv_matrix = np.diag(np.where(S > threshold, 1 / S, 0))
# print(S_inv_matrix)

# print(S_inv_matrix.shape)
# print(S_inv_matrix)

# print(Vh.shape)
# print(S_inv_matrix.shape)
# print(U.T.shape)


inflation_matrix = Vh.T @ S_inv_matrix @ U.T
# print(inflation_matrix.shape)

# X_train = raw set of 3000 measurements (3000, 25)
# so take 25 pixel measurement and inflate to 1500 rough guess
X_train_rough = X_train @ inflation_matrix.T
# (3000, 1500)
# print(X_train_rough)

sample_idx = 42

plt.figure(figsize=(12, 5))
plt.plot(wavelengths, y_train[sample_idx], label="Ground Truth (The Target)", color='black', alpha=0.5)
plt.plot(wavelengths, X_train_rough[sample_idx], label="SVD Rough Sketch (The Input)", color='red')
plt.title(f"Sample {sample_idx}: Ground Truth vs. SVD Reconstruction")
plt.legend()
plt.show()


# now build neural network to clear up this rough mess, MLP
# need to experiemnt with different node sizes
# we have 25, they had 108
# Input: 1500 nodes
# Layer 1 : 32 nodes
# Layer 2 : 32 nodes
# OutputL 1500 nodes

# divide into batch of 32

class SolverNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        # y = Wx + b
        self.layer1 = nn.Linear(1500, 512)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 512)
        self.layer4 = nn.Linear(512, 1500)

        
        # constraint
        # forces value to be between 0 and 1
        self.ReLU = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.ReLU(x)

        x = self.layer2(x)
        x = self.ReLU(x)

        x = self.layer3(x)
        x = self.ReLU(x)
        
        x = self.layer4(x)
        x = self.ReLU(x)

        return x


# generate pytorch dataset
class SpectrumDataset(Dataset):
    def __init__(self, X_numpy, y_numpy):
        self.X = torch.tensor(X_numpy, dtype=torch.float32)
        self.y = torch.tensor(y_numpy, dtype=torch.float32)


    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# split data into train and val
# 80% training 20% val
split_idx = int(0.8 * len(X_train_rough))

# training
X_train_final = X_train_rough[:split_idx]
y_train_final = y_train[:split_idx]

#validation
X_val_ref = X_train_rough[split_idx:]
y_val_ref = y_train[split_idx:]


# has 3000 samples total
train_dataset = SpectrumDataset(X_train_final, y_train_final)
val_dataset = SpectrumDataset(X_val_ref, y_val_ref)

# train in batches of 32
# shuffle so never see data in the same order twice 
train_loader = DataLoader(train_dataset, batch_size = 32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


model = SolverNetwork()
# loss_function = weighted_mse_loss()
# how much to adjust weight after each backprop
# for large jumpts, slow them down, for small jumps, make them bigger
# lr = step size
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

# need more since its still converging, more training
num_of_epochs = 200
loss_history = []
print("starting epochs: ")
for epoch in range(num_of_epochs):
    model.train()
    
    train_loss = 0.0

    # do in batches of 32
    # push 32 rough sketches into 32 node-bottleneck (adjust this)
    for X_batch, y_batch in train_loader:
        

        predictions = model(X_batch)

        # compare prediction to perfect ground truth
        loss = weighted_mse_loss(predictions, y_batch, 1 + 70 * torch.abs(y_batch))

        # to reset memory from previous batch
        optimizer.zero_grad()

        # takes loss and runs it backward, calculates partial derivative of weights
        loss.backward()
        # updates the updates from backprop
        optimizer.step()

        train_loss += loss.item()
    
    # lock model and then shut off backprop, runs once per epoch
    model.eval()
    running_val_loss = 0.0

    # test on 20% of data
    with torch.no_grad():
        for X_val, y_val in val_loader:
            val_preds = model(X_val)
            batch_loss = weighted_mse_loss(val_preds, y_val, 1 + 70 * torch.abs(y_val))
            running_val_loss += batch_loss.item()

    
    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = running_val_loss / len(val_loader)
    loss_history.append(avg_val_loss)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_of_epochs}] | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")


# 1. Lock the model for testing
model.eval()

# 2. Grab exactly one batch of validation data
with torch.no_grad():
    X_val_batch, y_val_batch = next(iter(val_loader))
    
    predictions = model(X_val_batch)

true_spectrum = y_val_batch[0].numpy()
predicted_spectrum = predictions[0].numpy()
wavelengths = np.linspace(1550, 1565, 1500)

# 4. Plot the overlay
plt.figure(figsize=(10, 5))
plt.plot(wavelengths, true_spectrum, label="Ground Truth (Clean)", color="black", linewidth=2)
plt.plot(wavelengths, predicted_spectrum, label="NN Prediction", color="red", linestyle="--", linewidth=2)

plt.title("Solver-Informed Reconstruction: Prediction vs. Reality")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Normalized Intensity")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
print("training done")



# plt.plot(loss_history)
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title("Training Loss")
# plt.show()



