# SINN Project — Complete Technical Walkthrough

## The Big Picture

The spectrometer has 25 sensors that capture light across 1500 wavelength points (1550-1565 nm). The fundamental challenge is reconstruction: given 25 measurements, recover the original 1500-point spectrum. This is a massively underdetermined problem — 25 equations, 1500 unknowns — with infinitely many valid solutions.

The existing approach from the Performance Limits paper (Kumar et al.) uses SVD with truncated singular values followed by simulated annealing. However, they only tested this on single laser lines (one spike) or pairs of laser lines (two spikes). The question our project addresses is: can we do better on complex, real-world spectra using a solver-informed neural network with type-specific routing?

---

## Step 1: Understanding the Data

We work with three arrays:
- **X_train_np** (3000, 25): Raw sensor measurements. These are what the spectrometer actually captures — 25 numbers per spectrum.
- **y_train_np** (3000, 1500): Ground truth spectra. These are the true spectra we're trying to reconstruct.
- **X_train_rough** (3000, 1500): Reg NNLS rough drafts. The solver's best attempt at reconstructing the spectrum from the 25 measurements.
- **labels_list** (3000,): String labels identifying each spectrum's type (generate_sparse_spectrum, generate_broad_emission, etc.)

The data was synthetically generated across four types: sparse emission, broad emission, narrow absorption, and broad absorption. Each was passed through the calibration matrix with added noise at varying SNR levels (30, 40, 50 dB).

---

## Step 2: Why Reg NNLS Output Quality Varies

Reg NNLS solves: minimize ||I - Tx||² + α||Lx||² subject to x ≥ 0

Three forces are at play:
1. **Least squares (||I - Tx||²)**: Find any spectrum that matches the sensor readings
2. **Regularization (α||Lx||²)**: Prefer smooth solutions (L is a first-order difference matrix)
3. **Non-negativity (x ≥ 0)**: All values must be positive (physically, light intensity can't be negative)

The non-negativity constraint has a known mathematical property: it biases solutions toward sparse, spiky outputs. When you force all values to be non-negative and minimize squared error, the optimizer tends to put energy in a few big spikes rather than spreading it smoothly. This is because it's mathematically easier to satisfy the equations that way.

This bias creates a type-dependent quality:
- **Sparse spectra**: Bias matches reality → decent NNLS output
- **Broad spectra with good sensor coverage**: Enough information to partially overcome bias → okay output
- **Broad spectra with poor sensor coverage**: Not enough information, bias wins → spiky garbage

Other factors affecting quality:
- **Sensor coverage**: The 25 sensors aren't evenly sensitive across all wavelengths. Peaks in well-covered regions reconstruct better.
- **SNR level**: 30 dB has more noise → worse reconstruction than 50 dB
- **Spectrum complexity**: A single broad hump is easier than multiple overlapping peaks

---

## Step 3: Filtering Data for Broad Spectra Only

Since we're building a type-specific solver for broad spectra, we subset the data:

```python
broad_types = ["generate_broad_emission", "generate_broad_absorption"]
broad_mask = np.array([label in broad_types for label in labels_list])
X_broad_rough = X_train_rough[broad_mask]  # ~1479 samples
y_broad_true = y_train_np[broad_mask]
```

This gives us roughly half the dataset (2 broad types out of 4 total).

---

## Step 4: Train/Test Split

We split the broad subset 80/20:

```python
split = int(0.8 * len(X_broad_rough))
X_train = X_broad_rough[:split]      # 1183 samples
X_test = X_broad_rough[split:]       # 296 samples
y_train = y_broad_true[:split]
y_test = y_broad_true[split:]
```

Why split? Even though the data is synthetic, we need to measure whether the network generalizes — can it refine a broad spectrum it's never seen before, or is it just memorizing training examples?

---

## Step 5: PyTorch Dataset and DataLoader

**Dataset**: An adapter that teaches PyTorch how to access our data. It converts NumPy arrays to PyTorch tensors and provides a way to access individual (input, target) pairs by index.

**DataLoader**: Sits on top of the Dataset and handles two things:
- Batching: Groups samples (batch_size=32) so the network processes 32 at a time rather than one by one. Faster training and more stable gradients.
- Shuffling: Randomizes order each epoch so the network doesn't learn patterns based on data ordering.

```python
class BroadSpectrumDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()  # .float() for 32-bit precision
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
```

---

## Step 6: The Broad Solver Network Architecture

### V1 — Baseline (Best Performer)
```
Input (1500) → Linear(1024) → ReLU → Linear(512) → ReLU → Linear(1500) → Output
```

Design decisions:
- **Bottleneck shape (1500→1024→512→1500)**: The middle layers are smaller than input/output. This forces the network to compress the information before reconstructing. For broad spectra this is natural — broad curves are inherently simple, so a compressed representation should capture them well. Noise and artifacts can't survive the compression.
- **ReLU between hidden layers**: Standard nonlinearity that lets the network learn complex, non-linear mappings rather than just linear transformations.
- **No activation on output**: Spectrum values are continuous (roughly 0-1 range). ReLU would clip negatives, sigmoid would squash values. We want raw continuous output.

---

## Step 7: The Custom Loss Function

```python
def broad_loss(predicted, target, smoothness_lambda=0.1):
    mse = nn.MSELoss()(predicted, target)
    smoothness = torch.mean((predicted[:, 1:] - predicted[:, :-1]) ** 2)
    return mse + smoothness_lambda * smoothness
```

Two components:

**MSE (mean squared error)**: Measures average squared difference between prediction and ground truth. Standard reconstruction metric.

**Smoothness penalty**: This is the key innovation for the broad solver.
- `predicted[:, 1:]` = every point except the first
- `predicted[:, :-1]` = every point except the last
- Subtracting gives the difference between each pair of neighbors
- Squaring and averaging penalizes large jumps between adjacent wavelength points
- For broad spectra, the ground truth IS smooth, so this penalty nudges the network toward smooth outputs

**smoothness_lambda (0.1)**: Controls the balance. Too small = smoothness does nothing. Too large = network outputs a flat line (maximum smoothness but wrong shape).

This is the whole motivation for type-specific solvers. A sparse solver would need the OPPOSITE — an L1/sparsity penalty that encourages sharp peaks. One loss function cannot serve both types.

---

## Step 8: The Training Loop

```python
for epoch in range(num_epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        predicted = model(batch_X)           # Forward pass
        loss = broad_loss(predicted, batch_y) # Calculate loss
        optimizer.zero_grad()                 # Clear old gradients
        loss.backward()                       # Compute new gradients
        optimizer.step()                      # Update weights
```

What happens each iteration:
1. **Forward pass**: Batch of 32 Reg NNLS outputs goes through the network, producing 32 predicted spectra
2. **Loss calculation**: Our custom loss measures how wrong the predictions are
3. **zero_grad()**: PyTorch accumulates gradients by default. Must reset before each batch.
4. **backward()**: Computes gradients — how should each weight change to reduce the loss?
5. **step()**: Actually updates the weights using those gradients. Adam optimizer adjusts learning rate per-parameter automatically.

### V1 Results
- Training loss: ~0.008 → ~0.002 over 100 epochs
- Test MSE: 0.074 (Reg NNLS) → 0.011 (network) = 85% improvement
- Average cases look good — red line tracks ground truth
- Worst cases: network gets shape right but underestimates amplitude

---

## Step 9: Analyzing the Results

### Why the average MSE looks great but worst cases are bad
The 85% improvement and 0.011 MSE are averages across all 296 test samples. Most samples are easy cases where Reg NNLS gives a decent rough draft. On those, the network does great, pulling MSE way down. The few hard cases (MSE of 0.08-0.17) barely move the average but look terrible visually. It's like getting 90% on most assignments but bombing one exam — GPA still looks good.

### What the worst cases showed
All four worst cases had the same pattern: the network gets the general shape right but consistently underestimates the amplitude. The red line sits below the black ground truth everywhere. This told us two things:
1. The smoothness penalty might be too strong — penalizing the network for going high
2. The network is compromising between easy and hard cases

---

## Step 10: Experiments to Improve

### Experiment: Reducing smoothness_lambda (0.1 → 0.01)
**Hypothesis**: Maybe λ=0.1 is too strong, pushing the network toward flat/low outputs.
**Result**: Training loss got lower (0.001 vs 0.002) but test MSE slightly WORSE (0.012 vs 0.011). Improvement dropped from 85% to 84%.
**What this means**: The network is overfitting — lower training loss but worse generalization. Classic sign. Also, the amplitude problem didn't improve. Reducing the penalty wasn't the fix.

### Experiment: Residual Connection (V2)
**Hypothesis**: Instead of learning the entire output spectrum from scratch, the network learns a CORRECTION to the Reg NNLS input. Output = input + network(input).

**The idea**: When Reg NNLS is already decent, the correction is small (easy). When it's terrible, the correction is large (hard but structured). The network can adapt per-sample.

**How it works mechanically**:
```python
def forward(self, x):
    residual = x                    # Save original input
    x = self.ReLU(self.layer1(x))   # Transform x through layers
    x = self.ReLU(self.layer2(x))
    x = self.layer3(x)
    return residual + x             # Original + learned correction
```

During training, the loss compares `residual + x` to `ground_truth`. Backpropagation adjusts weights so that `x ≈ ground_truth - residual`, i.e., the network learns the error/difference.

**Result**: 71% improvement, MSE = 0.023. SIGNIFICANTLY WORSE.

**Why it failed**: The Reg NNLS output for broad spectra is full of spikes. Adding it back to the network's output injects all that noise. The network can't fully cancel out every spike. You could see this visually — the V2 output was spiky, not smooth.

**Key insight**: Residual connections work when the input is already close to the target (like image denoising where the input is a recognizable but noisy image). When the input is fundamentally different from the target (spiky vs. smooth), the residual hurts.

**Implication for sparse solver**: Residual connections might actually HELP for sparse spectra, since Reg NNLS naturally produces spiky outputs that match sparse targets. Worth testing later.

### Experiment: Wider Network (V3)
**Hypothesis**: More network capacity might help handle both easy and hard cases better.

Architecture: 1500→2048→1024→1500 (vs V1's 1500→1024→512→1500)

**Result**: 86% improvement, MSE = 0.010. Marginally better than V1.

**What was interesting**: The worst-case MSEs were more consistent (0.11, 0.11, 0.11, 0.13) vs V1's spread (0.08, 0.09, 0.10, 0.17). The worst single case improved from 0.17 to 0.13. So V3 is more reliable even if the average is similar.

**What this tells us**: More capacity helps a little but we're hitting a fundamental information limit. The problem isn't the network size — it's that some Reg NNLS inputs just don't contain enough information for any network to recover the true broad shape.

### Experiment: Dual Input (V4)
**Hypothesis**: Give the network both the Reg NNLS rough draft (1500 points) AND the raw sensor measurements (25 points). The raw measurements might provide additional information about "how trustworthy is this Reg NNLS output?"

Architecture: Input = concatenated [1500 + 25] = 1525 → 1024 → 512 → 1500

**Important detail — normalization**: The first run failed (loss barely moved, 0.33→0.29) because the raw sensor values were on a completely different scale than the 0-1 Reg NNLS output. The network's first layer was dominated by the 25 large values. After normalizing the raw measurements using the same mean/std from earlier, training worked normally.

**Important bug found**: The split index was calculated from the full 3000-sample dataset instead of the ~1479 broad subset. This meant X_raw_train had 1479 rows but X_train had 1183 rows — every batch paired mismatched samples. After fixing, retrained properly.

**Result**: 84% improvement, MSE = 0.012. No improvement over V1.

**Why it didn't help**: The Reg NNLS output already encodes the information from the raw measurements — it's computed from them using the calibration matrix. The 25 raw values are redundant. The network isn't getting new information.

---

## Summary of All Experiments

| Version | Architecture | Key Change | MSE | Improvement | Notes |
|---------|-------------|-----------|-----|-------------|-------|
| V1 | 1500→1024→512→1500 | Baseline | 0.011 | 85% | Best simplicity/performance ratio |
| V2 | Same + residual | output = input + correction | 0.023 | 71% | Failed — spiky input bleeds through |
| V3 | 1500→2048→1024→1500 | Wider layers | 0.010 | 86% | Marginal gain, more consistent worst cases |
| V4 | 1525→1024→512→1500 | Added raw measurements | 0.012 | 84% | Raw data is redundant with NNLS output |

All results fluctuate slightly between runs due to random weight initialization and DataLoader shuffling. Across multiple runs, V1 and V3 consistently land around 85-86%, V2 around 68-72%, V4 around 84%.

---

## Key Insights for the Paper

### 1. Reg NNLS has a built-in bias toward spiky solutions
The non-negativity constraint mathematically favors sparse outputs. This works well for sparse spectra but poorly for broad spectra. This is the fundamental motivation for type-specific solvers.

### 2. Residual connections don't work when the input is fundamentally wrong
Standard deep learning wisdom says residual connections always help. Our experiment shows they hurt when the solver output is structurally different from the target. This is type-dependent — residuals may help for sparse where NNLS output already matches.

### 3. The ~85% improvement may be near a physical ceiling
25 sensors → 1500 points is 60x underdetermined. Some information is physically lost. The SINN paper had 108 sensors. The Performance Limits paper tested on trivially simple spectra (single laser lines).

### 4. Visual solver quality ≠ usefulness as NN input
NNLS output looks terrible (spiky) but contains positional and energy information the network can use. SVD output looks smoother but may have destroyed the useful information. "Messy but informative" beats "clean but empty."

### 5. The paper comparison we need
The Performance Limits paper uses SVD + simulated annealing but only tested on single laser lines. Our SINN pipeline tackles complex spectra — a harder problem. To complete the paper, we need to run their approach on our complex synthetic data and show our pipeline beats it.

---

## What's Next

1. Build the sparse solver (same pattern, different loss — L1 penalty instead of smoothness)
2. Implement SVD + simulated annealing baseline for comparison on complex spectra
3. Full pipeline evaluation: classifier → type-specific solver → type-specific NN
4. Consider type-specific solvers: potentially SVD for broad input, NNLS for sparse input
5. Quantitative NNLS vs SVD comparison (MSE, not just visual)
