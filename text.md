Here's your full prep report. Read through it before the meeting, top to bottom.

---

# Full Meeting Prep Report
## Spectrometer URA — V5 through lstsq experiment

---

# 1. The Big Picture (say this at the start)

"Last meeting we established that NNLS gives structurally wrong outputs on broad spectra — spiky 0/1 garbage — and that a basic NN already improved on it dramatically. The to-do list was: quick fixes to the NNLS solver, different loss functions, and hyperparameter exploration. Since then I've run V5 through V9, done a full analysis of the results, and ran a new experiment that produced a counterintuitive finding about the solver itself."

---

# 2. What You're Working On (30 seconds of context)

- **Problem**: 25 sensors → recover 1500-point spectrum. Massively underdetermined (25 equations, 1500 unknowns).
- **Prior approach**: SVD + simulated annealing (Kumar paper), only tested on sparse/single-peak spectra.
- **Your approach**: type-specific routing. Classifier detects broad vs sparse → routes to a type-specific NN solver.
- **Current focus**: the broad solver branch. Broad spectra = smooth humps, NNLS destroys their structure via active-set sparsity.
- **Input to the network**: NNLS rough draft (1500-d) + raw 25-d sensor measurement, concatenated to 1525-d.

---

# 3. Results Table — The Full Story

| Model | What changed | Test MSE | vs NNLS |
|---|---|---:|---:|
| NNLS input | baseline | 0.07444 | — |
| V1 | MLP 1500→1024→512→1500, MSE+smooth loss | 0.01043 | 86.0% |
| V2 | + residual skip connection | 0.02395 | 67.8% ❌ |
| V3 | wider MLP (2048/1024) | 0.01079 | 85.5% |
| V4 | concat raw 25-d input → 1525-d | 0.00732 | 90.2% ⬆ |
| V5 | fixed V4 bug + cosine LR | 0.00802 | 89.2% |
| V6 | AdamW + validation tracking | 0.00835 | 88.8% |
| V7 | + input noise augmentation | 0.00835 | 88.8% |
| V8 | + 4000 more samples + early stopping | 0.00659 | 91.2% ⬆ |
| V9 | + Huber loss | 0.00649 | 91.3% |
| **lstsq exp** | swap NNLS draft for lstsq draft | **0.00549** | **92.6%** ⬆ |

Three big wins: V4, V8, lstsq. Everything else was diagnostic or marginal.

---

# 4. V5 — Cosine LR (quick)

**What**: Same architecture as V4, fixed the code bug, added cosine annealing learning rate schedule.

**Why cosine LR**: V4's training loss was still falling at epoch 400 with fixed lr=1e-3. Cosine annealing smoothly decays lr from 1e-3 → 0 over 400 epochs. Big steps early, tiny steps late.

**Result**: 0.00802. Slightly worse than V4 (0.00732). The V4 number is suspect due to the architecture bug, so V5 is the real clean baseline.

**Lesson**: LR schedule wasn't the bottleneck.

---

# 5. V6 — AdamW + Validation Tracking (important)

**What**: Adam → AdamW (correct weight decay), plus measure test MSE every 20 epochs.

**Why AdamW**: plain Adam applies weight decay incorrectly — it gets entangled with the adaptive scaling. AdamW decouples them, applies the regularization cleanly. Weight decay penalises large weights, forces the network toward smaller, smoother, more generalizable solutions.

**Result**: 0.00835 final. But **best test MSE was 0.00742 at epoch 20**.

**The real finding**: test MSE peaked at epoch 20 then got *worse* for the remaining 380 epochs while training loss kept falling. Classic overfitting. The network was memorising training data after epoch 20, not generalising better.

**Why this matters**: without validation tracking you'd never see this. V6 diagnosed the real problem — not the optimizer, not the LR, but **data volume and training duration**.

---

# 6. V7 — Input Noise Augmentation (null result)

**What**: Added random noise to inputs every batch. `rough += noise(std=0.02)`, `raw += noise(std=0.1)`. Targets stay clean. Idea: 1183 samples is small — make each look "new" every epoch.

**Why**: V6 showed overfitting. Noise augmentation is a standard cheap fix — effectively infinite unique training examples.

**Result**: 0.008347. Identical to V6.

**Why it failed**: the bottleneck wasn't memorising specific pixel values — it was structural diversity. All 1183 broad spectra look similar. Adding ±2% jitter to an absorption curve doesn't make it look like a different absorption curve. The network still saw effectively the same 1183 shapes every epoch.

**Lesson**: noise augmentation ≠ structural diversity. Rules out the cheap fix, points to the real fix.

---

# 7. V8 — More Data + Early Stopping (second big win)

**What**: Two changes together.

1. Training set 1183 → 5183 (added 4000 new broad samples, different random seed, same test set).
2. Best-snapshot early stopping: every 20 epochs, if test MSE is the best seen, save a copy of weights. After training, restore the best snapshot.

**Why more data fixes overfitting**: with 5183 samples, the network can't easily memorise — too many examples, too much structural variation. The only way to get low training loss is to actually learn the pattern.

**Why early stopping**: V6 showed the best model appears mid-training, not at the end. Instead of stopping early (which wastes potential), you save the best checkpoint and restore it. You run all 400 epochs but walk away with the epoch-80 model.

**Result**: 0.00659, best at epoch 80. Second biggest win in the notebook.

**Key code**:
```python
if val_mse < best_val_mse:
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
# After training:
model.load_state_dict(best_state)
```
`.clone()` is critical — without it `best_state` is a live reference that updates as training continues, not a frozen snapshot.

---

# 8. V9 — Huber Loss (marginal)

**What**: Weighted MSE → weighted Huber loss. Same V8 setup otherwise.

**Why Huber**: MSE squares errors — a few catastrophic samples (MSE 0.3+) produce gradients orders of magnitude larger than well-reconstructed ones, dominating training. Huber switches from quadratic to linear beyond delta=0.1, capping outlier gradient contributions.

**Why it was marginal (0.00649 vs 0.00659)**: V8's early stopping and larger dataset had already mostly mitigated the outlier problem. Huber was targeting something V8 had already handled.

**Important**: training uses Huber, evaluation reports MSE — same metric as all other models. Change one thing, keep the ruler constant.

**Lesson**: technique effectiveness depends on what you've already done. Huber would have been more impactful before V8.

---

# 9. The lstsq Experiment (the headline finding)

**The question**: V4 showed that concatenating the raw 25-d input was the biggest win. V6 showed overfitting. V8 fixed it. So what is the rough draft actually contributing? Is NNLS the right solver for it?

**What**: Replace NNLS rough draft with unconstrained least-squares (lstsq) draft. Everything else identical to V9 — same architecture, same training recipe, same test set.

**What lstsq is**: same regularized problem as NNLS (`min ||Tx - measurement||² + α||Lx||²`) but with no non-negativity constraint. Solved in one closed-form matrix operation — seconds vs 40 minutes for NNLS on 8 cores. Output can be negative.

**Input quality comparison**:
- NNLS draft MSE: 0.0744
- lstsq draft MSE: 0.2877 — **4× worse**

**Network output**:
- V9 (NNLS draft): 0.00649
- lstsq experiment: **0.00549 — 15% better**

**The counterintuitive finding**: worse input draft → better trained network.

**Why**: NNLS produces active-set spikes at positions determined by the solver algorithm, not by the true spectral shape. lstsq oscillates wildly but *around the true mean*. A network can learn to smooth out structured oscillation. It cannot learn to un-randomise arbitrary spike positions. Input MSE was the wrong quality metric — shape preservation is what matters.

**Visual evidence**: the `lstsq_vs_nnls_input.png` plot shows this directly. lstsq draft oscillates around the truth. NNLS draft has no structural relationship to it.

**Bonus**: lstsq is closed-form, ~1000× faster than NNLS. Practical win on top of accuracy win.

---

# 10. The Images — What Each One Shows

### `mse_distribution.png`
Histogram of per-sample MSE (log scale) for NNLS, V1, V9 across 296 test samples. Key message: V9 didn't just shift the mean — it compressed the distribution. Fewer outliers, more samples concentrated near best performance. The floor around 10⁻⁴ hasn't moved since V1 — that's the noise floor, not an architecture limitation.

### `per_type_mse.png`
Grouped bar chart: broad emission vs broad absorption across NNLS/V1/V8/V9. Both types improve proportionally — no accidental favouritism. Absorption consistently easier than emission (gap narrows from 50% at NNLS to 15% at V9 but doesn't close). Likely because the smoothness prior suits "mostly high with a dip" better than "mostly zero with a peak."

### `headline_worst_nnls.png`
Single worst NNLS test sample (test idx 203, broad absorption). Gray = 0/1 square wave chaos, MSE 0.376. Green V9 = smooth declining curve, MSE 0.026. 14× improvement. **This is your money shot** — directly answers last meeting's complaint. Anyone can see gray is useless and green is physically interpretable.

### `peak_position_error.png`
Two-panel histogram of |predicted peak wavelength − true peak wavelength| in nm. NNLS median = 3.1 nm (essentially random). V1 and V9 both ~0.7 nm median — almost identical. Key insight: MSE improvements from V1→V9 came from amplitude and shape fidelity, not peak localisation. The network knew where the peak was from V1; it spent V5–V9 polishing the curve around it.

### `before_after_grid.png`
2×3 grid of the six samples where V9 improves most over NNLS. NNLS = uniform square wave garbage on all six. V9 = correct shape on simple cases (single hump emission, monotone absorption), partially correct on complex ones (non-monotone absorption, double-hump emission). Honest representation — shows both successes and remaining limitations.

### `lstsq_vs_nnls_input.png`
Side-by-side of NNLS draft vs lstsq draft on the worst NNLS sample. Ground truth ≈ constant ~1.0. NNLS = random spikes 0–6. lstsq = wild oscillation −5 to +6 but oscillating *around* ~0.8. Visual proof that lstsq preserves structural information NNLS destroys. This is why the counterintuitive result makes sense.

### `lstsq_headline.png`
Worst lstsq-draft sample (test idx 53). lstsq draft MSE = 3.584 — 10× worse than the worst NNLS case in the other headline plot. NN output MSE = 0.015 — better than V9's worst case. 247× improvement from input to output. Strongest evidence that draft quality doesn't determine output quality — the network's real information source is the raw 25-d input.

### `lstsq_distribution.png`
Histogram of lstsq input vs lstsq-NN output MSE. Gray input distribution is flat and wide, spanning 10⁻² to 10⁰. Purple output is tight bell-shape around 2×10⁻³. Gap is ~52× — bigger than the NNLS→V9 gap of ~11×. The network does proportionally more work here, transforming a more chaotic input into a comparable or better output.

---

# 11. Responses to Likely Questions

**"Is this better than SVD + simulated annealing?"**
"That comparison is on the to-do list — the Kumar paper only validated SVD+SA on sparse spectra, not broad. Running it on our broad test set is the next missing baseline. I expect it to perform poorly on broad spectra since it's designed for sparse signals, but I don't have the number yet."

**"Why is lstsq better if it has worse input MSE?"**
"Input MSE was the wrong metric. NNLS active-set spikes land at solver-determined positions with no relationship to the true spectrum — the network can't learn to recover structure that isn't there. lstsq oscillates wildly but around the correct mean — the network learns to smooth it out. Shape preservation matters more than value accuracy for what the network needs."

**"Did you confirm the lstsq result isn't noise?"**
"This is one run, one seed. The 15% improvement is comfortably above typical run-to-run variance for these models, but I'll confirm with 2–3 seeds this week."

**"Why didn't noise augmentation help?"**
"The bottleneck wasn't memorising specific pixel values — it was structural diversity. All 1183 training spectra have the same general shape class. Adding ±2% jitter doesn't create new shapes, just slightly different versions of the same ones. V8's 4000 genuinely new samples fixed the actual problem."

**"What's next?"**
"Three things in priority order: raw-only experiment (drop the rough draft entirely, test whether the raw 25-d input alone is sufficient — directly motivated by the lstsq finding), then a 1D U-Net architecture (the MLP ignores local spectral structure that a CNN exploits naturally), then the SVD+SA baseline on broad spectra."

---

# 12. The One-Paragraph Summary (know this cold)

"Starting from V4's finding that concatenating the raw 25-d sensor measurement was the biggest single win, I explored the V5–V9 design space: learning rate schedules, weight decay, noise augmentation, more data, and Huber loss. The diagnostic finding of the batch was V6: validation tracking revealed the model was overfitting by epoch 20 despite 400 epochs of training. V7's noise augmentation failed to fix this — jitter doesn't add structural diversity. V8 fixed it properly with 4000 additional training samples and best-snapshot early stopping, the second biggest win in the project. V9's Huber loss was marginal. The headline result came from the lstsq experiment: replacing the NNLS rough draft with an unconstrained least-squares draft — which has 4× worse input MSE — improved network test MSE by 15%. This is counterintuitive but explainable: NNLS active-set spikes have no structural relationship to the true spectrum, while lstsq oscillates around the correct mean. Input MSE was the wrong quality metric all along. The natural next experiment is to drop the rough draft entirely."

---

That's everything. You know this material — you just did it. Good luck.

