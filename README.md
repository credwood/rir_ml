## Discrete-Time Acoustic Metric Formulas

**Reference:** RIR is a 1D Numpy array `h[n]`, sampled at `fs` Hz.

---

### 1. RT60 (Reverberation Time)

Based on the Energy Decay Curve (EDC) computed from the Schroeder integral:

**Step 1: Compute the EDC (Schroeder integration)**

```
EDC[n] = ∑ (from k = n to N−1) of h[k]^2
EDC_dB[n] = 10 * log10(EDC[n] / max(EDC) + ε)
```

Where `ε` (e.g., 1e-12) avoids log(0).

**Step 2: Fit a line to part of the EDC (in dB)**  
Use the T30 method: fit a linear regression over −5 dB to −35 dB. Then:

```
RT60 = -60 / a
```

Where `a` is the slope of the line fit to the EDC in that range.

---

### 2. EDT (Early Decay Time)

Same as RT60, but the slope is fitted over the first 10 dB of decay (0 to −10 dB):

```
EDT = -60 / a
```

---

### 3. C50 (Clarity Index)

```
n_50 = floor(0.05 * fs)  # number of samples in first 50 ms

C50 = 10 * log10(
          sum(h[n]^2 for n = 0 to n_50) /
          sum(h[n]^2 for n = n_50+1 to N−1)
      )
```

---

### 4. D50 (Definition Index)

```
D50 = sum(h[n]^2 for n = 0 to n_50) / sum(h[n]^2 for n = 0 to N−1)
```

(Same `n_50` as above.)

---

## Deep Learnign Model Design

### Model Architecture: 1D CNN for RIR-Based Acoustic Metric Prediction

This model processes raw Room Impulse Responses (RIRs) through a series of 1D convolutions to extract time-domain features and predict key acoustic metrics (RT60, EDT, C50, D50).

| **Layer**               | **Kernel Size** | **Stride** | **Padding** | **Output Channels** | **Role / Purpose**                                                                 |
|-------------------------|-----------------|------------|-------------|----------------------|------------------------------------------------------------------------------------|
| **Conv1d #1**           | 5               | 1          | 2           | 32                   | Detect very **local features** like the direct sound impulse or sharp transients. Maintains full time resolution. |
| **Conv1d #2**           | 15              | 2          | 7           | 64                   | Capture **early reflection patterns** (clusters of echoes). Slightly reduces time resolution to summarize short segments. |
| **Conv1d #3**           | 63              | 2          | 31          | 128                  | Model **global decay structure** of the reverberant tail. Large kernel and stride increase receptive field and reduce temporal size. |
| **AdaptiveAvgPool1d**   | —               | —          | —           | —                    | Collapses the time dimension by computing the **mean over time**, regardless of input length. Produces a single vector per channel. |
| **Linear → ReLU**       | —               | —          | —           | 64                   | Transforms pooled features into a **latent representation**, with nonlinearity for expressive power. |
| **Final Linear Layer**  | —               | —          | —           | 4                    | Outputs the predicted acoustic metrics: **RT60, EDT, C50, D50**. |

#### Tensor Dimensions Through the 1D CNN

Assuming an input RIR of shape `[B, T]` (e.g., `T = 8192`), the model processes it through the following transformations:

| **Layer**               | **Input Shape**      | **Output Shape**     | **Description**                                                   |
|-------------------------|----------------------|-----------------------|-------------------------------------------------------------------|
| **Raw Input**           | `[B, 1, T]`             | —                     | Raw 1D RIR signal                                                 |                               |
| **Conv1d #1**           | `[B, 1, 8192]`       | `[B, 32, 8192]`       | Local features, no downsampling                                  |
| **Conv1d #2**           | `[B, 32, 8192]`      | `[B, 64, 4096]`       | Mid-range features, downsample by 2                              |
| **Conv1d #3**           | `[B, 64, 4096]`      | `[B, 128, 2048]`      | Long-range decay modeling, downsample by 2 again                 |
| **AdaptiveAvgPool1d(1)**| `[B, 128, 2048]`     | `[B, 128, 1]`         | Global average pooling across time                               |
| **Flatten**             | `[B, 128, 1]`        | `[B, 128]`            | Flatten for fully connected layer                                |
| **Linear → ReLU**       | `[B, 128]`           | `[B, 64]`             | Dense feature projection with nonlinearity                       |
| **Final Linear Layer**  | `[B, 64]`            | `[B, 4]`              | Output predicted metrics: RT60, EDT, C50, D50                    |

---

### CNN Design Philosophy

- Start with **fine-grained filters** to capture sharp transients.
- Gradually increase kernel size and stride to model **longer-range temporal decay**.
- Use **adaptive pooling** to summarize time series of variable lengths.
- End with a **regression head** to output multiple continuous-valued metrics.

```
Output length = (Input_Length + 2*padding - kernel)/stride
```
