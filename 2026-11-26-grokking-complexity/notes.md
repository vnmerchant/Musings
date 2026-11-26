# Daily Experiment: Complexity Dynamics of Grokking (2026-11-26)

**Original liked tweet:** https://x.com/hive_echo/status/1988330691478909314  
**Author:** @hive_echo  
**Summary:** Compression (via quantization/SVD + zip) is a practical signal of Kolmogorov complexity; during grokking, model complexity rises during memorization then falls as a simpler general rule is discovered. Spectral entropy of weight matrices measures this “spread vs concentration.”

## Core Paper

- **Title:** The Complexity Dynamics of Grokking  
- **Authors:** Branton DeMoss, Silvia Sapora, Jakob Foerster, Nick Hawes, Ingmar Posner (Oxford)  
- **arXiv:** https://arxiv.org/abs/2412.09810 (v2, 21 Aug 2025)  
- **Journal:** Physica D: Nonlinear Phenomena 482 (2025) 134859 — https://doi.org/10.1016/j.physd.2025.134859  
- **Code:** https://github.com/brantondemoss/GrokkingComplexity  
- **Author blog:** https://brantondemoss.com/research/grokking  

### Key Claims (with sources)

1. Grokking is a **complexity phase transition**: complexity rises while the network memorizes, then falls as it discovers a simpler generalizing solution. Unregularized nets stay in the high-complexity memorization phase.  
   Source: Abstract + Fig. 1 of https://arxiv.org/abs/2412.09810

2. Complexity is measured by **lossy compression** of the weights (coarse-graining via quantization + low-rank SVD, then off-the-shelf compressor such as bzip2). This gives a practical upper bound on Kolmogorov complexity. The method achieves 30–40× better compression ratios than naïve approaches.  
   Source: Abstract + Section on rate-distortion / Algorithm 1 of https://arxiv.org/abs/2412.09810

3. **Spectral entropy regularization** \( H_{\mathrm{svd}}(W) = -\sum_i \bar{\sigma}_i \log \bar{\sigma}_i \) (normalized singular values) encourages low-rank / low-intrinsic-dimension solutions and produces the most compressible networks among the regularizers studied.  
   Source: Section 4.2 / Eq. (16) of https://arxiv.org/abs/2412.09810

4. Link to generalization via Minimum Description Length: best model minimizes \( H(D|M) + C(M) \).  
   Source: Introduction of https://arxiv.org/abs/2412.09810 and https://brantondemoss.com/research/grokking

## Experiments Run Today

Lightweight PyTorch prototypes (CPU, ~minutes) that implement the core ideas of spectral entropy regularization and compression-based complexity proxies. Full-scale grokking (thousands of steps on transformers) is outside the single-run budget; the official code is the reference.

### 1. Differentiable Spectral Entropy Penalty

Script: `se_experiment.py`

- Task: modular addition \( a + b \mod 19 \), TinyMLP with embedding.
- Compare \(\beta = 0\) vs \(\beta = 0.15\).
- Track accuracy, mean normalized spectral entropy of weight matrices, and compressed size after 8-bit quantization + 50 % SVD truncation + bzip2.

**Results (final epoch):**

| Setting | Test Acc | Mean SE | Compressed Size |
|---------|----------|---------|-----------------|
| No reg  | 0.983    | 0.9659  | 4645 B         |
| SE reg  | 0.988    | 0.9633  | 4592 B         |

- Spectral entropy reduced by 0.0026  
- Compressed size reduced by ~1.1 %  
- Accuracy preserved (slightly higher)  

Source of measurement: local run of `se_experiment.py`.

### 2. Singular Value Spectra & Truncation Robustness

Script: `sv_spectra_demo.py`

- Same idea, stronger \(\beta = 0.3\).
- Plot of normalized singular value spectra of the largest linear layer.
- Accuracy after keeping only a fraction of singular values (SVD truncation = coarse-graining).

**Observations:**

- SE-reg spectrum is slightly more concentrated.
- Under moderate-to-aggressive truncation the SE-regularized model is at least as robust (sometimes better) than the unregularized model.

## Improvement Insights

1. **Make the complexity proxy online and cheap.**  
   Full Bayes-optimized coarse-graining (as in the official code) is expensive. Tracking spectral entropy / effective rank of weight matrices (or of activations) is a cheap differentiable surrogate that already correlates with compressibility.  
   Related later work on spectral entropy of *representations*: https://arxiv.org/abs/2604.13123

2. **Combine with progressive quantization / low-rank adapters.**  
   During the high-complexity phase the network is effectively memorizing; a temporary increase in capacity (or delayed application of strong SE penalty) may help it escape the memorization basin before the complexity collapse. This is consistent with the “noisy gradients + weight decay + SE” recipe in the paper.

3. **Use spectral entropy of activations as a leading indicator.**  
   Recent work shows normalized spectral entropy of the representation covariance collapses *before* test accuracy jumps (lead time ~1k steps on modular tasks). Monitoring both weight-space and activation-space entropy could give earlier warning of the phase transition.  
   Source: https://arxiv.org/abs/2604.13123

4. **Scale the experiment.**  
   The official JAX code (https://github.com/brantondemoss/GrokkingComplexity) trains transformers on modular arithmetic for tens of thousands of steps. A useful next step is to instrument that codebase with a lightweight SE monitor and compare against the full bzip2 complexity curve.

## Visual Evidence for Sharing

Attach any of the generated PNGs:
- differentiable_se_regularization.png
- singular_value_spectra.png
- truncation_robustness.png

All files live in this folder.
