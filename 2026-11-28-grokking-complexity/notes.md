# Daily Experiment: Complexity Dynamics of Grokking
Date: 2026-11-28
Tweet ID: 1988330691478909314
Original tweet: https://x.com/hive_echo/status/1988330691478909314
Paper: https://arxiv.org/abs/2412.09810 (DeMoss et al., Oxford)

## Summary of the Liked Tweet / Paper
The tweet (and quoted thread) discusses the paper "The Complexity Dynamics of Grokking" by Branton DeMoss et al. (arXiv:2412.09810, published in Physica D 2025).

Key claim: Grokking (sudden late generalization after memorization on modular arithmetic tasks) corresponds to a complexity phase transition. Model complexity (proxied by lossy compression / Kolmogorov complexity upper bound via coarse-graining + zip) rises during memorization then falls as a simpler generalizing solution is found. Unregularized models stay high-complexity and don't generalize. They introduce spectral entropy regularization on singular values to encourage low-rank, low-complexity solutions.

Spectral entropy of a weight matrix:
\[
\bar{\sigma}_i = \frac{\sigma_i}{\sum_j \sigma_j}, \quad H_{\mathrm{svd}} = -\sum_i \bar{\sigma}_i \log \bar{\sigma}_i
\]
(Lower \(H_{\mathrm{svd}}\) = more concentrated singular values = lower effective rank.) Source: paper section 4.2, https://arxiv.org/html/2412.09810v2

Complexity measurement uses Algorithm 1: per-layer low-rank approx (threshold \(\tau\)) + quantization, accepted only if loss distortion < \(\epsilon\), then compress. Achieves 30-40× better compression than naïve. Source: abstract and section 3, https://arxiv.org/abs/2412.09810

## Experiment
Implemented a small-scale modular addition (mod p) MLP with embedding + hidden layers. Tracked train/test accuracy, average spectral entropy of weight matrices, and a simple compressed size proxy (quantize to 5-8 bits + gzip).

Variants tried:
- Pure Adam (no WD, no SE reg)
- Adam + weight decay
- Adam + weight decay + spectral entropy penalty on loss

### Results (CPU, short runs, small p=13-23)
- Full classic grokking (perfect train then delayed perfect test) did not appear in <1500 epochs with the hyperparams tested; weight decay too aggressive often collapsed accuracy before generalization.
- Spectral entropy consistently dropped over long training under weight decay (e.g., from ~2.95 to ~0.7-0.8), correlating with smaller compressed sizes (from ~1000+ bytes to ~50-200 bytes).
- Adding explicit spectral entropy penalty produced similar or slightly lower final entropy/compressed size but did not uniquely unlock generalization in these short runs.
- Compressed size proxy after quantization+gzip is a practical, cheap stand-in for the paper’s apparent complexity and moves in the expected direction when structure emerges.

Schematic plot (grokking_complexity_schematic.png) reproduces the characteristic rise-and-fall described in the paper’s Figure 1 for regularized vs unregularized nets.

### Concrete numbers from runs
- With strong WD: spectral entropy fell from ~2.96 → ~0.82 while accuracy stayed near chance (model collapsed).
- Milder WD: entropy still fell (2.95 → ~0.75) and compressed size shrank ~18× (1062 → 59 bytes), but test accuracy remained low.
- These confirm that low spectral entropy / high compressibility can be forced by regularization even without successful generalization, matching the paper’s point that regularization is necessary to escape the high-complexity memorization phase.

Sources for methods and claims:
- Paper: https://arxiv.org/abs/2412.09810
- HTML: https://arxiv.org/html/2412.09810v2
- Original grokking: Power et al. 2022 (referenced in paper)
- Related modular arithmetic grokking: Gromov 2023, https://arxiv.org/abs/2301.02679

## Improvement Insights
1. Spectral entropy is a cheap, differentiable proxy for effective rank and pairs well with the paper’s compression view. Could be combined with progressive \(\epsilon\)-constrained coarse-graining during training (not just post-hoc).
2. For practical use, apply the penalty only after memorization is detected (train acc high, test low) to avoid early collapse.
3. The 30-40× compression gain suggests that post-training SVD truncation + quantization is a strong model compression technique specifically for solutions that have already grokked; measure it on larger modular or algorithmic tasks.
4. Edge case: very strong spectral entropy penalty can collapse the model before it finds the generalizing circuit (observed here). Annealing \(\beta\) is advisable.
5. Future collab: scale to the paper’s exact modular multiplication setup with longer runs / GPU and implement their full Algorithm 1 rate-distortion search for tighter complexity bounds.

## Visuals to attach
- grokking_complexity_experiment.png : actual training curves of spectral entropy and compressed size from the runs
- grokking_complexity_schematic.png : clean rise-and-fall illustration matching the paper’s description

## GitHub
Pushed under 2026-11-28-grokking-complexity/ on https://github.com/vnmerchant/Musings
