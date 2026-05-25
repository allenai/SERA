# DualPipeV Nopack Training: Findings & Comparison with Axolotl

## Summary

DualPipeV pipeline-parallel training with no-packing (one sample per sequence) now runs stably with the unpad/repad approach. The loss trajectory is close to but consistently higher than the Axolotl DeepSpeed baseline (+0.08 by step 20). The remaining discrepancy is attributed to the loss function: DPV uses `fused_linear_cross_entropy` (bitsandbytes) while Axolotl uses `cut_cross_entropy` (CCE).

## Timeline of fixes

1. **Per-layer padding masking** (commit 8483de7): Zeroed padding at every layer. Fixed NaN at step 3 but NaN reappeared at step 16 — masking creates degenerate attention patterns in backward pass.

2. **NaN gradient clamping** (commit 35f3d46): Added `nan_to_num_(0.0)` before optimizer step. Got through 20 steps, but this treats symptoms not cause.

3. **Unpad/repad** (commit 8722f9b): Strip padding before computation, restore only for P2P communication. Model never sees padding. Eliminates NaN entirely — 20 steps with zero NaN, no clamping needed.

## Data alignment verification

- Axolotl Arrow dataset and DPV converted PT file are byte-identical (all 16,000 samples verified)
- Same dataset cache hash: `c011032257370c005ff5924f4d6d2666`
- Same shuffle seed (42) produces identical permutation
- Same 32 samples per step (verified for first 5 steps)

## 20-step comparison: DPV nopack vs Axolotl DeepSpeed

Experiment: `01KSFDTPQQMM2KF6S0D932WQKK`
Baseline: `axolotl_8gpu_deepspeed_200step` (trainer_state.json)

```
Step  DPV loss  DPV gn    Axolotl loss  Axolotl gn   Delta loss
   1    0.5817   14.11       0.5579       13.99        +0.024
   2    0.5563   13.89       0.5793       14.22        -0.023
   3    0.5907   14.67       0.5503       13.05        +0.040
   4    0.5382   12.91       0.5522       13.59        -0.014
   5    0.5644   13.40       0.5576       13.18        +0.007
   6    0.5830   13.18       0.5510       12.29        +0.032
   7    0.5718   13.74       0.5318       11.30        +0.040
   8    0.5267   12.08       0.4822        6.91        +0.044
   9    0.5370   12.00       0.5026        6.35        +0.034
  10    0.5477   11.59       0.4695        6.06        +0.078
  11    0.5460   11.10       0.4449        2.08        +0.101
  12    0.5046    8.22       0.4246        1.87        +0.080
  13    0.4991    8.12       0.4160        1.59        +0.083
  14    0.4874    6.38       0.4071        1.21        +0.080
  15    0.4677    6.02       0.3701        1.07        +0.098
  16    0.4684    4.83       0.3876        1.13        +0.081
  17    0.4409    4.02       0.3742        0.99        +0.067
  18    0.4367    3.37       0.3624        0.86        +0.074
  19    0.4391    2.95       0.3533        0.66        +0.086
  20    0.4354    2.45       0.3580        0.74        +0.077
```

## Key observations

1. **Grad norms are comparable** at step 1 (14.1 vs 14.0), confirming gradient scaling is equivalent. No packing/unpadding artifacts.

2. **Loss gap opens at step 7-8**: Axolotl's grad_norm drops sharply (11.3 → 6.9) while DPV stays high (13.7 → 12.1). By step 11, Axolotl grad_norm is 2.1 vs DPV at 11.1. Axolotl learns faster.

3. **By step 20, gap is ~0.077** (DPV 0.435 vs Axolotl 0.358). Trend is steady, not diverging.

4. **Remaining variable: loss function**. Data, seed, batch composition, model, optimizer, LR schedule, and gradient clipping are all identical. The only difference is `fused_linear_cross_entropy` (bitsandbytes, chunked cuBLAS with fp32 logsumexp) vs `cut_cross_entropy` (CCE, Triton kernel with gradient filtering via `filter_eps`).

## Hypothesis: CCE gradient filtering

CCE has a `filter_eps` parameter (default: `'auto'`) that prunes gradient elements below a threshold. This gradient filtering:
- Reduces effective grad_norm (explains the faster drop in Axolotl)
- May improve training signal quality by suppressing noise
- Could account for the ~0.08 loss gap

## Next step

Switch DPV to use `cut_cross_entropy.linear_cross_entropy` instead of `fused_linear_cross_entropy` and re-run the 20-step comparison.
