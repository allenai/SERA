# DualPipeV Nopack Training: Findings & Comparison with Axolotl

## Summary

DualPipeV pipeline-parallel training with no-packing (one sample per sequence) now runs stably with the unpad/repad approach. The loss trajectory is consistently ~0.08 higher than the Axolotl DeepSpeed baseline by step 20. Switching to the same loss function (CCE) made no measurable difference (<0.002), ruling out loss computation as the cause. The discrepancy is in the forward/backward pass implementation differences between bitsandbytes custom kernels and standard PyTorch.

## Timeline of fixes

1. **Per-layer padding masking** (commit 8483de7): Zeroed padding at every layer. Fixed NaN at step 3 but NaN reappeared at step 16 — masking creates degenerate attention patterns in backward pass.

2. **NaN gradient clamping** (commit 35f3d46): Added `nan_to_num_(0.0)` before optimizer step. Got through 20 steps, but this treats symptoms not cause.

3. **Unpad/repad** (commit 8722f9b): Strip padding before computation, restore only for P2P communication. Model never sees padding. Eliminates NaN entirely — 20 steps with zero NaN, no clamping needed.

4. **CCE loss function** (commit 6beccdd): Switched from `fused_linear_cross_entropy` to `cut_cross_entropy.linear_cross_entropy` to match Axolotl. Result: <0.002 difference — loss function is NOT the cause.

## Data alignment verification

- Axolotl Arrow dataset and DPV converted PT file are byte-identical (all 16,000 samples verified)
- Same dataset cache hash: `c011032257370c005ff5924f4d6d2666`
- Same shuffle seed (42) produces identical permutation
- Same 32 samples per step (verified for first 5 steps)

## 20-step comparison: DPV (fused CE) vs DPV (CCE) vs Axolotl

```
Step  DPV+fused  DPV+CCE   Axolotl    fused-CCE  CCE-Axolotl
      loss       loss      loss       diff       diff
─────────────────────────────────────────────────────────────
   1  0.5817     0.5814    0.5579     -0.0003    +0.024
   2  0.5563     0.5560    0.5793     -0.0003    -0.023
   3  0.5907     0.5903    0.5503     -0.0004    +0.040
   4  0.5382     0.5384    0.5522     +0.0002    -0.014
   5  0.5644     0.5636    0.5576     -0.0008    +0.006
   6  0.5830     0.5829    0.5510     -0.0001    +0.032
   7  0.5718     0.5705    0.5318     -0.0013    +0.039
   8  0.5267     0.5263    0.4822     -0.0004    +0.044
   9  0.5370     0.5367    0.5026     -0.0003    +0.034
  10  0.5477     0.5471    0.4695     -0.0006    +0.078
  11  0.5460     0.5466    0.4449     +0.0006    +0.102
  12  0.5046     0.5041    0.4246     -0.0005    +0.080
  13  0.4991     0.4990    0.4160     -0.0001    +0.083
  14  0.4874     0.4866    0.4071     -0.0008    +0.080
  15  0.4677     0.4672    0.3701     -0.0005    +0.097
  16  0.4684     0.4680    0.3876     -0.0004    +0.080
  17  0.4409     0.4405    0.3742     -0.0004    +0.066
  18  0.4367     0.4366    0.3624     -0.0001    +0.074
  19  0.4391     0.4391    0.3533      0.0000    +0.086
  20  0.4354     0.4354    0.3580      0.0000    +0.077
```

## Key observations

1. **Loss function makes no difference**: DPV+fused vs DPV+CCE differ by <0.002 at every step. The loss function is conclusively ruled out.

2. **Grad norms comparable at step 1** (14.1 vs 14.0), confirming gradient scaling is equivalent.

3. **Grad norms diverge sharply from step 8**: Axolotl grad_norm drops from 11.3→6.9 at step 7→8, while DPV stays at 13.7→12.1. By step 11, Axolotl is at 2.1 vs DPV at 11.1. Axolotl learns faster.

4. **Gap is ~0.08 by step 20** (DPV 0.435 vs Axolotl 0.358). Consistent, not diverging further.

## Ruled out

- ~~Padding/masking artifacts~~ — unpad/repad eliminates all padding from computation
- ~~Loss function~~ — CCE vs fused_linear_cross_entropy makes <0.002 difference
- ~~Data ordering~~ — verified byte-identical data, same seed, same batch composition
- ~~Gradient scaling~~ — matching grad_norms at step 1 confirms equivalent scaling
- ~~Learning rate schedule~~ — both use cosine warmup with same parameters (LR reporting has off-by-one between systems but actual applied LR is identical)

## Remaining candidates

The gap must come from differences in the **forward/backward pass implementation**:

1. **Custom kernels**: DPV uses bitsandbytes custom CUDA kernels for RMSNorm, RoPE, SwiGLU, and attention. Axolotl uses standard PyTorch/HuggingFace implementations with flash_attn. Numerical differences in these kernels accumulate through 36 layers and compound across training steps.

2. **Gradient checkpointing**: DPV uses `checkpoint_cpu_offload` (bitsandbytes) while Axolotl uses standard gradient checkpointing with activation offloading. Different recomputation strategies during backward may produce numerically different gradients.

3. **Model implementation**: DPV uses `FullFinetuneModel` (bitsandbytes) which implements the forward pass differently from HuggingFace's `Qwen2ForCausalLM`. Layer ordering, tensor layouts, and intermediate precision could all differ.

## Experiments

| ID | Description | Status |
|---|---|---|
| `01KSFDTPQQMM2KF6S0D932WQKK` | DPV nopack + fused CE, 20 steps | Complete |
| `01KSFF6ZN14E89VCQ87SRYQE61` | DPV nopack + CCE, 20 steps | Complete |
| Axolotl baseline | `axolotl_8gpu_deepspeed_200step` | Complete (200 steps) |
