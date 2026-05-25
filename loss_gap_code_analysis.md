# DPV vs Axolotl Loss Gap: Code-Level Analysis

Systematic comparison of code paths between DualPipeV (bitsandbytes FullFinetuneModel) and Axolotl (HuggingFace Trainer + DeepSpeed ZeRO-1) to identify the source of the ~0.08 loss gap at step 20.

## Ruled out by experiment

| Component | DPV impl | Axolotl impl | Experiment | Result |
|---|---|---|---|---|
| Loss function | `fused_linear_cross_entropy` | `cut_cross_entropy` (CCE) | Swapped DPV to CCE | <0.002 diff |
| RMSNorm kernel | CUDA `rmsnorm` | PyTorch `nn.RMSNorm` | Replaced with PyTorch | <0.001 diff |
| RoPE kernel | CUDA `rope` | HF `apply_rotary_pos_emb` | Replaced with PyTorch | <0.001 diff |
| SwiGLU kernel | CUDA `swiglu` | PyTorch `F.silu(gate) * up` | Replaced with PyTorch | <0.001 diff |
| Data ordering | seed=42, `torch.randperm` | seed=42, `DistributedSampler` | Verified byte-identical data, same 32 samples/step | Confirmed identical |
| Padding artifacts | Unpad/repad at comm boundaries | No padding (variable-length) | Unpad/repad approach | Zero NaN, stable training |

## Ruled out by code inspection

### 1. Optimizer configuration — IDENTICAL

**DPV** (`train_dualpipev_packed.py:643`):
```python
optimizer = torch.optim.AdamW(trainable, lr=1e-5,
    betas=(0.9, 0.95), weight_decay=0.01)
# eps defaults to 1e-8
```

**Axolotl** (`axolotl_8gpu_200step.yaml`):
```yaml
optimizer: adamw_torch  # → torch.optim.AdamW
learning_rate: 1e-5
adam_beta1: 0.9
adam_beta2: 0.95
weight_decay: 0.01
# eps defaults to 1e-8 via HF TrainingArguments
```

Both use `torch.optim.AdamW` with identical hyperparameters. No difference.

### 2. Learning rate schedule — IDENTICAL (off-by-one in reporting only)

**DPV** (`train_dualpipev_packed.py:377`):
```python
def lr_lambda(step):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return max(0.0, 0.5 * (1.0 + cos(pi * progress)))
scheduler = LambdaLR(optimizer, lr_lambda)
```

**Axolotl**: Uses HF's `get_cosine_schedule_with_warmup` — identical formula.

DPV reports the LR *after* `scheduler.step()` (next step's LR), while Axolotl reports *before* (current step's LR). This is a logging difference, not a training difference. Both apply lr=0.0 at step 1, lr=2.08e-7 at step 2, etc.

### 3. Gradient accumulation scaling — IDENTICAL

**DPV** (`train_dualpipev_packed.py:766-769`):
- Pipeline sums gradients across 32 chunks
- Then divides by `grad_accum=32`
- Result: mean gradient over 32 samples

**Axolotl** (HF Trainer + DeepSpeed ZeRO-1):
- Each micro-step: loss divided by `gradient_accumulation_steps=4`
- 4 micro-steps accumulated per rank
- `allreduce MEAN` across 8 ranks
- Result: mean gradient over 32 samples (4 × 8)

Both produce `mean(32 gradients)`. Confirmed by matching grad_norms at step 1 (14.0 vs 14.0).

### 4. Gradient clipping — IDENTICAL

**DPV** (`bitsandbytes/dualpipev.py:1003`):
- `clip_grad_norm_` computes local norm², allreduces across pipeline ranks, takes sqrt
- Clips if global_norm > max_norm (1.0)
- Returns pre-clipping norm

**Axolotl** (DeepSpeed):
- Same allreduce-based global norm computation
- Same max_grad_norm=1.0
- Reports pre-clipping norm

### 5. Weight initialization — IDENTICAL

Both load from `Qwen/Qwen3-8B` pretrained. DPV's `FullFinetuneModel.__init__` extracts weights from the HF model and stores them as separate `nn.Parameter`s in bf16 (norms in fp32). No weight transformation occurs — the values are copied verbatim. Verified by matching step-1 losses (within 0.003).

### 6. Attention implementation — IDENTICAL

DPV uses `flash_attn_varlen_func` (packed format with `cu_seqlens=[0, content_len]`). For a single-sequence sample, this is mathematically identical to `flash_attn_func`. Axolotl uses standard `flash_attn` through HF's model. Both use causal attention.

### 7. Data dtype — IDENTICAL

Both compute in bf16 with fp32 optimizer states. DPV stores weights in bf16 (norms in fp32). Axolotl loads in bf16 with `bf16: auto`.

## NOT ruled out — remaining suspects

### 1. Gradient checkpointing implementation — DIFFERENT

**DPV** (`bitsandbytes/training.py`):
Uses custom `_CPUOffloadCheckpointFunction`:
- Forward: copies all layer inputs to CPU (pinned, async), runs forward under `torch.no_grad()`
- Backward: copies inputs back from CPU to GPU, recomputes forward under `torch.enable_grad()`, calls `torch.autograd.backward()` on outputs
- Preserves RNG state for reproducible dropout (though Qwen3 doesn't use dropout)

**Axolotl** (`activation_checkpointing.py`):
Uses PyTorch's `apply_activation_checkpointing` (wraps each `GradientCheckpointingLayer`) + TRL's `OffloadActivations`:
- `OffloadActivations` is a `SavedTensorsHook` that intercepts tensors saved for backward
- It moves them to CPU during forward, brings them back during backward
- Supports streaming with separate CUDA streams for overlap
- Does NOT recompute forward — it saves and restores actual activations

**Key difference**: DPV recomputes the forward pass during backward. Axolotl saves activations and restores them. Both should be mathematically equivalent, but:
- Recomputation may accumulate differently due to different memory allocation patterns
- CPU→GPU async transfer timing could affect which operations overlap with compute
- The `torch.autograd.backward()` call in DPV's custom Function creates a nested autograd context

### 2. Pipeline parallel vs data parallel — DIFFERENT

**DPV**: Each GPU owns a slice of layers. Input flows through the pipeline:
- Activations communicated via P2P (`send`/`recv`) between stages
- Each chunk traverses all stages sequentially
- Loss computed only on the LM head stage (rank 0)
- Backward: gradient flows back through the pipeline via P2P
- The DualPipeV schedule interleaves forward and backward of different chunks

**Axolotl**: Each GPU has the full model, processes different data:
- No P2P communication for activations
- `allreduce` for gradient synchronization
- Each rank computes its own loss

The pipeline parallel approach means that during backward, a stage receives activation gradients from the next stage via P2P, then computes local backward through its layers, then sends activation gradients to the previous stage. This creates a sequential dependency chain that doesn't exist in data parallel.

### 3. DualPipeV step() scheduling — DIFFERENT

The DualPipeV `step()` function (`dualpipev.py:922`) implements a specific chunk scheduling algorithm:
- It processes `num_chunks=32` through the pipeline
- Forward and backward of different chunks are interleaved
- The schedule determines which chunks are in flight simultaneously
- This interleaving means that one chunk's backward may execute while another chunk's forward is running on the same GPU

In data parallel (Axolotl), all micro-steps within a gradient accumulation cycle are fully sequential — forward1, backward1, forward2, backward2, forward3, backward3, forward4, backward4. No interleaving.

### 4. Loss computation and reporting — POTENTIALLY DIFFERENT

**DPV** (`train_dualpipev_packed.py:758-759`):
```python
loss_val = loss_tensor.mean().item()
```
This averages per-chunk losses. Each chunk's loss is the mean over its trainable tokens. This is **mean-of-means** — each sample contributes equally regardless of token count.

**Axolotl**: HF Trainer computes per-sample loss with `reduction='mean'`, then averages across accumulation steps and ranks. Also mean-of-means.

Both use mean-of-means, so reported loss values should be comparable. BUT the training gradients are what actually matter, and the gradient contribution from each sample IS proportional to its token count (since `reduction='mean'` divides by N tokens, the gradient per parameter is `dL/dθ * 1/N`, and accumulating these gives more weight to shorter sequences). This is the same in both systems.

## Next steps to investigate

1. **Test without gradient checkpointing** — run DPV with `--no-cpu-offload` and standard `torch.utils.checkpoint` instead of the custom CPU offload. If losses match Axolotl, the checkpoint implementation is the culprit.

2. **Test with single GPU** — run both DPV and a simple training loop on 1 GPU with the same data, to isolate pipeline-parallel effects from the core computation.

3. **Instrument the DualPipeV step()** — log per-chunk loss and gradient norms during the pipeline schedule to verify they accumulate correctly.
