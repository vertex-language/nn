# nn

[![package: vs-package](https://img.shields.io/badge/package-vs--package-f4f4f5?style=flat-square&labelColor=e4e4e7&color=18181b)](https://github.com/vertex-language)
[![devices: metal | cpu](https://img.shields.io/badge/devices-metal%20%7C%20cpu-f4f4f5?style=flat-square&labelColor=e4e4e7&color=18181b)](https://github.com/vertex-language/nn)
[![layers: transformer | moe | audio](https://img.shields.io/badge/layers-transformer%20%7C%20moe%20%7C%20audio-f4f4f5?style=flat-square&labelColor=e4e4e7&color=18181b)](https://github.com/vertex-language/nn)
[![license: MIT](https://img.shields.io/badge/license-MIT-f4f4f5?style=flat-square&labelColor=e4e4e7&color=18181b)](LICENSE)

Neural network layers and model architectures in Vertex: dense and block-quantized linear projections, stateful KV caches, multi-head attention with GQA/MQA, SwiGLU MLPs, on-device Mixture of Experts (MoE), declarative decoder specification, and speech/audio sequence layers.

---

## Architecture Overview

`nn` builds upon `tensor` and `gpu` to provide modular, zero-allocation neural network layers for modern transformer decoders and deep learning pipelines:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Text Generation Facade                          │
│                   llm (Families, Greedy Decoding)                      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                       nn Repository (This Library)                     │
│  nn            · Linear (dense & Q4_0/Q8_0/Q4_K/Q6_K), Embedding,       │
│                  RMSNorm, Norm, Attention, GatedMLP, MLP, FFN          │
│  nn (conv.vs)  · Dense, Conv1d, ConvTranspose1d, LSTM, WeightNormed,   │
│                  InstanceNorm1d, AdaIN1d, AdaLayerNorm, ChannelNorm    │
│  nn/cache      · Multi-sequence Memory protocol, KV cache, Run batches │
│  nn/decoder    · Spec architecture declarative builder, Network,       │
│                  Standard decoder blocks, GGUF/HF Aliases              │
│  nn/moe        · Mixture of Experts (softmax, sigmoid, sqrtSoftplus),  │
│                  Top-K routing with selection bias, GemvIds dispatch   │
│  nn/llama      · Llama 1/2/3, TinyLlama, SmolLM2, Mixtral loader       │
│  nn/qwen       · Qwen 1, Qwen 2/2.5, Qwen 3, Qwen2/3-MoE loader        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                      Accelerated Backend Libraries                     │
│    gpu/linalg (Gemv, Gemm, GemvIds) · gpu/neural · gpu/attention       │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

Run the test suites in `cmd/` with `vsc run`:

```bash
vsc run test-nn        # 16 checks: decoder layers & causal attention vs host math
vsc run test-layers    # 29 checks: sequence & audio layers vs PyTorch (Kokoro-82M)
```

---

## Packages Summary

| Package | Capabilities | Hardware & Verification |
| --- | --- | --- |
| **`nn`** (Core) | `Linear`: supports `F32`, `F16`, `BF16`, `Q4_0`, `Q8_0`, `Q4_K`, `Q6_K` weights, stacking up to 3 projections (e.g. Q/K/V or Gate/Up) into one kernel with zero weight copies (`Stacked`), per-part biases and scales, residual accumulation (`accumulate: true`), and batched evaluation (`linalg.Gemm`). `Embedding`: scalar lookup, batched lookup, and on-device token lookup (`LookupRow`) with zero host round-trips. `RMSNorm` & `Norm`. `Attention`: GQA, MQA, RoPE (`RopeTable`), QK normalization (`HeadNorm`), sliding windows, QKV clamping, and KV cache integration. `GatedMLP` (SwiGLU, GeGLU), `MLP`, and `FFN` protocol. | Verified by `cmd/test-nn` on CPU and Metal against host math and causal references. |
| **`nn`** (Audio & Sequence) | Sequence layers in `conv.vs` over `[channels, length]` buffers: `Dense` (linear with row bias), `Conv1d` and `ConvTranspose1d` (arbitrary stride, padding, dilation, groups), `WeightNormed` ($w = g \cdot v / \|v\|$), bidirectional `LSTM`, `InstanceNorm1d`, `AdaIN1d`, `AdaLayerNorm` ($(1+\gamma)x+\beta$), and `ChannelNorm`. | Verified by `cmd/test-layers` against PyTorch fixtures from `testdata/layers.safetensors` within $10^{-5}$ tolerance. |
| **`nn/cache`** | Multi-sequence state management: `Memory` protocol (`Length`, `Truncate`, `Clear`), `KV` cache storing key and value history per layer and sequence up to 256 concurrent sequences (`MaxSequences`), and `Run` sequence slice descriptors. | Powers sequence batching and speculative drafting across `nn` and `llm`. |
| **`nn/decoder`** | Declarative decoder specification and network synthesis: `Spec` captures all architectural variations in llama.cpp (RoPE kinds, norms, QK-norm, sliding window attention, bias/scale rules, MoE configurations), `Build` loads a GGUF checkpoint into a `Network`, `Standard` decoder blocks, `Aliases` for Hugging Face and GGUF tensor/config name resolution, and `Weights` extraction. | Verified against 50+ architectures in `llm` test suites. |
| **`nn/moe`** | On-device Mixture of Experts matching llama.cpp's `build_moe_ffn`: `Gating` modes (`.softmax`, `.sigmoid`, `.softmaxWeight`, `.sqrtSoftplus`), GPU router kernel selecting top-$k$ experts with selection bias correction and normalization, `Experts` dispatch using `linalg.GemvIds` in all block formats with fused GateUp or separate weights, SwiGLU clamping, per-expert scales, and `Gated` shared experts. | Verified against 18 MoE fixtures in `llm/testdata/archs` against llama.cpp. |
| **`nn/llama`** | Dedicated network loader for the Llama family: Llama 1, Llama 2, Llama 3 (with `rope_freqs`), TinyLlama, SmolLM2, Mixtral, and fine-tunes. | Bit-accurate logits against llama.cpp. |
| **`nn/qwen`** | Dedicated network loader for all Qwen generations: Qwen 1, Qwen 2 and 2.5 (QKV bias support), Qwen 3 (QK normalization), Qwen2-MoE (gated shared expert), and Qwen3-MoE. | Verified against Qwen test models and llama.cpp reference logits. |

---

## Detailed Package Specifications

### 1. Core Layers (`nn`)

```vertex
import "nn"
```

#### `Linear`
Evaluates $y = W \cdot x$ (or $Y = X \cdot W^T$ for batched inputs):
- **Data Types**: Float32, Float16, BFloat16, and block-quantized weights: `Q4_0`, `Q8_0`, `Q4_K`, `Q6_K`.
- **Zero-Copy Weight Stacking**: Combines up to 3 weights of identical input dimension (e.g. Q, K, V projections or Gate and Up projections) into a single `Linear` using `Linear(stacking:)` or `Stacked([Linear])`. Evaluates all parts in a single `Gemv` or `Gemm` dispatch without concatenating weight buffers.
- **Biases and Scales**: Each stacked part independently supports a float32 bias vector and scalar multiplier. `BiasFirst` controls the evaluation order:
  - Attention order: $y = \text{scale} \cdot (W \cdot x) + \text{bias}$
  - MLP order: $y = \text{scale} \cdot (W \cdot x + \text{bias})$
- **Residual Accumulation**: `Forward(..., accumulate: true)` fuses residual addition into the output buffer pass.
- **Batched Token Processing**: `Forward(x, rows: n, into: y, stride:, column:)` dispatches a $32 \times 32$ tiled `linalg.Gemm`, reusing weight tiles across 32 tokens.

#### `Embedding`
Vector lookup table for token representations:
- **Lookup Modes**:
  - `Lookup(token, into: x)`: Standard lookup for a single host token.
  - `Lookup(tokens: Buffer<int32>, at: index, into: x)`: On-device lookup via `dtype.DequantizeRow`. Reads the next token index directly from GPU memory (e.g. from greedy `Argmax`) without synchronizing with the host.
  - `Lookup(tokens: Buffer<int32>, first:, count:, into: x)`: Batched token lookup via `dtype.DequantizeRows` in one launch.
- **Quantization Support**: Dequantizes Float16, BFloat16, Q4_0, Q8_0, Q4_K, and Q6_K rows directly into the destination activation buffer.

#### `RMSNorm` & `Norm`
- **`RMSNorm`**: Scales activations to unit root-mean-square, multiplied by a learned float32 weight:
  $$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}} \odot w$$
- **`Norm`**: Generalized normalization supporting `NormKind` (`.rms`, `.layer`, `.none`) with optional weights and biases.

#### `Attention`
Full transformer multi-head self-attention:
- **Projections**: Fused/stacked `QKV` projections and output projection `O`.
- **Head Configurations**: Standard Multi-Head Attention, Grouped-Query Attention (GQA), and Multi-Query Attention (MQA) where `Heads % KVHeads == 0`.
- **Rotary Position Embeddings (`Rope`)**:
  - `Base`: Base frequency (e.g. 10,000 or 500,000).
  - `Layout`: `.halves` (split head dimensions) or `.interleaved` (adjacent pairs).
  - `Dims`: Rotary dimension count (supports partial rotary embeddings; `-1` for none).
  - `Table`: Shared precomputed `neural.RopeTable` storing cosines and sines.
- **Head Normalization (`HeadNorm`)**: Normalizes query and key heads. Supports normalizing each head individually (`each: true`) or all heads together (`each: false`). Controlled by `NormAfterRope` (before rotation for Qwen3; after rotation for Maincoder).
- **Sliding Window Attention (`Window`)**: Restricts attention visibility to local context windows.
- **QKV Activation Clamping (`Clamp`)**: Clamps projected queries, keys, and values to $\pm\text{Clamp}$ (e.g. DBRX).
- **Cache Management**: Binds with `Cache` and `Run` to attend across previously generated context.

#### `GatedMLP` & `MLP`
- **`GatedMLP`**: Implements SwiGLU, GeGLU, and gated feed-forward networks:
  $$y = \text{Down}\Big(\text{Activate}\big(\text{Gate}(x)\big) \odot \text{Up}(x)\Big)$$
  - `Fused(gate:up:down:activation:)`: Automatically stacks separate Gate and Up weights into a unified `GateUp` linear layer.
  - Activation functions from `gpu/neural`: `.SiLU`, `.GELU`, `.GELUTanh`, `.ReLU`, `.ReLUSquared`.
- **`MLP`**: Standard two-layer feed-forward network without gating ($y = \text{Down}(\text{Activate}(\text{Up}(x)))$).
- **`FFN` Protocol**: Common interface adopted by `GatedMLP`, `MLP`, `moe.MoE`, `moe.Gated`, and `decoder.Sum`.

---

### 2. Sequence & Audio Layers (`conv.vs`)

Specialized layers designed for speech synthesis and audio generation networks (such as Kokoro-82M), operating over `[channels, length]` tensors:

- **`Dense`**: Linear transformation with learned bias evaluated over rows.
- **`Conv1d` & `ConvTranspose1d`**: One-dimensional convolutions with configurable stride, padding, dilation, and channel groups.
- **`WeightNormed`**: PyTorch-compatible weight normalization:
  $$w = g \cdot \frac{v}{\|v\|}$$
- **`LSTM`**: Bidirectional Long Short-Term Memory recurrent network.
- **`InstanceNorm1d`**: Per-sample, per-channel normalization along the temporal dimension.
- **`AdaIN1d`**: Adaptive Instance Normalization for audio style conditioning.
- **`AdaLayerNorm`**: Style-modulated layer normalization ($y = (1 + \gamma) \cdot x + \beta$).
- **`ChannelNorm`**: Normalization across channels.

---

### 3. Cache & Sequence Memory (`nn/cache`)

```vertex
import "nn/cache"
```

Provides stateful memory management for autoregressive generation and multi-sequence batching:

- **`Memory` Protocol**:
  - `Length(sequence:) -> int`: Number of cached tokens for a given sequence index.
  - `Truncate(sequence:, from: int)`: Rolls back cache state to position `from` (used during speculative decoding rejection).
  - `Clear(sequence:)`: Resets sequence state.
- **`KV` Class**:
  - Allocates key and value buffers across all layers and sequences.
  - Supports up to 256 concurrent sequences (`MaxSequences = 256`).
- **`Run` Struct**:
  - Encapsulates one sequence's portion of a batch: `Row` index in batch, token `Count`, starting `Position`, and target `Cache`.

---

### 4. Decoder Architecture Builder (`nn/decoder`)

```vertex
import "nn/decoder"
```

A specification-driven system that builds complete transformer networks from GGUF and Hugging Face checkpoints matching llama.cpp's execution graphs:

#### `Spec` Descriptor
Declares all architectural variations across model families:
- **RoPE Configurations**: `RopeKind` (`.norm`, `.neox`, `.none`), `RopeFactors` (Llama 3 frequency scaling), `NoRopeEvery` (SmolLM3), `RopeSlidingOnly` (EXAONE 4).
- **Normalization Placement**: `Norm` (`.rms`, `.layer`, `.none`), `NormBias`, `NormWeights` (false for OLMo), `PreNorms` (false for OLMo 2), `FFNNorm`, `PostAttnNorm` / `PostFFNNorm` (Gemma 2, OLMo 2).
- **QK Normalization**: `QKNormKind` (`.none`, `.beforeRope`, `.afterRope`), `QKNormEach` (individual head normalization).
- **Embedding Scaling**: `EmbedSqrtDim` (scales embeddings by $\sqrt{d}$ for Gemma).
- **Projection Scaling & Biases**: `AttnOutBias`, `AttnOutScale`, `FFNBias`, `FFNScale`, `OutputBias`, `OutputScale`.
- **Feed-Forward Structure**: `FFNGated`, `FFNFused` (ChatGLM, GLM-4, Phi-3 fused GateUp weights), `FFNActivation`.
- **Global Scaling**: `AttentionScale`, `ResidualScale` (Granite), `LogitScale` (`.none`, `.multiply`, `.divide`).
- **Sliding Window Attention (SWA)**: `SWA`, `SWAPattern`, `SWADenseFirst`.
- **Mixture of Experts**: `MoE`, `MoEGating`, `MoENorm`, `MoEFileGating`, `MoEFileNorm`, `MoEExpertScales`, `MoESharedExpert`.

#### Network Execution
- `Build(checkpoint, on: device, spec) async throws -> Network`: Constructs and initializes a model.
- `Network.Forward(input, runs:, wanted:, memory:)`: Executes batched inference across arbitrary token sequences and hidden-state extraction masks.
- `Network.Step(input, runs:, memory:, token:)`: Executes single-token autoregressive decoding on the GPU with zero host round-trips.

---

### 5. Mixture of Experts (`nn/moe`)

```vertex
import "nn/moe"
```

Implements high-performance Mixture of Experts routing and execution matching llama.cpp's `build_moe_ffn`:

- **Router Gating (`Gating`)**:
  - `.softmax`: Standard softmax across all experts (Mixtral, Qwen).
  - `.sigmoid`: Independent sigmoid gating per expert (DeepSeek V3, GLM-4.5).
  - `.softmaxWeight`: Raw routing logits, followed by softmax over chosen top-$k$ experts only.
  - `.sqrtSoftplus`: $\sqrt{\text{softplus}}$ gating per expert (DeepSeek V4).
- **GPU Router Kernel (`_route`)**:
  - Evaluates gating probabilities directly in shared memory.
  - Applies expert selection bias (DeepSeek V3 correction term).
  - Picks top-$k$ experts per token with deterministic tie-breaking.
  - Normalizes and scales chosen expert weights.
- **Expert Dispatch (`Experts`)**:
  - Dispatches expert weights in any block format (`F32`, `F16`, `BF16`, `Q4_0`, `Q8_0`, `Q4_K`, `Q6_K`) via `linalg.GemvIds`.
  - Supports fused GateUp tensors (`ffn_gate_up_exps`) or separate Gate and Up tensors (`ffn_gate_exps`, `ffn_up_exps`).
  - Supports SwiGLU activation clamping and per-expert scaling factors.
- **Shared Experts (`Gated`)**:
  - Evaluates shared expert layers behind a sigmoid gate (e.g. Granite MoE).

---

### 6. Built-in Network Loaders (`nn/llama`, `nn/qwen`)

- **`nn/llama`**:
  - Architecture names: `LlamaForCausalLM`, `llama`.
  - Supports Llama 1/2/3, TinyLlama, SmolLM2, Mixtral, and related fine-tunes.
  - Automatically loads Llama 3 `rope_freqs` and projection biases/scales.
- **`nn/qwen`**:
  - Architecture names: `QWenLMHeadModel`, `qwen`, `Qwen2ForCausalLM`, `qwen2`, `Qwen3ForCausalLM`, `qwen3`, `Qwen2MoeForCausalLM`, `qwen2moe`, `Qwen3MoeForCausalLM`, `qwen3moe`.
  - Configures QKV biases for Qwen 2/2.5, QK-normalization for Qwen 3, and shared gated experts for Qwen MoE.

---

## License

[MIT](LICENSE)
