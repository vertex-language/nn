# nn

[![package: vs-package](https://img.shields.io/badge/package-vs--package-f4f4f5?style=flat-square&labelColor=e4e4e7&color=18181b)](https://github.com/vertex-language)
[![layers: linear | norm | attention](https://img.shields.io/badge/layers-linear%20%7C%20norm%20%7C%20attention-f4f4f5?style=flat-square&labelColor=e4e4e7&color=18181b)](https://github.com/vertex-language/nn)

Neural network layers and building blocks: Linear projections, quantized weights, RMSNorm, GatedMLP, Attention, and KV cache implementations.

---

## Quick Start

Run the neural network tests in `cmd/` directly with `vsc run`:

```bash
vsc run test-nn        # the decoder layers against plain host math
vsc run test-layers    # the sequence layers against torch.nn (testdata/oracle/layers.py)
```

---

## Packages

| Package | Built | Tested |
| --- | --- | --- |
| `nn` | `Linear` (y = W·x; float32, or Q4_0/Q8_0 blocks decoded in place by `linalg.Gemv`), `Embedding` (a row, dequantized if it is blocks), `RMSNorm`, `GatedMLP` (SwiGLU and the like; gate and up one fused `GateUp` Linear), `Attention` (the Q/K/V projections one fused `QKV` Linear, RoPE over q and k in one launch, grouped-query heads, a token at a time) and its KV `Cache`. `Fused` makes either from separate weights. Every output can `accumulate` into the residual stream | `test-nn` (16 checks): each layer against the same math written plainly on the host, on the CPU device and Metal; `Attention` decoding a token at a time with its cache matches one causal pass over all the tokens |
| `nn` (batches) | Every layer above over n tokens at once: `Linear.Forward(rows:)` is one `linalg.Gemm` (any format), `Embedding.Lookup(tokens:first:count:)` one launch, `Attention.Forward(rows:runs:)` attends each `Run` -- a sequence's tokens -- to its own cache with the tiled multi-query kernel. A `Linear` part may carry a bias and a scale (llama.cpp's `.bias` and `.scale`), in either order | `test-archs` in `llm`: the batch, split and two-sequence paths of `nn/llama` against llama.cpp |
| `nn/cache` | `Memory`: a model's state between batches, by sequence -- `Length`, `Truncate` (drop from a position on: a rejected draft), `Clear` -- and `KV`, full attention's, a cache per layer per sequence made when first written; `Run`, a sequence's part of a batch | with `nn/llama` |
| `nn/decoder` | llama.cpp's spellings of what decoders share: `Aliases` (config keys and tensor names as its converter writes them into a GGUF), `GGUFName`, and `Floats`/`Scalar` for the small tensors read on the host; `Network` and `Standard` layers; `Spec`, a decoder described by flags (RoPE kind, norms and where they sit, QK-norm, biases and scales, sliding windows, parallel layers, multipliers, mixtures of experts) and `Build`, which reads a GGUF by one; `MixtureOf`, a layer's experts, shared expert and selection bias | `test-checkpoint` and `test-archs` in `llm` |
| `nn/moe` | A mixture of experts as llama.cpp's `build_moe_ffn` runs it, chosen on the device: the router kernel (softmax, sigmoid, softmax-of-chosen and √softplus gating, a selection bias, top-k, weights normalized and scaled), the experts through `linalg.GemvIds` in any block format -- gate and up apart or one weight -- with per-expert scales and an optional SwiGLU clamp, and the weighted combine. `Gated`, a shared expert behind a sigmoid gate | `test-archs` in `llm`: 18 MoE fixtures against llama.cpp |
| `nn/llama` | The Llama network, which every task that runs a Llama shares: `Network.Load(checkpoint)`, `Forward(input, runs:, wanted:, memory:)` over a batch of tokens or embeddings, `Step` for a token the device chose. Optional biases and scales on every projection and Llama 3's `rope_freqs` | `test-archs` (`llama-dense` 5e-10 from llama.cpp) and `test-llama` in `llm` |
| `nn` (sequences) | Layers of speech and audio networks over [channels, length] buffers, each its `torch.nn` namesake with the same weights: `Dense` (Linear with a bias, over rows), `Conv1d`, `ConvTranspose1d`, `WeightNormed` (from `weight_g` and `weight_v`), a bidirectional `LSTM`, `InstanceNorm1d`, `AdaIN1d` and `AdaLayerNorm` (a style's (1 + γ)·x + β), `ChannelNorm` | `test-layers` (29 cases): every layer against torch at Kokoro-82M's shapes, strides and paddings, 1e-7 relative on the CPU device and Metal |

---

## License

[MIT](LICENSE)
