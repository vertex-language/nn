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
| `nn` (sequences) | Layers of speech and audio networks over [channels, length] buffers, each its `torch.nn` namesake with the same weights: `Dense` (Linear with a bias, over rows), `Conv1d`, `ConvTranspose1d`, `WeightNormed` (from `weight_g` and `weight_v`), a bidirectional `LSTM`, `InstanceNorm1d`, `AdaIN1d` and `AdaLayerNorm` (a style's (1 + γ)·x + β), `ChannelNorm` | `test-layers` (29 cases): every layer against torch at Kokoro-82M's shapes, strides and paddings, 1e-7 relative on the CPU device and Metal |

---

## License

[MIT](LICENSE)
