# nn

The layers models are made of (`proposed_ai_packages.md` §6.2), over
`tensor`. Weights are `tensor.Tensor`s of any format, and each layer
dispatches on its weight's `DType`. Activations are float32 `gpu.Buffer`s.
A layer keeps the scratch it needs, so decoding a token allocates nothing.

| Package | Built | Tested |
| --- | --- | --- |
| `nn` | `Linear` (y = W·x; float32, or Q4_0/Q8_0 blocks decoded in place by `linalg.Gemv`), `Embedding` (a row, dequantized if it is blocks), `RMSNorm`, `GatedMLP` (SwiGLU and the like), `Attention` (Q/K/V/O projections, RoPE, grouped-query heads, a token at a time) and its KV `Cache` | `test-nn`: each layer against the same math written plainly on the host, on the CPU device and Metal; `Attention` decoding a token at a time with its cache matches one causal pass over all the tokens |

`Attention` lives in `nn` itself for now, rather than in `nn/attention`:
a module of that name would clash with `gpu/attention`'s module name,
`attention`. Batches, prefill, MoE, SSMs, LoRA and losses come with the
models and training that need them.

```console
$ vsc run test-nn
```
