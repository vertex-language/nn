// Package decoder is what every decoder network shares below its family:
// llama.cpp's spellings of the names decoders share -- tensor names and
// config keys as its converter writes them into a GGUF -- for reading a
// GGUF in Hugging Face's vocabulary. A network sets them on its checkpoint
// before it reads (ck.Aliases = decoder.Aliases).
package decoder

import (
    "gpu"
    "model"
    "nn"
    "tensor"
)

/// GGUFName is the GGUF name llama.cpp's converter gives a Hugging Face
/// tensor name, by the table most decoders share (gguf-py's
/// tensor_mapping.py): "model.layers.3.self_attn.q_proj.weight" is
/// "blk.3.attn_q.weight". nil for a name outside the table: a family asks
/// for such a tensor with its GGUF name given.
public func GGUFName(_ hf: string) -> string? {
    var base = hf
    var suffix = ""
    for s in [".weight", ".bias"] {
        if hf.hasSuffix(s) {
            base = String(hf.dropLast(s.count))
            suffix = s
            break
        }
    }
    if let top = topNames[base] {
        return top + suffix
    }
    let prefix = "model.layers."
    if !base.hasPrefix(prefix) {
        return nil
    }
    let rest = String(base.dropFirst(prefix.count))
    guard let dot = rest.firstIndex(of: ".") else {
        return nil
    }
    let layer = String(rest[rest.startIndex..<dot])
    let part = String(rest[rest.index(after: dot)..<rest.endIndex])
    guard int(layer) != nil, let name = layerNames[part] else {
        return nil
    }
    return "blk.\(layer).\(name)\(suffix)"
}

let topNames: [string: string] = [
    "model.embed_tokens": "token_embd",
    "model.norm": "output_norm",
    "lm_head": "output",
]

// Within a layer. post_attention_layernorm is the llama convention (the
// norm before the MLP); a family whose name means something else there
// (Gemma's sandwich norms) names its GGUF tensors itself.
let layerNames: [string: string] = [
    "input_layernorm": "attn_norm",
    "self_attn.q_proj": "attn_q",
    "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_output",
    "self_attn.q_norm": "attn_q_norm",
    "self_attn.k_norm": "attn_k_norm",
    "post_attention_layernorm": "ffn_norm",
    "mlp.gate_proj": "ffn_gate",
    "mlp.up_proj": "ffn_up",
    "mlp.down_proj": "ffn_down",
]

/// The config.json keys every decoder shares, as llama.cpp's converter
/// writes them into a GGUF (gguf-py's constants, under the architecture).
let decoderKeys: [string: string] = [
    "num_hidden_layers": "block_count",
    "hidden_size": "embedding_length",
    "intermediate_size": "feed_forward_length",
    "num_attention_heads": "attention.head_count",
    "num_key_value_heads": "attention.head_count_kv",
    "head_dim": "attention.key_length",
    "rms_norm_eps": "attention.layer_norm_rms_epsilon",
    "layer_norm_eps": "attention.layer_norm_epsilon",
    "rope_theta": "rope.freq_base",
    "max_position_embeddings": "context_length",
    "vocab_size": "vocab_size",
    "sliding_window": "attention.sliding_window",
    "num_experts": "expert_count",
    "num_experts_per_tok": "expert_used_count",
    "bos_token_id": "tokenizer.ggml.bos_token_id",
    "eos_token_id": "tokenizer.ggml.eos_token_id",
    "pad_token_id": "tokenizer.ggml.padding_token_id",
]

/// Aliases are llama.cpp's spellings of the names every decoder
/// shares, for reading a GGUF in Hugging Face's vocabulary: a family sets
/// them on its checkpoint before it reads (ck.Aliases = decoder.Aliases).
public let Aliases = model.Aliases(configKeys: decoderKeys, tensorName: { GGUFName($0) })

/// Floats is a small float32 tensor's values, read on the host: a weight's
/// scale, rope_freqs. Other dtypes are refused.
public func Floats(_ ck: model.Checkpoint, _ t: model.TensorInfo) throws -> [float32] {
    if t.DType != tensor.DType.F32 {
        throw model.OpenError.unsupported("\(t.Name) is \(t.SourceType); a float32 tensor is read here")
    }
    let b = ck.Copy(t)
    var out: [float32] = []
    var i = 0
    while i + 4 <= b.count {
        out.append(float32(bitPattern: uint32(b[i]) | uint32(b[i + 1]) << 8 | uint32(b[i + 2]) << 16 | uint32(b[i + 3]) << 24))
        i += 4
    }
    return out
}

/// Scalar is a one-element float32 tensor's value: llama.cpp's per-weight
/// ".scale".
public func Scalar(_ ck: model.Checkpoint, _ t: model.TensorInfo) throws -> float32 {
    let v = try Floats(ck, t)
    if v.count != 1 {
        throw model.OpenError.unsupported("\(t.Name) holds \(v.count) values where a scale is one")
    }
    return v[0]
}

/// _suffixed is a GGUF base name with a suffix, or nil where there is no
/// GGUF name: the Hugging Face name is read through Aliases then.
func _suffixed(_ gg: string?, _ suffix: string) -> string? {
    if let g = gg {
        return g + suffix
    }
    return nil
}

/// Weights reads a checkpoint's tensors onto a device for a network:
/// by Hugging Face name (through Aliases for a GGUF), with llama.cpp's
/// GGUF name for what only a GGUF has.
public final class Weights {
    public let Checkpoint: model.Checkpoint
    public let Device: gpu.Device

    public init(_ ck: model.Checkpoint, on d: gpu.Device) {
        self.Checkpoint = ck
        self.Device = d
    }

    /// Alloc is a buffer of count floats on the device: apart from the
    /// async readers because the compiler's SIL verifier rejects
    /// CreateBuffer specialized inside a branch of an async function.
    public func Alloc(_ count: int) throws -> gpu.Buffer<float32> {
        return try Device.CreateBuffer(of: float32.self, count: count)
    }

    /// Has is whether the tensor is there: hf by its Hugging Face name, or
    /// gg by its GGUF name.
    public func Has(_ hf: string, _ gg: string? = nil) -> bool {
        return Checkpoint.Has(hf, gguf: Checkpoint.Format == .gguf ? gg : nil)
    }

    /// Tensor is the named tensor on the device, in the file's format.
    public func Tensor(_ hf: string, _ gg: string? = nil) async throws -> tensor.Tensor {
        return try await Checkpoint.Tensor(hf, gguf: Checkpoint.Format == .gguf ? gg : nil, on: Device)
    }

    /// Floats is the named tensor as float32 on the device.
    public func Floats(_ hf: string, _ gg: string? = nil) async throws -> gpu.Buffer<float32> {
        return try (try await Checkpoint.Float32(hf, gguf: Checkpoint.Format == .gguf ? gg : nil, on: Device)).Floats()
    }

    /// Optional is the named tensor as float32, or nil where it is not.
    public func Optional(_ hf: string, _ gg: string? = nil) async throws -> gpu.Buffer<float32>? {
        if !Has(hf, gg) {
            return nil
        }
        return try await Floats(hf, gg)
    }

    /// Linear is a projection: base's ".weight", and the ".bias" and
    /// llama.cpp's ".scale" a checkpoint may give it (gg the GGUF base
    /// name, where it differs), applied in the order biasFirst says.
    /// bias and scale say whether the network's graph reads them: llama.cpp
    /// loads a ".scale" for every projection that has one, and each graph
    /// applies those it names.
    public func Linear(_ base: string, _ gg: string? = nil, biasFirst: bool = false, bias useBias: bool = true,
                       scale useScale: bool = true) async throws -> nn.Linear {
        var bias: gpu.Buffer<float32>? = nil
        if useBias && Has(base + ".bias", _suffixed(gg, ".bias")) {
            bias = try await Floats(base + ".bias", _suffixed(gg, ".bias"))
        }
        var scale: float32 = 1
        if useScale && Checkpoint.Format == .gguf && gg != nil {
            if let t = Checkpoint.Info(base + ".scale", gguf: gg! + ".scale") {
                scale = try Scalar(Checkpoint, t)
            }
        }
        let weight = try await Tensor(base + ".weight", _suffixed(gg, ".weight"))
        return try nn.Linear(weight, bias: bias, scale: scale, biasFirst: biasFirst)
    }

    /// RMS is the RMSNorm whose weight is base's ".weight" (and ".bias",
    /// where it has one).
    public func RMS(_ base: string, _ eps: float32, gg: string? = nil) async throws -> nn.Norm {
        return try await Norm(.rms, base, eps, gg: gg)
    }

    /// Norm is the norm of kind whose weight is base's ".weight", and
    /// ".bias" where it has one.
    public func Norm(_ kind: nn.NormKind, _ base: string, _ eps: float32, gg: string? = nil) async throws -> nn.Norm {
        let weight = try await Floats(base + ".weight", _suffixed(gg, ".weight"))
        var bias: gpu.Buffer<float32>? = nil
        if Has(base + ".bias", _suffixed(gg, ".bias")) {
            bias = try await Floats(base + ".bias", _suffixed(gg, ".bias"))
        }
        return nn.Norm(kind, weight: weight, bias: bias, eps: eps)
    }
}

/// QKV is a layer's Q, K and V projections, stacked: one fused weight
/// (GGUF's attn_qkv, with its bias, or with separate Q, K and V biases)
/// where the checkpoint has one, else the three. hf and gg are the
/// layer's attention prefixes ("model.layers.3.self_attn.", "blk.3.").
public func QKV(_ w: Weights, _ hf: string, _ gg: string, bias: bool = true, scale: bool = true) async throws -> [nn.Linear] {
    if w.Has(hf + "qkv_proj.weight", gg + "attn_qkv.weight") {
        let fused = try await w.Linear(hf + "qkv_proj", gg + "attn_qkv", bias: bias, scale: scale)
        if bias && fused.Biases[0] == nil && w.Has(hf + "q_proj.bias", gg + "attn_q.bias") {
            // Separate biases over a fused weight: one bias of the three.
            let qb = try await w.Floats(hf + "q_proj.bias", gg + "attn_q.bias")
            let kb = try await w.Floats(hf + "k_proj.bias", gg + "attn_k.bias")
            let vb = try await w.Floats(hf + "v_proj.bias", gg + "attn_v.bias")
            let all = try w.Alloc(qb.count + kb.count + vb.count)
            try await all.Slice(from: 0, count: qb.count).Copy(from: qb)
            try await all.Slice(from: qb.count, count: kb.count).Copy(from: kb)
            try await all.Slice(from: qb.count + kb.count, count: vb.count).Copy(from: vb)
            return [try nn.Linear(fused.Weight, bias: all, scale: fused.Scales[0])]
        }
        return [fused]
    }
    return try nn.Stacked([try await w.Linear(hf + "q_proj", gg + "attn_q", bias: bias, scale: scale),
                           try await w.Linear(hf + "k_proj", gg + "attn_k", bias: bias, scale: scale),
                           try await w.Linear(hf + "v_proj", gg + "attn_v", bias: bias, scale: scale)])
}
