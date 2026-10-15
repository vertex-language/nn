// Package decoder is what every decoder network shares below its family:
// llama.cpp's spellings of the names decoders share -- tensor names and
// config keys as its converter writes them into a GGUF -- for reading a
// GGUF in Hugging Face's vocabulary. A network sets them on its checkpoint
// before it reads (ck.Aliases = decoder.Aliases).
package decoder

import (
    "model"
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
