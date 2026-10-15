// Package qwen is the Qwen network, every generation of it llama.cpp
// runs that is dense attention so far: Qwen (1), Qwen2 and Qwen2.5 (QKV
// biases), and Qwen3 (QK-norm). Each generation is its own Config kind;
// the blocks are nn/decoder's, with what each generation's graph in
// llama.cpp reads and nothing it does not.
//
// It is the network alone, so that every task that runs a Qwen shares
// it: llm/qwen generates text with it, and Qwen3-Embedding, Qwen3-ASR and
// Qwen3-TTS run it too.
package qwen

import (
    "gpu"
    "gpu/neural"
    "model"
    "nn"
    "nn/decoder"
)

/// LoadError is a checkpoint that is not a Qwen this can run, and says why.
public enum LoadError: Error, CustomStringConvertible {
    case unsupported(string)

    public var description: string {
        switch self {
        case .unsupported(let why): return "qwen: " + why
        }
    }
}

/// Generation is which Qwen a checkpoint is.
public enum Generation: Equatable {
    case qwen
    case qwen2
    case qwen3
}

/// Names are the architecture names this network loads: Hugging Face's and
/// GGUF's.
public let Names = ["QWenLMHeadModel", "qwen", "Qwen2ForCausalLM", "Qwen2Model", "qwen2", "Qwen3ForCausalLM", "Qwen3Model", "qwen3"]

/// GenerationOf is the generation an architecture name is, or nil.
public func GenerationOf(_ arch: string) -> Generation? {
    switch arch {
    case "QWenLMHeadModel", "qwen": return .qwen
    case "Qwen2ForCausalLM", "Qwen2Model", "qwen2": return .qwen2
    case "Qwen3ForCausalLM", "Qwen3Model", "qwen3": return .qwen3
    default: return nil
    }
}

/// Config is a Qwen's shape.
public struct Config {
    public var Generation: Generation
    public var Layers: int
    public var Dim: int
    public var Heads: int
    public var KVHeads: int
    public var HeadDim: int
    public var Context: int
    public var Eps: float32
    public var RopeBase: float32

    /// Read is a checkpoint's Qwen config.
    public static func Read(_ ck: model.Checkpoint) throws -> Config {
        ck.Aliases = decoder.Aliases
        guard let g = GenerationOf(ck.Architecture) else {
            throw LoadError.unsupported("\(ck.Architecture) is not a Qwen")
        }
        let c = ck.Config
        func need(_ key: string) throws -> int {
            guard let n = c.Int(key) else {
                throw LoadError.unsupported("no \(key) in the config")
            }
            return n
        }
        let dim = try need("hidden_size")
        let heads = try need("num_attention_heads")
        if let experts = c.Int("num_experts"), experts > 0 {
            throw LoadError.unsupported("a mixture of \(experts) experts is not run yet")
        }
        return Config(
            Generation: g,
            Layers: try need("num_hidden_layers"),
            Dim: dim,
            Heads: heads,
            KVHeads: c.Int("num_key_value_heads") ?? heads,
            HeadDim: c.Int("head_dim") ?? dim / heads,
            Context: c.Int("max_position_embeddings") ?? 32768,
            Eps: float32(c.Double("rms_norm_eps") ?? 1e-6),
            RopeBase: float32(c.Double("rope_theta") ?? 10000))
    }
}

/// Load reads a Qwen checkpoint onto device d.
public func Load(_ ck: model.Checkpoint, on d: gpu.Device) async throws -> decoder.Network {
    let c = try Config.Read(ck)
    let w = decoder.Weights(ck, on: d)
    // Qwen's rotary embeddings turn halves of a head (NeoX's), in llama.cpp
    // and Hugging Face alike.
    let rope = nn.Rope(base: c.RopeBase, layout: neural.RopeLayout.halves)
    let table = try await neural.RopeTable.Make(on: d, dim: c.HeadDim, positions: c.Context, base: c.RopeBase)
    let embed = try nn.Embedding(try await w.Tensor("model.embed_tokens.weight"))
    var blocks: [decoder.Block] = []
    for i in 0..<c.Layers {
        let p = "model.layers.\(i)."
        let g = "blk.\(i)."
        let attention = try nn.Attention(
            qkv: try await decoder.QKV(w, p + "self_attn.", g),
            // Qwen's graph adds no bias to its output projection; Qwen2's
            // and Qwen3's do where there is one.
            o: try await w.Linear(p + "self_attn.o_proj", g + "attn_output", bias: c.Generation != .qwen),
            heads: c.Heads, kvHeads: c.KVHeads, headDim: c.HeadDim, rope: rope)
        attention.Table = table
        if c.Generation == .qwen3 {
            attention.QNorm = nn.HeadNorm(try await w.RMS(p + "self_attn.q_norm", c.Eps, gg: g + "attn_q_norm"))
            attention.KNorm = nn.HeadNorm(try await w.RMS(p + "self_attn.k_norm", c.Eps, gg: g + "attn_k_norm"))
        }
        // The MLP's biases never, and its scales only in Qwen3's graph.
        let s = c.Generation == .qwen3
        let mlp = try nn.GatedMLP.Fused(
            gate: try await w.Linear(p + "mlp.gate_proj", g + "ffn_gate", bias: false, scale: s),
            up: try await w.Linear(p + "mlp.up_proj", g + "ffn_up", bias: false, scale: s),
            down: try await w.Linear(p + "mlp.down_proj", g + "ffn_down", bias: false, scale: s),
            activation: .SiLU)
        blocks.append(decoder.Standard(attnNorm: try await w.RMS(p + "input_layernorm", c.Eps), attention: attention,
                                       ffnNorm: try await w.RMS(p + "post_attention_layernorm", c.Eps), ffn: mlp))
    }
    let tied = !w.Has("lm_head.weight", "output.weight")
    let output = tied ? try nn.Linear(embed.Weight)
                      : try await w.Linear("lm_head", "output", bias: c.Generation == .qwen2)
    return decoder.Network(on: d, dim: c.Dim, context: c.Context, embed: embed, blocks: blocks, kvHeads: c.KVHeads,
                           headDim: c.HeadDim, norm: try await w.RMS("model.norm", c.Eps), output: output)
}
