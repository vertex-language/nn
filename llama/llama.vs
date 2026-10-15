// Package llama is the Llama network: Llama 1 to 3, and the models that
// share its architecture (LlamaForCausalLM on the Hugging Face Hub,
// general.architecture "llama" in GGUF: TinyLlama, SmolLM2, Mistral 7B,
// most fine-tunes) -- an embedding, blocks of RMSNorm, grouped-query
// attention with rotary position embeddings, RMSNorm and a SwiGLU MLP,
// each added back to what went in, then a final RMSNorm and the head.
//
// It is the network alone -- a decoder.Network of decoder.Standard
// blocks -- so that every task that runs a Llama shares it: llm/llama
// generates text with it, and embedding, speech recognition and speech
// synthesis families run it too.
//
// It is written once, against model.Checkpoint: the same code loads a
// Hugging Face snapshot (safetensors) or a GGUF file. The one thing it
// must tell apart is the layout of Q and K: llama.cpp permutes their rows
// for interleaved RoPE when it writes a llama GGUF, and Hugging Face's are
// in halves.
package llama

import (
    "gpu"
    "gpu/neural"
    "model"
    "nn"
    "nn/cache"
    "nn/decoder"
    "tensor"
)

/// LoadError is a checkpoint that is not a llama this can run, and says why.
public enum LoadError: Error, CustomStringConvertible {
    case unsupported(string)

    public var Message: string {
        switch self {
        case .unsupported(let why): return "llama: " + why
        }
    }

    public var description: string { return Message }
}

/// Names are the architecture names this network loads: Hugging Face's and
/// GGUF's.
public let Names = ["LlamaForCausalLM", "llama"]

/// Config is a llama's shape.
public struct Config {
    public var Layers: int
    public var Dim: int
    public var Hidden: int
    public var Heads: int
    public var KVHeads: int
    public var HeadDim: int
    public var Vocab: int
    public var Context: int
    public var Eps: float32
    public var RopeBase: float32
    /// TiedEmbeddings is whether the head is the embedding.
    public var TiedEmbeddings: bool

    /// Read is a checkpoint's llama config, asked in config.json's keys.
    public static func Read(_ ck: model.Checkpoint) throws -> Config {
        ck.Aliases = decoder.Aliases
        if !Names.contains(ck.Architecture) {
            throw LoadError.unsupported("\(ck.Architecture) is not a llama")
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
        var cfg = Config(
            Layers: try need("num_hidden_layers"),
            Dim: dim,
            Hidden: try need("intermediate_size"),
            Heads: heads,
            KVHeads: c.Int("num_key_value_heads") ?? heads,
            HeadDim: c.Int("head_dim") ?? dim / heads,
            Vocab: c.Int("vocab_size") ?? 0,
            Context: c.Int("max_position_embeddings") ?? 2048,
            Eps: float32(c.Double("rms_norm_eps") ?? 1e-5),
            RopeBase: float32(c.Double("rope_theta") ?? 10000),
            TiedEmbeddings: false)
        if ck.Format == .gguf {
            cfg.TiedEmbeddings = !ck.Has("lm_head.weight")
            if let rope = c.Int("rope_dimension_count", gguf: "rope.dimension_count"), rope != cfg.HeadDim {
                throw LoadError.unsupported("rotary embeddings over \(rope) of a head's \(cfg.HeadDim) dimensions")
            }
            if let experts = c.Int("num_experts"), experts > 0 {
                throw LoadError.unsupported("a mixture of \(experts) experts (Mixtral's) is not run yet")
            }
            if cfg.Vocab == 0, let t = ck.Info("model.embed_tokens.weight") {
                cfg.Vocab = t.Shape[0]
            }
        } else {
            cfg.TiedEmbeddings = c.Bool("tie_word_embeddings") ?? !ck.Has("lm_head.weight")
            if let scaling = c.Raw?["rope_scaling"], !scaling.IsNull {
                let kind = scaling["rope_type"]?.String ?? scaling["type"]?.String ?? "?"
                if kind != "default" {
                    throw LoadError.unsupported("rope_scaling '\(kind)' (Llama 3.1's) is not run yet")
                }
            }
            if let act = c.String("hidden_act"), act != "silu" {
                throw LoadError.unsupported("a '\(act)' MLP; llama's is silu")
            }
            if c.Bool("attention_bias") == true || c.Bool("mlp_bias") == true {
                throw LoadError.unsupported("biases in attention or the MLP")
            }
        }
        return cfg
    }
}

/// Network is a llama's network: nn/decoder's.
public typealias Network = decoder.Network

/// Load reads a llama checkpoint onto device d. Weights stay in the
/// file's format -- f32, f16, bf16 or quantized -- and where the device
/// reads the file's mapping in place, nothing is copied.
public func Load(_ ck: model.Checkpoint, on d: gpu.Device) async throws -> decoder.Network {
    let c = try Config.Read(ck)
    // llama.cpp permutes Q and K for interleaved RoPE when it writes a
    // llama GGUF; Hugging Face's weights are in halves.
    let rope = nn.Rope(base: c.RopeBase, layout: ck.Format == .gguf ? neural.RopeLayout.interleaved : neural.RopeLayout.halves)
    let w = decoder.Weights(ck, on: d)
    var factors: [float32]? = nil
    if let t = ck.Info("rope_freqs.weight", gguf: "rope_freqs.weight") {
        factors = try decoder.Floats(ck, t)
    }
    // One table of the rotation's cosines and sines for every layer, its
    // frequencies divided by Llama 3's factors where the checkpoint has
    // them.
    let table = try await neural.RopeTable.Make(on: d, dim: c.HeadDim, positions: c.Context, base: c.RopeBase, factors: factors)
    let embed = try nn.Embedding(try await w.Tensor("model.embed_tokens.weight"))
    var blocks: [decoder.Block] = []
    for i in 0..<c.Layers {
        let p = "model.layers.\(i)."
        let g = "blk.\(i)."
        let attention = try nn.Attention(
            qkv: try nn.Stacked([try await w.Linear(p + "self_attn.q_proj", g + "attn_q"),
                                 try await w.Linear(p + "self_attn.k_proj", g + "attn_k"),
                                 try await w.Linear(p + "self_attn.v_proj", g + "attn_v")]),
            o: try await w.Linear(p + "self_attn.o_proj", g + "attn_output"),
            heads: c.Heads, kvHeads: c.KVHeads, headDim: c.HeadDim, rope: rope)
        attention.Table = table
        let mlp = try nn.GatedMLP.Fused(
            gate: try await w.Linear(p + "mlp.gate_proj", g + "ffn_gate", biasFirst: true),
            up: try await w.Linear(p + "mlp.up_proj", g + "ffn_up", biasFirst: true),
            down: try await w.Linear(p + "mlp.down_proj", g + "ffn_down", biasFirst: true),
            activation: .SiLU)
        blocks.append(decoder.Standard(attnNorm: try await w.RMS(p + "input_layernorm", c.Eps), attention: attention,
                                       ffnNorm: try await w.RMS(p + "post_attention_layernorm", c.Eps), ffn: mlp))
    }
    let output = c.TiedEmbeddings ? try nn.Linear(embed.Weight) : try await w.Linear("lm_head", "output")
    return decoder.Network(on: d, dim: c.Dim, context: c.Context, embed: embed, blocks: blocks, kvHeads: c.KVHeads,
                           headDim: c.HeadDim, norm: try await w.RMS("model.norm", c.Eps), output: output)
}
