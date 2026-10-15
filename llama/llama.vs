// Package llama is the Llama network: Llama 1 to 3, and the models that
// share its architecture (LlamaForCausalLM on the Hugging Face Hub,
// general.architecture "llama" in GGUF: TinyLlama, SmolLM2, Mistral 7B,
// most fine-tunes) -- an embedding, blocks of RMSNorm, grouped-query
// attention with rotary position embeddings, RMSNorm and a SwiGLU MLP,
// each added back to what went in, then a final RMSNorm and the head.
//
// It is the network alone, over a batch of rows and a memory of
// sequences, so that every task that runs a Llama shares it: llm/llama
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

/// Block is one transformer layer.
public final class Block {
    public let AttnNorm: nn.RMSNorm
    public let Attention: nn.Attention
    public let FFNNorm: nn.RMSNorm
    public let MLP: nn.GatedMLP

    init(attnNorm: nn.RMSNorm, attention: nn.Attention, ffnNorm: nn.RMSNorm, mlp: nn.GatedMLP) {
        self.AttnNorm = attnNorm
        self.Attention = attention
        self.FFNNorm = ffnNorm
        self.MLP = mlp
    }
}

/// Input is what a batch's rows start from: tokens, or embeddings that
/// stand in for them (an image's or audio's, from an adapter).
public enum Input {
    case tokens([int32])
    case embeddings(gpu.Buffer<float32>)
}

/// Network is a llama's weights on a device, and the buffers it decodes
/// with.
public final class Network {
    public let Config: Config
    public let Device: gpu.Device
    public let Embed: nn.Embedding
    public let Blocks: [Block]
    public let Norm: nn.RMSNorm
    public let Output: nn.Linear
    // One row's buffers.
    let _x: gpu.Buffer<float32>
    let _h: gpu.Buffer<float32>
    let _logits: gpu.Buffer<float32>
    // A batch's, grown to the most rows seen.
    var _rows = 0
    var _bx: gpu.Buffer<float32>? = nil
    var _bh: gpu.Buffer<float32>? = nil
    var _btokens: gpu.Buffer<int32>? = nil
    var _bout: gpu.Buffer<float32>? = nil

    init(_ config: Config, _ d: gpu.Device, _ embed: nn.Embedding, _ blocks: [Block], _ norm: nn.RMSNorm, _ output: nn.Linear) throws {
        self.Config = config
        self.Device = d
        self.Embed = embed
        self.Blocks = blocks
        self.Norm = norm
        self.Output = output
        self._x = try d.CreateBuffer(of: float32.self, count: config.Dim)
        self._h = try d.CreateBuffer(of: float32.self, count: config.Dim)
        self._logits = try d.CreateBuffer(of: float32.self, count: output.Out)
    }

    /// Vocab is how many logits a row has.
    public var Vocab: int { return Output.Out }

    /// Load reads a llama checkpoint onto device d. Weights stay in the
    /// file's format -- f32, f16, bf16 or quantized -- and where the device
    /// reads the file's mapping in place, nothing is copied.
    public static func Load(_ ck: model.Checkpoint, on d: gpu.Device) async throws -> Network {
        let c = try llama.Config.Read(ck)
        // llama.cpp permutes Q and K for interleaved RoPE when it writes a
        // llama GGUF; Hugging Face's weights are in halves.
        let rope = nn.Rope(base: c.RopeBase, layout: ck.Format == .gguf ? neural.RopeLayout.interleaved : neural.RopeLayout.halves)
        func weight(_ name: string) async throws -> tensor.Tensor {
            return try await ck.Tensor(name, on: d)
        }
        func norm(_ name: string) async throws -> nn.RMSNorm {
            return nn.RMSNorm(try await ck.Float32(name, on: d), eps: c.Eps)
        }
        // A projection: its weight, and the bias and scale a checkpoint may
        // give it -- hf's ".bias", and llama.cpp's ".scale" (gg names it).
        func linear(_ hf: string, _ gg: string, biasFirst: bool = false) async throws -> nn.Linear {
            var bias: gpu.Buffer<float32>? = nil
            if ck.Has(hf + ".bias") {
                bias = try (try await ck.Float32(hf + ".bias", on: d)).Floats()
            }
            var scale: float32 = 1
            if ck.Format == .gguf, let t = ck.Info(hf + ".scale", gguf: gg + ".scale") {
                scale = try decoder.Scalar(ck, t)
            }
            return try nn.Linear(try await weight(hf + ".weight"), bias: bias, scale: scale, biasFirst: biasFirst)
        }
        let embed = try nn.Embedding(try await weight("model.embed_tokens.weight"))
        var blocks: [Block] = []
        for i in 0..<c.Layers {
            let p = "model.layers.\(i)."
            let g = "blk.\(i)."
            let attention = try nn.Attention(
                qkv: try nn.Stacked([try await linear(p + "self_attn.q_proj", g + "attn_q"),
                                     try await linear(p + "self_attn.k_proj", g + "attn_k"),
                                     try await linear(p + "self_attn.v_proj", g + "attn_v")]),
                o: try await linear(p + "self_attn.o_proj", g + "attn_output"),
                heads: c.Heads, kvHeads: c.KVHeads, headDim: c.HeadDim, rope: rope)
            let mlp = try nn.GatedMLP.Fused(
                gate: try await linear(p + "mlp.gate_proj", g + "ffn_gate", biasFirst: true),
                up: try await linear(p + "mlp.up_proj", g + "ffn_up", biasFirst: true),
                down: try await linear(p + "mlp.down_proj", g + "ffn_down", biasFirst: true),
                activation: .SiLU)
            blocks.append(Block(attnNorm: try await norm(p + "input_layernorm.weight"), attention: attention,
                                ffnNorm: try await norm(p + "post_attention_layernorm.weight"), mlp: mlp))
        }
        if c.HeadDim % 2 == 0 {
            // One table of the rotation's cosines and sines for every layer,
            // its frequencies divided by Llama 3's factors where the
            // checkpoint has them.
            var factors: [float32]? = nil
            if let t = ck.Info("rope_freqs.weight", gguf: "rope_freqs.weight") {
                factors = try decoder.Floats(ck, t)
            }
            let table = try await neural.RopeTable.Make(on: d, dim: c.HeadDim, positions: c.Context, base: c.RopeBase, factors: factors)
            for b in blocks {
                b.Attention.Table = table
            }
        }
        let final = try await norm("model.norm.weight")
        let output = c.TiedEmbeddings ? try nn.Linear(embed.Weight) : try await linear("lm_head", "output")
        return try Network(c, d, embed, blocks, final, output)
    }

    /// Memory is a memory for this network: sequences sequences of
    /// capacity positions each (the model's context when nil).
    public func Memory(sequences: int = 1, capacity: int? = nil) -> cache.KV {
        return cache.KV(on: Device, layers: Config.Layers, kvHeads: Config.KVHeads, headDim: Config.HeadDim,
                        capacity: capacity ?? Config.Context, sequences: sequences)
    }

    /// Forward runs a batch through the network: input's rows, in runs of
    /// sequences, each added to its sequence's memory (dropping what the
    /// sequence held from the run's position on). It is the wanted rows'
    /// outputs, one after another: logits, or with hidden the final
    /// normalized hidden states. The buffer is the network's, good until
    /// the next Forward or Step.
    public func Forward(_ input: Input, runs: [cache.Run], wanted: [int], hidden: bool = false,
                        memory: cache.KV) async throws -> gpu.Buffer<float32> {
        var n = 0
        for r in runs {
            if r.Row != n {
                throw LoadError.unsupported("runs that do not cover the batch's rows in order")
            }
            if r.Position > memory.Length(r.Sequence) {
                throw LoadError.unsupported("sequence \(r.Sequence) at position \(r.Position) past the \(memory.Length(r.Sequence)) it holds")
            }
            if r.Position + r.Count > memory.Capacity {
                throw LoadError.unsupported("sequence \(r.Sequence) past a memory of \(memory.Capacity) positions")
            }
            n += r.Count
        }
        for w in wanted where w < 0 || w >= n {
            throw LoadError.unsupported("row \(w) wanted of a batch of \(n)")
        }
        let dim = Config.Dim
        switch input {
        case .tokens(let t):
            if t.count != n { throw LoadError.unsupported("\(t.count) tokens for \(n) rows") }
        case .embeddings(let e):
            if e.count < n * dim { throw LoadError.unsupported("embeddings of \(e.count) floats for \(n) rows of \(dim)") }
        }
        if n == 0 {
            return _logits.Slice(from: 0, count: 0)
        }
        for r in runs {
            memory.Truncate(r.Sequence, from: r.Position)
        }
        if n == 1 && wanted == [0] {
            // One row: matrix-vector products, and attention's decoding kernel.
            switch input {
            case .tokens(let t): try await Embed.Lookup(int(t[0]), into: _x)
            case .embeddings(let e): try await _x.Copy(from: e.Slice(from: 0, count: dim))
            }
            let r = runs[0]
            for i in 0..<Blocks.count {
                let b = Blocks[i]
                try await b.AttnNorm.Forward(_x, into: _h)
                try await b.Attention.Forward(_h, position: r.Position, cache: try memory.Layer(i, sequence: r.Sequence), into: _x, accumulate: true)
                try await b.FFNNorm.Forward(_x, into: _h)
                try await b.MLP.Forward(_h, into: _x, accumulate: true)
            }
            try memory.Wrote(r.Sequence, through: r.Position + 1)
            try await Norm.Forward(_x, into: _h)
            if hidden {
                return _h
            }
            try await Output.Forward(_h, into: _logits)
            return _logits
        }
        if _rows < n {
            try _grow(n)
        }
        let x = _bx!.Slice(from: 0, count: n * dim), h = _bh!.Slice(from: 0, count: n * dim)
        switch input {
        case .tokens(let t):
            let tokens = _btokens!.Slice(from: 0, count: n)
            try await tokens.Upload(t)
            try await Embed.Lookup(tokens: tokens, first: 0, count: n, into: x)
        case .embeddings(let e):
            try await x.Copy(from: e.Slice(from: 0, count: n * dim))
        }
        for i in 0..<Blocks.count {
            let b = Blocks[i]
            var nr: [nn.Run] = []
            for r in runs {
                nr.append(nn.Run(row: r.Row, count: r.Count, position: r.Position, cache: try memory.Layer(i, sequence: r.Sequence)))
            }
            try await b.AttnNorm.Forward(x, rows: n, into: h)
            try await b.Attention.Forward(h, rows: n, runs: nr, into: x, accumulate: true)
            try await b.FFNNorm.Forward(x, rows: n, into: h)
            try await b.MLP.Forward(h, rows: n, into: x, accumulate: true)
        }
        for r in runs {
            try memory.Wrote(r.Sequence, through: r.Position + r.Count)
        }
        // The wanted rows, gathered to the front of h, normalized in x.
        let w = wanted.count
        var all = w == n
        for j in 0..<w where wanted[j] != j {
            all = false
        }
        if !all {
            for j in 0..<w {
                try await h.Slice(from: j * dim, count: dim).Copy(from: x.Slice(from: wanted[j] * dim, count: dim))
            }
        }
        let rows = all ? x : h
        let normed = all ? h : x
        try await Norm.Forward(rows.Slice(from: 0, count: w * dim), rows: w, into: normed.Slice(from: 0, count: w * dim))
        if hidden {
            return normed.Slice(from: 0, count: w * dim)
        }
        if _bout == nil || _bout!.count < w * Vocab {
            try _growOut(w)
        }
        let out = _bout!.Slice(from: 0, count: w * Vocab)
        try await Output.Forward(normed.Slice(from: 0, count: w * dim), rows: w, into: out)
        return out
    }

    // _grow and _growOut make a batch's buffers room for n rows. They are
    // apart from Forward because the compiler's SIL verifier rejects
    // CreateBuffer specialized inside a branch of an async function.
    func _grow(_ n: int) throws {
        _bx = try Device.CreateBuffer(of: float32.self, count: n * Config.Dim)
        _bh = try Device.CreateBuffer(of: float32.self, count: n * Config.Dim)
        // A float's bytes viewed as ints: CreateBuffer(of: int32.self)
        // specialized here fails the same verifier.
        _btokens = try Device.CreateBuffer(of: float32.self, count: n).View(as: int32.self)
        _rows = n
    }

    /// Tokens is a buffer of count token ids on the network's device: where
    /// a decoding loop the device runs keeps its sequence.
    public func Tokens(count: int) throws -> gpu.Buffer<int32> {
        // A float's bytes viewed as ints: CreateBuffer(of: int32.self)
        // specialized here fails the compiler's SIL verifier.
        return try Device.CreateBuffer(of: float32.self, count: count).View(as: int32.self)
    }

    func _growOut(_ w: int) throws {
        _bout = try Device.CreateBuffer(of: float32.self, count: max(w, 1) * Vocab)
    }

    /// Step runs one token through the network that the device chose:
    /// tokens[index], at position of sequence -- greedy decoding's next
    /// token, looked up with nothing read back -- and is its logits.
    public func Step(tokens: gpu.Buffer<int32>, at index: int, position: int, sequence: int = 0,
                     memory: cache.KV) async throws -> gpu.Buffer<float32> {
        if position > memory.Length(sequence) || position >= memory.Capacity {
            throw LoadError.unsupported("sequence \(sequence) at position \(position) of \(memory.Length(sequence)) held")
        }
        memory.Truncate(sequence, from: position)
        try await Embed.Lookup(tokens: tokens, at: index, into: _x)
        for i in 0..<Blocks.count {
            let b = Blocks[i]
            try await b.AttnNorm.Forward(_x, into: _h)
            try await b.Attention.Forward(_h, position: position, cache: try memory.Layer(i, sequence: sequence), into: _x, accumulate: true)
            try await b.FFNNorm.Forward(_x, into: _h)
            try await b.MLP.Forward(_h, into: _x, accumulate: true)
        }
        try memory.Wrote(sequence, through: position + 1)
        try await Norm.Forward(_x, into: _h)
        try await Output.Forward(_h, into: _logits)
        return _logits
    }
}
