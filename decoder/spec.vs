package decoder

import (
    "gpu"
    "gpu/neural"
    "math"
    "model"
    "nn"
)

/// RopeKind is how a network turns queries and keys, llama.cpp's
/// LLAMA_ROPE_TYPE: pairs of neighbours (NORM, which llama.cpp's converter
/// permutes Q and K into), halves of a head (NEOX), or not at all.
public enum RopeKind: Equatable {
    case norm
    case neox
    case none
}

/// QKNormKind is where a network normalizes Q's and K's heads: not at all,
/// before they turn (Qwen3's), or after (Maincoder's).
public enum QKNormKind: Equatable {
    case none
    case beforeRope
    case afterRope
}

/// LogitScaleKind is what a GGUF's {arch}.logit_scale does to the logits,
/// which llama.cpp's graphs disagree about: Granite divides by it, Command
/// R and Talkie multiply.
public enum LogitScaleKind: Equatable {
    case none
    case multiply
    case divide
}

/// Spec is what a decoder network of decoder.Standard layers does, in the
/// terms llama.cpp's graphs differ in: which norms it has and of what
/// kind, which optional biases and scales its graph reads (llama.cpp loads
/// a projection's `.bias` and `.scale` where a file has them, and each
/// graph applies those it names), where it normalizes Q and K, how it
/// turns them, and its MLP. A family that is Standard layers is a Spec and
/// its names; Build reads a GGUF by it.
public struct Spec {
    public var Rope: RopeKind = .norm
    /// RopeFactors is whether the graph reads rope_freqs (Llama 3's).
    public var RopeFactors: bool = false
    /// NoRopeEvery is n where every n-th layer (1-based) does not turn
    /// (SmolLM3's 4); 0 for none.
    public var NoRopeEvery: int = 0
    public var Norm: nn.NormKind = .rms
    /// NormBias is whether the block and final norms read a bias.
    public var NormBias: bool = false
    /// NormWeights is whether the norms have weights at all (OLMo's do not).
    public var NormWeights: bool = true
    /// PreNorms is whether a layer normalizes the stream before attention
    /// and the MLP (OLMo 2 only after).
    public var PreNorms: bool = true
    /// FFNNorm is the GGUF name of the norm before the MLP; nil where the
    /// MLP of a parallel layer reads attention's norm (Phi-2's, Command
    /// R's).
    public var FFNNorm: string? = "ffn_norm"
    /// PostAttnNorm and PostFFNNorm are the GGUF names of norms after each
    /// branch (OLMo 2's, Gemma 2's), or nil.
    public var PostAttnNorm: string? = nil
    public var PostFFNNorm: string? = nil
    public var QKNorm: QKNormKind = .none
    /// QKNormKind is the Q and K norms' kind, and QKNormEach whether they
    /// normalize each head by itself where their weights are all heads'.
    public var QKNormKind: nn.NormKind = .rms
    public var QKNormEach: bool = false
    /// EmbedSqrtDim is whether the embedding is scaled by √width (Gemma's).
    public var EmbedSqrtDim: bool = false
    public var AttnOutBias: bool = true
    public var AttnOutScale: bool = true
    /// FFNGated is whether the MLP is gated -- a(gate)·up, then down --
    /// or plain -- a(up), then down -- and FFNActivation its activation.
    public var FFNGated: bool = true
    /// FFNFused is whether a gated MLP's gate and up are one weight,
    /// ffn_up, its first half the gate (llama.cpp's LLM_FFN_SWIGLU: ChatGLM's,
    /// GLM-4's, Phi-3's).
    public var FFNFused: bool = false
    public var FFNActivation: neural.Activation = .SiLU
    public var FFNBias: bool = false
    public var FFNScale: bool = false
    public var OutputBias: bool = false
    public var OutputScale: bool = true
    /// AttentionScale is whether the graph reads {arch}.attention.scale for
    /// the scores' scale (else 1/√head).
    public var AttentionScale: bool = false
    /// ResidualScale is whether the graph scales each branch by
    /// {arch}.residual_scale before adding it (Granite's).
    public var ResidualScale: bool = false
    public var LogitScale: LogitScaleKind = .none
    public var Parallel: bool = false
    /// SWA is whether the graph reads {arch}.attention.sliding_window: layers
    /// that see only that many keys back, which the file's
    /// sliding_window_pattern names -- an array, one a layer, or n, every
    /// n-th layer full -- or else SWAPattern does; SWADenseFirst is
    /// llama.cpp's dense_first (layer 0 full, rather than the n-th).
    public var SWA: bool = false
    public var SWAPattern: int = 0
    public var SWADenseFirst: bool = false

    public init() {}
}

/// Hparams are a GGUF's shape for a decoder, read by llama.cpp's key names
/// under the architecture.
public struct Hparams {
    public var Layers: int
    public var Dim: int
    public var Heads: int
    public var KVHeads: int
    public var HeadDim: int
    public var RotaryDims: int
    public var Context: int
    public var Eps: float32
    public var RopeBase: float32

    /// Read is the checkpoint's decoder shape.
    public static func Read(_ ck: model.Checkpoint, norm: nn.NormKind = .rms) throws -> Hparams {
        ck.Aliases = Aliases
        let c = ck.Config
        func need(_ key: string) throws -> int {
            guard let n = c.Int(key) else {
                throw DecoderError.invalid("no \(key) in the config")
            }
            return n
        }
        let dim = try need("hidden_size")
        let heads = try need("num_attention_heads")
        let headDim = c.Int("head_dim") ?? dim / heads
        var eps = float32(c.Double("rms_norm_eps") ?? 1e-5)
        if norm == .layer {
            eps = float32(c.Double("layer_norm_eps") ?? 1e-5)
        }
        return Hparams(
            Layers: try need("num_hidden_layers"),
            Dim: dim,
            Heads: heads,
            KVHeads: c.Int("num_key_value_heads") ?? heads,
            HeadDim: headDim,
            RotaryDims: c.Int("rope_dimension_count", gguf: "rope.dimension_count") ?? headDim,
            Context: c.Int("max_position_embeddings") ?? 4096,
            Eps: eps,
            RopeBase: float32(c.Double("rope_theta") ?? 10000))
    }
}

/// Key is a float under the checkpoint's architecture in a GGUF
/// ({arch}.name), or 0 where it is not there.
public func Key(_ ck: model.Checkpoint, _ name: string) -> float32 {
    return float32(ck.Config.Double(name, gguf: name) ?? 0)
}

/// Build reads a GGUF of Standard layers onto device d, by its Spec.
public func Build(_ ck: model.Checkpoint, on d: gpu.Device, _ s: Spec) async throws -> Network {
    let h = try Hparams.Read(ck, norm: s.Norm)
    let w = Weights(ck, on: d)
    if let experts = ck.Config.Int("num_experts"), experts > 0 {
        throw DecoderError.invalid("a mixture of \(experts) experts is not run yet")
    }
    var factors: [float32]? = nil
    if s.RopeFactors, let t = ck.Info("rope_freqs.weight", gguf: "rope_freqs.weight") {
        factors = try Floats(ck, t)
    }
    var table: neural.RopeTable? = nil
    if s.Rope != .none {
        table = try await neural.RopeTable.Make(on: d, dim: h.RotaryDims, positions: h.Context, base: h.RopeBase, factors: factors)
    }
    let layout = s.Rope == .neox ? neural.RopeLayout.halves : neural.RopeLayout.interleaved
    let scale = s.AttentionScale ? Key(ck, "attention.scale") : float32(0)
    let residual = s.ResidualScale ? Key(ck, "residual_scale") : float32(0)
    let swa = SlidingLayers(ck, h.Layers, s)
    var swaTable = table
    let swaBase = float32(ck.Config.Double("rope.freq_base_swa", gguf: "rope.freq_base_swa") ?? float64(h.RopeBase))
    if s.Rope != .none && swaBase != h.RopeBase {
        swaTable = try await neural.RopeTable.Make(on: d, dim: h.RotaryDims, positions: h.Context, base: swaBase, factors: factors)
    }
    let embed = try nn.Embedding(try await w.Tensor("token_embd.weight", "token_embd.weight"))
    var blocks: [Block] = []
    for i in 0..<h.Layers {
        let g = "blk.\(i)."
        let noRope = s.Rope == .none || (s.NoRopeEvery > 0 && (i + 1) % s.NoRopeEvery == 0)
        let sliding = swa.window > 0 && swa.layers[i]
        let rope = nn.Rope(base: sliding ? swaBase : h.RopeBase, layout: layout,
                           dims: noRope ? -1 : (h.RotaryDims == h.HeadDim ? 0 : h.RotaryDims))
        let attention = try nn.Attention(
            qkv: try await QKV(w, g, g),
            o: try await w.Linear(g + "attn_output", g + "attn_output", bias: s.AttnOutBias, scale: s.AttnOutScale),
            heads: h.Heads, kvHeads: h.KVHeads, headDim: h.HeadDim, rope: rope)
        attention.Table = sliding ? swaTable : table
        attention.Scale = scale
        if sliding {
            attention.Window = swa.window
        }
        if s.QKNorm != .none {
            attention.QNorm = nn.HeadNorm(try await w.Norm(s.QKNormKind, g + "attn_q_norm", h.Eps, gg: g + "attn_q_norm"), each: s.QKNormEach)
            attention.KNorm = nn.HeadNorm(try await w.Norm(s.QKNormKind, g + "attn_k_norm", h.Eps, gg: g + "attn_k_norm"), each: s.QKNormEach)
            attention.NormAfterRope = s.QKNorm == .afterRope
        }
        let ffn = try await _specFFN(w, s, g)
        let b = Standard(attnNorm: s.PreNorms ? try await _specNorm(w, s, h, d, g + "attn_norm") : nil, attention: attention,
                         ffnNorm: s.PreNorms && s.FFNNorm != nil ? try await _specNorm(w, s, h, d, g + s.FFNNorm!) : nil, ffn: ffn)
        if let p = s.PostAttnNorm {
            b.PostAttnNorm = try await _specNorm(w, s, h, d, g + p)
        }
        if let p = s.PostFFNNorm {
            b.PostFFNNorm = try await _specNorm(w, s, h, d, g + p)
        }
        b.Parallel = s.Parallel
        if residual != 0 {
            b.ResidualScale = residual
        }
        blocks.append(b)
    }
    let tied = !w.Has("output.weight", "output.weight")
    let output = tied ? try nn.Linear(embed.Weight)
                      : try await w.Linear("output", "output", bias: s.OutputBias, scale: s.OutputScale)
    let n = Network(on: d, dim: h.Dim, context: h.Context, embed: embed, blocks: blocks, kvHeads: h.KVHeads, headDim: h.HeadDim,
                    norm: try await _specNorm(w, s, h, d, "output_norm"), output: output)
    let e = Key(ck, "embedding_scale")
    if e != 0 {
        n.EmbedScale = e
    }
    if s.EmbedSqrtDim {
        n.EmbedScale = math.Sqrt(float32(h.Dim))
    }
    let l = Key(ck, "logit_scale")
    if l != 0 {
        switch s.LogitScale {
        case .multiply: n.LogitScale = l
        case .divide: n.LogitScale = 1 / l
        case .none: break
        }
    }
    return n
}

// _specNorm is the norm a Spec's network has at base.
func _specNorm(_ w: Weights, _ s: Spec, _ h: Hparams, _ d: gpu.Device, _ base: string) async throws -> nn.Norm {
    if !s.NormWeights {
        return try await nn.Norm.Plain(s.Norm, count: h.Dim, on: d, eps: h.Eps)
    }
    let weight = try await w.Floats(base + ".weight", base + ".weight")
    var bias: gpu.Buffer<float32>? = nil
    if s.NormBias && w.Has(base + ".bias", base + ".bias") {
        bias = try await w.Floats(base + ".bias", base + ".bias")
    }
    return nn.Norm(s.Norm, weight: weight, bias: bias, eps: h.Eps)
}

// _specFFN is layer g's MLP by a Spec.
func _specFFN(_ w: Weights, _ s: Spec, _ g: string) async throws -> nn.FFN {
    let a = s.FFNActivation
    let down = try await w.Linear(g + "ffn_down", g + "ffn_down", biasFirst: true, bias: s.FFNBias, scale: s.FFNScale)
    let up = try await w.Linear(g + "ffn_up", g + "ffn_up", biasFirst: true, bias: s.FFNBias, scale: s.FFNScale)
    if s.FFNGated && s.FFNFused {
        return try nn.GatedMLP(gateUp: [up], down: down, activation: a)
    }
    if !s.FFNGated {
        return try nn.MLP(up: up, down: down, activation: a)
    }
    let gate = try await w.Linear(g + "ffn_gate", g + "ffn_gate", biasFirst: true, bias: s.FFNBias, scale: s.FFNScale)
    return try nn.GatedMLP.Fused(gate: gate, up: up, down: down, activation: a)
}

/// SlidingLayers is which of a network's layers attend through a sliding
/// window, and its width (0: none), as llama.cpp's load_swa_pattern reads
/// them for a Spec.
public func SlidingLayers(_ ck: model.Checkpoint, _ layers: int, _ s: Spec) -> (window: int, layers: [bool]) {
    var out = [bool](repeating: false, count: layers)
    if !s.SWA {
        return (window: 0, layers: out)
    }
    guard let window = ck.Config.Int("sliding_window", gguf: "attention.sliding_window"), window > 0 else {
        return (window: 0, layers: out)
    }
    let key = "attention.sliding_window_pattern"
    if let a = ck.Config.Ints(key, gguf: key), a.count == layers && layers > 1 {
        for i in 0..<layers {
            out[i] = a[i] != 0
        }
        return (window: window, layers: out)
    }
    var n = s.SWAPattern
    if let v = ck.Config.Int(key, gguf: key) {
        n = v
    }
    for i in 0..<layers {
        out[i] = s.SWADenseFirst ? (n == 0 || i % n != 0) : (n == 0 || i % n < n - 1)
    }
    return (window: window, layers: out)
}

/// MRope is whether a GGUF's rotary embeddings are multimodal (M-RoPE): its
/// rope.dimension_sections set, as llama.cpp's use_mrope asks. For text,
/// every section's position is the token's, so M-RoPE turns as NeoX does.
public func MRope(_ ck: model.Checkpoint) -> bool {
    let key = "rope.dimension_sections"
    guard let s = ck.Config.Ints(key, gguf: key), s.count >= 2 else {
        return false
    }
    return s[0] > 0 && s[1] > 0
}
