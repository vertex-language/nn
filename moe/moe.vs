// Package moe is a mixture of experts: a router scores each token against
// every expert, the best few are chosen, and the token's output is their
// outputs weighted -- llama.cpp's build_moe_ffn, step for step, with the
// choice made on the device so that nothing is read back.
//
// Experts' weights are a GGUF's ffn_{gate,up,down}_exps tensors as they
// are: each expert's matrix one after another. A family adds what is its
// own around this -- a shared expert, a gate on it, dense layers before
// the sparse ones.
package moe

import (
    "gpu"
    "gpu/dtype"
    "gpu/linalg"
    "gpu/neural"
    "gpu/parallel"
    "math"
    "nn"
    "tensor"
)

/// Gating is how a router's logits become the experts' probabilities:
/// llama.cpp's llama_expert_gating_func_type.
public enum Gating: Equatable {
    /// Softmax over all experts (Mixtral's, Qwen's).
    case softmax
    /// A sigmoid each (DeepSeek V3's, GLM-4.5's).
    case sigmoid
    /// The logits as they are, then a softmax over the chosen ones only.
    case softmaxWeight
    /// √softplus each (DeepSeek V4's).
    case sqrtSoftplus

    public var _code: int32 {
        switch self {
        case .softmax: return 0
        case .sigmoid: return 1
        case .softmaxWeight: return 2
        case .sqrtSoftplus: return 3
        }
    }

    /// From is the gating llama.cpp's {arch}.expert_gating_func names (1
    /// softmax, 2 sigmoid, 3 softmax of the chosen weights, 4 √softplus),
    /// or fallback for 0, absent.
    public static func From(_ code: int, fallback: Gating) -> Gating {
        switch code {
        case 1: return .softmax
        case 2: return .sigmoid
        case 3: return .softmaxWeight
        case 4: return .sqrtSoftplus
        default: return fallback
        }
    }
}

/// MoEError is a mixture this cannot run.
public enum MoEError: Error, CustomStringConvertible {
    case unsupported(string)

    public var description: string {
        switch self {
        case .unsupported(let why): return "moe: " + why
        }
    }
}

func _route(_ logits: gpu.Span<float32>, _ bias: gpu.Span<float32>, _ ids: gpu.MutableSpan<int32>, _ weights: gpu.MutableSpan<float32>,
            _ experts: int, _ used: int, _ gating: int32, _ hasBias: bool, _ norm: bool, _ scale: float32) kernel {
    // A token a workgroup: its experts' probabilities from its logits, the
    // `used` best by probability plus the selection bias (DeepSeek V3's
    // correction, which the weights themselves leave out), and their
    // weights -- each expert's probability; softmaxed over the chosen
    // where the gating says; divided by their sum where norm says; times
    // scale. The best first, the lower expert of equals.
    let probs = gpu.Shared<float32>(count: 1024)
    let sel = gpu.Shared<float32>(count: 1024)
    let t = gpu.GroupIndex.x
    let me = parallel.GroupRank()
    let step = parallel.GroupCount()
    let base = t * experts
    var m = -float32.infinity
    var j = me
    while j < experts {
        m = math.Max(m, logits[base + j])
        j += step
    }
    let most = parallel.GroupMax(m)
    var s: float32 = 0
    j = me
    while j < experts {
        s = s + math.Exp(logits[base + j] - most)
        j += step
    }
    let total = parallel.GroupSum(s)
    j = me
    while j < experts {
        let l = logits[base + j]
        var p = l
        if gating == 0 { p = math.Exp(l - most) / total }
        if gating == 1 { p = math.Sigmoid(l) }
        if gating == 3 { p = math.Sqrt(l > 20 ? l : math.Log(1 + math.Exp(l))) }
        probs[j] = p
        sel[j] = hasBias ? p + bias[j] : p
        j += step
    }
    gpu.Barrier()
    var picked: float32 = 0
    var c = 0
    while c < used {
        var best = -float32.infinity
        j = me
        while j < experts {
            best = math.Max(best, sel[j])
            j += step
        }
        let top = parallel.GroupMax(best)
        var first = float32(experts)
        j = me
        while j < experts {
            if sel[j] == top && float32(j) < first { first = float32(j) }
            j += step
        }
        let e = int(parallel.GroupMin(first))
        if me == 0 {
            ids[t * used + c] = int32(e)
            weights[t * used + c] = probs[e]
            sel[e] = -float32.infinity
        }
        gpu.Barrier()
        c += 1
    }
    if me == 0 {
        if gating == 2 {
            var mx = -float32.infinity
            c = 0
            while c < used { mx = math.Max(mx, weights[t * used + c]); c += 1 }
            var sum: float32 = 0
            c = 0
            while c < used { let v = math.Exp(weights[t * used + c] - mx); weights[t * used + c] = v; sum = sum + v; c += 1 }
            c = 0
            while c < used { weights[t * used + c] = weights[t * used + c] / sum; c += 1 }
        }
        if norm {
            var sum: float32 = 0
            c = 0
            while c < used { sum = sum + weights[t * used + c]; c += 1 }
            if sum < 6.103515625e-5 { sum = 6.103515625e-5 }
            c = 0
            while c < used { weights[t * used + c] = weights[t * used + c] / sum; c += 1 }
        }
        if scale != 0 && scale != 1 {
            c = 0
            while c < used { weights[t * used + c] = weights[t * used + c] * scale; c += 1 }
        }
        picked = 1
    }
}

func _expertScale(_ y: gpu.MutableSpan<float32>, _ ids: gpu.Span<int32>, _ s: gpu.Span<float32>, _ m: int, _ items: int) kernel {
    // Item i's row times its expert's scale: llama.cpp's *_exps_s.
    let r = gpu.Index.x
    let i = gpu.Index.y
    if r < m && i < items {
        y[i * m + r] = y[i * m + r] * s[int(ids[i])]
    }
}

func _gatedExperts(_ g: gpu.MutableSpan<float32>, _ u: gpu.Span<float32>, _ a: int32, _ gated: bool, _ limit: float32, _ n: int) kernel {
    // a(gate)·up, gate and up each item's row, into g -- or a(up) alone --
    // with up clamped to ±limit and a(gate) to limit where limit is set
    // (llama.cpp's swiglu_clamp_exp).
    let i = gpu.Index.x
    if i < n {
        if gated {
            var up = u[i]
            var act = neural.Apply(a, g[i])
            if limit > 0 {
                up = math.Max(-limit, math.Min(limit, up))
                act = math.Min(limit, act)
            }
            g[i] = act * up
        } else {
            g[i] = neural.Apply(a, u[i])
        }
    }
}

func _combine(_ d: gpu.Span<float32>, _ w: gpu.Span<float32>, _ y: gpu.MutableSpan<float32>, _ dim: int, _ used: int, _ n: int,
              _ accumulate: bool) kernel {
    // Token t's output: its experts' rows weighted and added, in the order
    // chosen.
    let c = gpu.Index.x
    let t = gpu.Index.y
    if c < dim && t < n {
        var s: float32 = 0
        var j = 0
        while j < used {
            s = s + w[t * used + j] * d[(t * used + j) * dim + c]
            j += 1
        }
        y[t * dim + c] = accumulate ? y[t * dim + c] + s : s
    }
}

/// Experts is one weight of every expert: a 3-d tensor [experts, rows,
/// in], each expert's rows one after another, in any block format; and
/// each expert's scale, where the file gives one.
public struct Experts {
    public let Weight: tensor.Tensor
    public let Scales: gpu.Buffer<float32>?

    public init(_ weight: tensor.Tensor, scales: gpu.Buffer<float32>? = nil) throws {
        if weight.Shape.count != 3 {
            throw MoEError.unsupported("an experts' weight is [experts, rows, in], not \(weight.Shape)")
        }
        self.Weight = weight
        self.Scales = scales
    }

    public var Count: int { return Weight.Shape[0] }
    public var Rows: int { return Weight.Shape[1] }
    public var In: int { return Weight.Shape[2] }

    /// Forward writes, for each item, its expert's rows times x's row --
    /// x's row i / xShared -- into y's row i.
    public func Forward(_ x: gpu.Buffer<float32>, ids: gpu.Buffer<int32>, items: int, xShared: int, into y: gpu.Buffer<float32>) async throws {
        let w = Weight.Storage
        let m = Rows, k = In, e = Count
        switch Weight.DType {
        case .F32: try await linalg.GemvIds(w, dtype.F32(), ids: ids, x, into: y, m: m, k: k, experts: e, items: items, xShared: xShared)
        case .F16: try await linalg.GemvIds(w, dtype.F16(), ids: ids, x, into: y, m: m, k: k, experts: e, items: items, xShared: xShared)
        case .BF16: try await linalg.GemvIds(w, dtype.BF16(), ids: ids, x, into: y, m: m, k: k, experts: e, items: items, xShared: xShared)
        case .Q4_0: try await linalg.GemvIds(w, dtype.Q4_0(), ids: ids, x, into: y, m: m, k: k, experts: e, items: items, xShared: xShared)
        case .Q8_0: try await linalg.GemvIds(w, dtype.Q8_0(), ids: ids, x, into: y, m: m, k: k, experts: e, items: items, xShared: xShared)
        case .Q4_K: try await linalg.GemvIds(w, dtype.Q4_K(), ids: ids, x, into: y, m: m, k: k, experts: e, items: items, xShared: xShared)
        case .Q6_K: try await linalg.GemvIds(w, dtype.Q6_K(), ids: ids, x, into: y, m: m, k: k, experts: e, items: items, xShared: xShared)
        default:
            throw MoEError.unsupported("experts of \(Weight.DType.Name)")
        }
        if let s = Scales {
            try await _expertScale.Launch(y, ids, s, m, items, over: ((m + 63) / 64 * 64, items, 1), workgroup: (64, 1, 1))
        }
    }
}

/// MoE is the sparse feed-forward network: a router (a Linear to one
/// logit an expert), the gating, Used experts chosen a token, and the
/// experts' gate, up and down -- gate nil for an ungated one.
public final class MoE: nn.FFN {
    public let Router: nn.Linear
    public let Gate: Experts?
    public let Up: Experts
    public let Down: Experts
    public let Used: int
    public var Gating: Gating = .softmax
    /// Activation is the experts' (SiLU for most).
    public var Activation: neural.Activation = .SiLU
    /// SelectionBias is added to the probabilities to choose by, not to
    /// the weights (DeepSeek V3's exp_probs_b).
    public var SelectionBias: gpu.Buffer<float32>? = nil
    /// NormWeights divides the chosen weights by their sum.
    public var NormWeights: bool = false
    /// WeightScale multiplies them (0 or 1: none).
    public var WeightScale: float32 = 0
    /// Clamp is llama.cpp's swiglu_clamp_exp: up within ±Clamp and the
    /// gate's activation at most Clamp (0: none).
    public var Clamp: float32 = 0
    let _ws: _Work

    public init(router: nn.Linear, gate: Experts?, up: Experts, down: Experts, used: int) throws {
        let e = up.Count
        if router.Out != e || down.Count != e || (gate != nil && gate!.Count != e) || used < 1 || used > e || e > 1024 {
            throw MoEError.unsupported("\(e) experts, \(used) used, a router to \(router.Out)")
        }
        self.Router = router
        self.Gate = gate
        self.Up = up
        self.Down = down
        self.Used = used
        self._ws = _Work()
    }

    public var Experts: int { return Up.Count }

    public func Forward(_ x: gpu.Buffer<float32>, rows n: int, into y: gpu.Buffer<float32>, accumulate: bool) async throws {
        let d = Router.Device
        let e = Experts, k = Used
        let hidden = Up.Rows
        let dim = Down.Rows
        let logits = try _ws.Floats(d, 0, n * e)
        try await Router.Forward(x, rows: n, into: logits)
        let ids = try _ws.Ids(d, n * k)
        let weights = try _ws.Floats(d, 1, n * k)
        let bias = SelectionBias ?? logits
        try await _route.Launch(logits, bias, ids, weights, e, k, Gating._code, SelectionBias != nil, NormWeights, WeightScale,
                                over: n * 64, workgroup: 64)
        let up = try _ws.Floats(d, 2, n * k * hidden)
        try await Up.Forward(x, ids: ids, items: n * k, xShared: k, into: up)
        let act = try _ws.Floats(d, 3, n * k * hidden)
        if let g = Gate {
            try await g.Forward(x, ids: ids, items: n * k, xShared: k, into: act)
        }
        try await _gatedExperts.Launch(act, up, Activation._code, Gate != nil, Clamp, n * k * hidden, over: n * k * hidden)
        let down = try _ws.Floats(d, 4, n * k * dim)
        try await Down.Forward(act, ids: ids, items: n * k, xShared: 1, into: down)
        try await _combine.Launch(down, weights, y, dim, k, n, accumulate, over: ((dim + 63) / 64 * 64, n, 1), workgroup: (64, 1, 1))
    }
}

/// _Work is a MoE's buffers by slot, grown to the most needed: apart from
/// Forward because the compiler's SIL verifier rejects CreateBuffer
/// specialized inside a branch of an async function.
final class _Work {
    var _bufs: [gpu.Buffer<float32>] = []
    var _have: [bool] = []
    var _ids: gpu.Buffer<int32>? = nil

    func Floats(_ d: gpu.Device, _ slot: int, _ count: int) throws -> gpu.Buffer<float32> {
        while _have.count <= slot {
            _have.append(false)
            _bufs.append(try d.CreateBuffer(of: float32.self, count: 1))
        }
        if !_have[slot] || _bufs[slot].count < count {
            _bufs[slot] = try d.CreateBuffer(of: float32.self, count: max(count, 1))
            _have[slot] = true
        }
        return _bufs[slot].Slice(from: 0, count: count)
    }

    func Ids(_ d: gpu.Device, _ count: int) throws -> gpu.Buffer<int32> {
        if _ids == nil || _ids!.count < count {
            // A float's bytes viewed as ints: CreateBuffer(of: int32.self)
            // specialized here fails the compiler's SIL verifier.
            _ids = try d.CreateBuffer(of: float32.self, count: max(count, 1)).View(as: int32.self)
        }
        return _ids!.Slice(from: 0, count: count)
    }
}

func _gatedAdd(_ s: gpu.Span<float32>, _ g: gpu.Span<float32>, _ y: gpu.MutableSpan<float32>, _ dim: int, _ n: int, _ accumulate: bool) kernel {
    // Token t's row of s times the sigmoid of its gate, into y's.
    let c = gpu.Index.x
    let t = gpu.Index.y
    if c < dim && t < n {
        let v = math.Sigmoid(g[t]) * s[t * dim + c]
        y[t * dim + c] = accumulate ? y[t * dim + c] + v : v
    }
}

/// Gated is a shared expert behind a gate: each token's output is the
/// expert's times the sigmoid of Gate·x, one logit a token -- Qwen2-MoE's
/// ffn_gate_inp_shexp.
public final class Gated: nn.FFN {
    public let Expert: nn.FFN
    public let Gate: nn.Linear
    public let Dim: int
    let _ws: _Work

    public init(_ expert: nn.FFN, gate: nn.Linear, dim: int) throws {
        if gate.Out != 1 {
            throw MoEError.unsupported("a shared expert's gate to \(gate.Out), not 1")
        }
        self.Expert = expert
        self.Gate = gate
        self.Dim = dim
        self._ws = _Work()
    }

    public func Forward(_ x: gpu.Buffer<float32>, rows n: int, into y: gpu.Buffer<float32>, accumulate: bool) async throws {
        let d = Gate.Device
        let s = try _ws.Floats(d, 0, n * Dim)
        try await Expert.Forward(x, rows: n, into: s, accumulate: false)
        let g = try _ws.Floats(d, 1, n)
        try await Gate.Forward(x, rows: n, into: g)
        try await _gatedAdd.Launch(s, g, y, Dim, n, accumulate, over: ((Dim + 63) / 64 * 64, n, 1), workgroup: (64, 1, 1))
    }
}
