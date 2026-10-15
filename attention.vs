// Attention, with the KV cache decoding keeps between tokens.
package nn

import (
    "gpu"
    "gpu/attention"
    "gpu/neural"
    "gpu/parallel"
    "math"
    "tensor"
)

/// Cache is the keys and values of the tokens so far, for one layer:
/// [kvHeads, capacity, headDim] each, the first Count rows of a head used.
public final class Cache {
    public let K: gpu.Buffer<float32>
    public let V: gpu.Buffer<float32>
    public let Capacity: int
    public let KVHeads: int
    public let HeadDim: int

    public init(on d: gpu.Device, kvHeads: int, headDim: int, capacity: int) throws {
        self.K = try d.CreateBuffer(of: float32.self, count: kvHeads * capacity * headDim)
        self.V = try d.CreateBuffer(of: float32.self, count: kvHeads * capacity * headDim)
        self.Capacity = capacity
        self.KVHeads = kvHeads
        self.HeadDim = headDim
    }
}

func _ropeStoreRows(_ qkv: gpu.Span<float32>, _ cos: gpu.Span<float32>, _ sin: gpu.Span<float32>,
                    _ ck: gpu.MutableSpan<float32>, _ cv: gpu.MutableSpan<float32>, _ qh: gpu.MutableSpan<float32>,
                    _ heads: int, _ kvHeads: int, _ dim: int, _ rot: int, _ row0: int, _ count: int, _ position0: int,
                    _ capacity: int, _ halves: int) kernel {
    // A run of count tokens (z), rows row0... of qkv -- each row the
    // token's Q, K and V heads one after another -- at positions
    // position0...: Q's heads turned into qh, laid out [heads, count, dim]
    // as attention takes it; K's turned into the cache; V's copied there.
    // A work-item takes two elements of a head (x, of dim/2) of one head
    // (y): the first rot/2 of them a pair turned by the table's cosine and
    // sine (neural.RopeTable, rot/2 pairs a position), the rest two
    // elements past the rotary dimensions, copied as they are -- partial
    // rotary embeddings. Attention checked the counts, so each is read and
    // written unchecked.
    let p = gpu.Index.x
    let h = gpu.Index.y
    let i = gpu.Index.z
    let slots = dim &>> 1
    if p >= slots || i >= count { return }
    let pairs = rot &>> 1
    var ia = 2 &* p
    var ib = ia &+ 1
    var turn = p < pairs
    if turn && halves != 0 {
        ia = p
        ib = p &+ pairs
    }
    if !turn {
        ia = rot &+ 2 &* (p &- pairs)
        ib = ia &+ 1
    }
    let position = position0 &+ i
    let x = qkv._base + ((row0 &+ i) &* (heads &+ 2 &* kvHeads) &+ h) &* dim
    var a = x[ia]
    var b = x[ib]
    if h >= heads &+ kvHeads {
        turn = false
    }
    if turn {
        let at = position &* pairs &+ p
        let c = cos._base[at], s = sin._base[at]
        let ra = a * c - b * s
        b = a * s + b * c
        a = ra
    }
    var row = qh._base
    if h < heads {
        row = qh._base + (h &* count &+ i) &* dim
    } else if h < heads &+ kvHeads {
        row = ck._base + ((h &- heads) &* capacity &+ position) &* dim
    } else {
        row = cv._base + ((h &- heads &- kvHeads) &* capacity &+ position) &* dim
    }
    row[ia] = a
    row[ib] = b
}

func _ropeRows(_ qkv: gpu.MutableSpan<float32>, _ cos: gpu.Span<float32>, _ sin: gpu.Span<float32>,
               _ heads: int, _ kvHeads: int, _ dim: int, _ rot: int, _ row0: int, _ count: int, _ position0: int, _ halves: int) kernel {
    // _ropeStoreRows's turn, in place in qkv: Q's and K's heads (y) of a
    // run of count tokens (z), for a norm that comes after it.
    let p = gpu.Index.x
    let h = gpu.Index.y
    let i = gpu.Index.z
    let pairs = rot &>> 1
    if p >= pairs || i >= count || h >= heads &+ kvHeads { return }
    var ia = 2 &* p
    var ib = ia &+ 1
    if halves != 0 {
        ia = p
        ib = p &+ pairs
    }
    let x = qkv._base + ((row0 &+ i) &* (heads &+ 2 &* kvHeads) &+ h) &* dim
    let at = (position0 &+ i) &* pairs &+ p
    let c = cos._base[at], s = sin._base[at]
    let a = x[ia], b = x[ib]
    x[ia] = a * c - b * s
    x[ib] = a * s + b * c
}

func _segmentNorm(_ x: gpu.MutableSpan<float32>, _ w: gpu.Span<float32>, _ b: gpu.Span<float32>, _ hasBias: bool, _ layer: bool,
                  _ width: int, _ offset: int, _ len: int, _ segs: int, _ shared: bool, _ eps: float32) kernel {
    // A workgroup a segment: elements offset + s·len ..< + len of row r of
    // x (width a row), normalized in place -- by root mean square, or with
    // layer to zero mean and unit variance -- then times the weight (and
    // plus the bias): the segment's own, or with shared the same for all.
    let g = gpu.GroupIndex.x
    let r = g / segs
    let s = g % segs
    let base = r * width + offset + s * len
    let me = parallel.GroupRank()
    let step = parallel.GroupCount()
    var sum: float32 = 0
    var j = me
    if layer {
        while j < len {
            sum = sum + x[base + j]
            j += step
        }
    }
    let mean = layer ? parallel.GroupSum(sum) / float32(len) : 0
    var sq: float32 = 0
    j = me
    while j < len {
        let v = x[base + j] - mean
        sq = sq + v * v
        j += step
    }
    let scale = 1 / math.Sqrt(parallel.GroupSum(sq) / float32(len) + eps)
    let wb = shared ? 0 : s * len
    j = me
    while j < len {
        var v = (x[base + j] - mean) * scale * w[wb + j]
        if hasBias { v = v + b[wb + j] }
        x[base + j] = v
        j += step
    }
}

/// Run is the part of a batch that is one sequence's: count tokens from
/// row Row of the batch, at positions Position..., attending to the keys
/// already in Cache before them.
public struct Run {
    public var Row: int
    public var Count: int
    public var Position: int
    public var Cache: Cache

    public init(row: int, count: int, position: int, cache: Cache) {
        self.Row = row
        self.Count = count
        self.Position = position
        self.Cache = cache
    }
}

/// Rope is how an Attention turns its queries and keys by position: the
/// frequency base, which elements of a head pair up -- a property of the
/// checkpoint's Q and K weights (neural.RopeLayout) -- and how many of a
/// head's dimensions turn (0: all; fewer is partial rotary embeddings, the
/// rest left as they are; -1: none, a layer without position embeddings).
public struct Rope {
    public var Base: float32
    public var Layout: neural.RopeLayout
    public var Dims: int

    public init(base: float32 = 10000, layout: neural.RopeLayout = .halves, dims: int = 0) {
        self.Base = base
        self.Layout = layout
        self.Dims = dims
    }
}

/// HeadNorm is a norm over Q's or K's heads before they turn: over each
/// head (a weight of head dimension, shared by the heads -- Qwen3's,
/// Gemma 3's), or over all the heads of a token at once (a weight of heads
/// times head dimension -- OLMo 2's).
public struct HeadNorm {
    public let Norm: Norm

    public init(_ norm: Norm) {
        self.Norm = norm
    }
}

/// Attention is a transformer's self-attention: the Q, K and V
/// projections -- one Linear, QKV, their rows one after the other, so one
/// product makes all three -- norms over Q's and K's heads where a model
/// has them, rotary position embeddings, the keys and values so far from
/// a Cache, and the output projection. Heads a multiple of KVHeads is
/// grouped-query attention.
public final class Attention {
    /// QKV is the Q, K and V projections, as Stacked makes them.
    public let QKV: [Linear]
    public let O: Linear
    public let Heads: int
    public let KVHeads: int
    public let HeadDim: int
    public let Rope: Rope
    /// QNorm and KNorm, when set, normalize Q's and K's heads before they
    /// turn.
    public var QNorm: HeadNorm? = nil
    public var KNorm: HeadNorm? = nil
    /// NormAfterRope is whether QNorm and KNorm come after the turn
    /// (Maincoder's) rather than before it.
    public var NormAfterRope: bool = false
    /// Scale is the scores' scale: 0 for 1/√HeadDim.
    public var Scale: float32 = 0
    /// Table is the cosines and sines the rotation is read from: a model
    /// gives its layers one to share; one without is made at the first
    /// Forward.
    public var Table: neural.RopeTable? = nil
    // The batch's buffers, grown to the most rows seen.
    var _rows = 0
    var _qkv: gpu.Buffer<float32>? = nil
    var _q: gpu.Buffer<float32>? = nil
    var _o: gpu.Buffer<float32>? = nil
    var _merged: gpu.Buffer<float32>? = nil

    public init(qkv: [Linear], o: Linear, heads: int, kvHeads: int, headDim: int, rope: Rope) throws {
        var out = 0
        for l in qkv { out += l.Out }
        if heads == 0 || kvHeads == 0 || heads % kvHeads != 0 || out != (heads + 2 * kvHeads) * headDim || o.In != heads * headDim {
            throw LayerError.unsupported("attention of \(heads) heads and \(kvHeads) KV heads of \(headDim), QKV to \(out), O from \(o.In)")
        }
        if headDim % 2 != 0 {
            throw LayerError.unsupported("attention heads of odd dimension \(headDim)")
        }
        if rope.Dims < -1 || rope.Dims > headDim || (rope.Dims > 0 && rope.Dims % 2 != 0) {
            throw LayerError.unsupported("rotary embeddings over \(rope.Dims) of a head's \(headDim) dimensions")
        }
        self.QKV = qkv
        self.O = o
        self.Heads = heads
        self.KVHeads = kvHeads
        self.HeadDim = headDim
        self.Rope = rope
    }

    /// Fused is an Attention of separate Q, K and V weights, stacked: one
    /// product makes all three, and no weight is copied.
    public static func Fused(q: Linear, k: Linear, v: Linear, o: Linear, heads: int, kvHeads: int, rope: Rope) throws -> Attention {
        return try Attention(qkv: try Stacked([q, k, v]), o: o, heads: heads, kvHeads: kvHeads, headDim: q.Out / heads, rope: rope)
    }

    /// RotaryDims is how many of a head's dimensions turn.
    public var RotaryDims: int { return Rope.Dims < 0 ? 0 : (Rope.Dims == 0 ? HeadDim : Rope.Dims) }

    // _grow makes the batch buffers room for n rows: apart from Forward
    // because the compiler's SIL verifier rejects CreateBuffer specialized
    // inside a branch of an async function.
    func _grow(_ n: int) throws {
        let d = O.Device
        let hd = Heads * HeadDim
        _qkv = try d.CreateBuffer(of: float32.self, count: n * (hd + 2 * KVHeads * HeadDim))
        _q = try d.CreateBuffer(of: float32.self, count: n * hd)
        _o = try d.CreateBuffer(of: float32.self, count: n * hd)
        _merged = try d.CreateBuffer(of: float32.self, count: n * hd)
        _rows = n
    }

    // _norm normalizes n rows' Q or K heads in place, where a norm says.
    func _norm(_ qkv: gpu.Buffer<float32>, rows n: int, _ hn: HeadNorm?, first: int, heads: int) async throws {
        guard let h = hn else { return }
        let norm = h.Norm
        let width = (Heads + 2 * KVHeads) * HeadDim
        let perHead = norm.Count == HeadDim
        if !perHead && norm.Count != heads * HeadDim {
            throw LayerError.unsupported("a head norm of \(norm.Count) for \(heads) heads of \(HeadDim)")
        }
        let len = perHead ? HeadDim : heads * HeadDim
        let segs = perHead ? heads : 1
        let w = try norm.Weights()
        let b = norm.Bias ?? w
        try await _segmentNorm.Launch(qkv, w, b, norm.Bias != nil, norm.Kind == .layer, width, first * HeadDim, len, segs, perHead,
                                      norm.Eps, over: n * segs * 64, workgroup: 64)
    }

    /// Forward attends x, the token at position, to itself and the tokens
    /// before it in cache, adds it to the cache, and writes the output
    /// projection into y -- or with accumulate adds it to y.
    public func Forward(_ x: gpu.Buffer<float32>, position: int, cache: Cache, into y: gpu.Buffer<float32>, accumulate: bool = false) async throws {
        try await Forward(x, rows: 1, runs: [Run(row: 0, count: 1, position: position, cache: cache)], into: y, accumulate: accumulate)
    }

    /// Forward attends n tokens -- x's rows, one or more runs of
    /// sequences -- each to itself and the tokens before it in its run's
    /// cache, adds them to the caches, and writes the output projection of
    /// each into its row of y, or with accumulate adds it.
    public func Forward(_ x: gpu.Buffer<float32>, rows n: int, runs: [Run], into y: gpu.Buffer<float32>, accumulate: bool = false) async throws {
        if _rows < n {
            try _grow(n)
        }
        let hd = Heads * HeadDim
        let qkv = _qkv!.Slice(from: 0, count: n * (hd + 2 * KVHeads * HeadDim))
        let qh = _q!, oh = _o!, merged = _merged!
        try await nn.Forward(QKV, x, rows: n, into: qkv)
        if !NormAfterRope {
            try await _norm(qkv, rows: n, QNorm, first: 0, heads: Heads)
            try await _norm(qkv, rows: n, KNorm, first: Heads, heads: KVHeads)
        }
        var rot = RotaryDims
        for r in runs {
            let cache = r.Cache
            if r.Count == 0 { continue }
            if r.Row < 0 || r.Row + r.Count > n || r.Position < 0 || r.Position + r.Count > cache.Capacity
                || cache.KVHeads != KVHeads || cache.HeadDim != HeadDim {
                throw LayerError.unsupported("a run of \(r.Count) tokens at row \(r.Row) and position \(r.Position) in a batch of \(n) and a cache of \(cache.Capacity)")
            }
            var cos = qkv, sin = qkv
            if rot > 0 {
                if Table == nil || !Table!.Fits(dim: rot, positions: cache.Capacity, base: Rope.Base) {
                    Table = try await neural.RopeTable.Make(on: O.Device, dim: rot, positions: cache.Capacity, base: Rope.Base)
                }
                cos = Table!.Cos
                sin = Table!.Sin
            }
            if NormAfterRope && (QNorm != nil || KNorm != nil) {
                // Turn in place, normalize, then store as they are.
                if rot > 0 {
                    try await _ropeRows.Launch(qkv, cos, sin, Heads, KVHeads, HeadDim, rot, r.Row, r.Count, r.Position,
                                               Rope.Layout == .halves ? 1 : 0,
                                               over: ((rot / 2 + 31) / 32 * 32, Heads + KVHeads, r.Count), workgroup: (32, 1, 1))
                }
                let run = qkv.Slice(from: r.Row * (hd + 2 * KVHeads * HeadDim), count: r.Count * (hd + 2 * KVHeads * HeadDim))
                try await _norm(run, rows: r.Count, QNorm, first: 0, heads: Heads)
                try await _norm(run, rows: r.Count, KNorm, first: Heads, heads: KVHeads)
                rot = 0
            }
            let q = qh.Slice(from: r.Row * hd, count: r.Count * hd)
            let o = oh.Slice(from: r.Row * hd, count: r.Count * hd)
            try await _ropeStoreRows.Launch(qkv, cos, sin, cache.K, cache.V, q, Heads, KVHeads, HeadDim, rot, r.Row, r.Count,
                                            r.Position, cache.Capacity, Rope.Layout == .halves ? 1 : 0,
                                            over: ((HeadDim / 2 + 31) / 32 * 32, Heads + 2 * KVHeads, r.Count), workgroup: (32, 1, 1))
            let shape = attention.Shape(heads: Heads, kvHeads: KVHeads, queries: r.Count, keys: r.Position + r.Count,
                                        headDim: HeadDim, scale: Scale, keyCapacity: cache.Capacity)
            try await attention.Forward(q: q, k: cache.K, v: cache.V, into: o, shape, mask: .Causal)
            if r.Count == 1 && n == 1 {
                // One token's heads, one after another, are its row.
                try await O.Forward(o, into: y, accumulate: accumulate)
                return
            }
            try await neural.MergeHeads(o, into: merged.Slice(from: r.Row * hd, count: r.Count * hd), rows: r.Count, heads: Heads, dim: HeadDim)
            rot = RotaryDims
        }
        try await O.Forward(merged, rows: n, into: y, accumulate: accumulate)
    }
}
