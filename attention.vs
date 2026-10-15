// Attention, with the KV cache decoding keeps between tokens.
package nn

import (
    "gpu"
    "gpu/attention"
    "gpu/neural"
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

func _store(_ k: gpu.Span<float32>, _ v: gpu.Span<float32>, _ ck: gpu.MutableSpan<float32>, _ cv: gpu.MutableSpan<float32>,
            _ position: int, _ capacity: int, _ d: int) kernel {
    // k and v are [kvHeads, d]; each row goes to its head's row position.
    let i = gpu.Index.x
    if i < k.count {
        let at = (i / d * capacity + position) * d + i % d
        ck[at] = k[i]
        cv[at] = v[i]
    }
}

func _ropeStore(_ qkv: gpu.MutableSpan<float32>, _ cos: gpu.Span<float32>, _ sin: gpu.Span<float32>,
                _ ck: gpu.MutableSpan<float32>, _ cv: gpu.MutableSpan<float32>,
                _ heads: int, _ kvHeads: int, _ dim: int, _ position: int, _ capacity: int, _ halves: int) kernel {
    // The token's Q, K and V heads one after another, a work-item a pair of
    // one head (x the pair, y the head): Q's pairs turned in place, K's
    // turned into the cache's row position, V's copied there. The turn's
    // cosine and sine are a table's (neural.RopeTable). Attention checked
    // the counts, so each is read and written unchecked.
    let p = gpu.Index.x
    let h = gpu.Index.y
    let pairs = dim &>> 1
    if p >= pairs { return }
    var ia = 2 &* p
    var ib = ia &+ 1
    if halves != 0 {
        ia = p
        ib = p &+ pairs
    }
    let x = qkv._base + h &* dim
    if h < heads &+ kvHeads {
        let at = position &* pairs &+ p
        let c = cos._base[at], s = sin._base[at]
        let a = x[ia], b = x[ib]
        let ra = a * c - b * s, rb = a * s + b * c
        if h < heads {
            x[ia] = ra
            x[ib] = rb
        } else {
            let row = ck._base + ((h &- heads) &* capacity &+ position) &* dim
            row[ia] = ra
            row[ib] = rb
        }
    } else {
        let row = cv._base + ((h &- heads &- kvHeads) &* capacity &+ position) &* dim
        row[ia] = x[ia]
        row[ib] = x[ib]
    }
}

func _ropeStoreRows(_ qkv: gpu.Span<float32>, _ cos: gpu.Span<float32>, _ sin: gpu.Span<float32>,
                    _ ck: gpu.MutableSpan<float32>, _ cv: gpu.MutableSpan<float32>, _ qh: gpu.MutableSpan<float32>,
                    _ heads: int, _ kvHeads: int, _ dim: int, _ row0: int, _ count: int, _ position0: int, _ capacity: int,
                    _ halves: int) kernel {
    // _ropeStore for a run of count tokens (z), rows row0... of qkv, at
    // positions position0...: Q's pairs turned into qh, laid out [heads,
    // count, dim] as attention takes it; K's turned into the cache, V's
    // copied there. Attention checked the counts.
    let p = gpu.Index.x
    let h = gpu.Index.y
    let i = gpu.Index.z
    let pairs = dim &>> 1
    if p >= pairs || i >= count { return }
    var ia = 2 &* p
    var ib = ia &+ 1
    if halves != 0 {
        ia = p
        ib = p &+ pairs
    }
    let position = position0 &+ i
    let x = qkv._base + ((row0 &+ i) &* (heads &+ 2 &* kvHeads) &+ h) &* dim
    if h < heads &+ kvHeads {
        let at = position &* pairs &+ p
        let c = cos._base[at], s = sin._base[at]
        let a = x[ia], b = x[ib]
        let ra = a * c - b * s, rb = a * s + b * c
        if h < heads {
            let row = qh._base + (h &* count &+ i) &* dim
            row[ia] = ra
            row[ib] = rb
        } else {
            let row = ck._base + ((h &- heads) &* capacity &+ position) &* dim
            row[ia] = ra
            row[ib] = rb
        }
    } else {
        let row = cv._base + ((h &- heads &- kvHeads) &* capacity &+ position) &* dim
        row[ia] = x[ia]
        row[ib] = x[ib]
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
/// frequency base, and which elements of a head pair up -- a property of
/// the checkpoint's Q and K weights (neural.RopeLayout).
public struct Rope {
    public var Base: float32
    public var Layout: neural.RopeLayout

    public init(base: float32 = 10000, layout: neural.RopeLayout = .halves) {
        self.Base = base
        self.Layout = layout
    }
}

/// Attention is a transformer's self-attention for one token at a time:
/// the Q, K and V projections -- one Linear, QKV, their rows one after the
/// other, so one product makes all three -- rotary position embeddings,
/// the keys and values so far from a Cache, and the output projection.
/// Heads a multiple of KVHeads is grouped-query attention.
public final class Attention {
    /// QKV is the Q, K and V projections, as Stacked makes them.
    public let QKV: [Linear]
    public let O: Linear
    public let Heads: int
    public let KVHeads: int
    public let HeadDim: int
    public let Rope: Rope
    let _qkv: gpu.Buffer<float32>
    let _qk: gpu.Buffer<float32>
    let _q: gpu.Buffer<float32>
    let _k: gpu.Buffer<float32>
    let _v: gpu.Buffer<float32>
    let _o: gpu.Buffer<float32>
    /// Table is the cosines and sines the rotation is read from: a model
    /// gives its layers one to share; one without is made at the first
    /// Forward.
    public var Table: neural.RopeTable? = nil
    // A batch's buffers, grown to the most rows seen.
    var _rows = 0
    var _batchQKV: gpu.Buffer<float32>? = nil
    var _batchQ: gpu.Buffer<float32>? = nil
    var _batchO: gpu.Buffer<float32>? = nil
    var _batchMerged: gpu.Buffer<float32>? = nil

    public init(qkv: [Linear], o: Linear, heads: int, kvHeads: int, headDim: int, rope: Rope) throws {
        var out = 0
        for l in qkv { out += l.Out }
        if heads == 0 || kvHeads == 0 || heads % kvHeads != 0 || out != (heads + 2 * kvHeads) * headDim || o.In != heads * headDim {
            throw LayerError.unsupported("attention of \(heads) heads and \(kvHeads) KV heads of \(headDim), QKV to \(out), O from \(o.In)")
        }
        self.QKV = qkv
        self.O = o
        self.Heads = heads
        self.KVHeads = kvHeads
        self.HeadDim = headDim
        self.Rope = rope
        let d = o.Device
        self._qkv = try d.CreateBuffer(of: float32.self, count: out)
        self._qk = _qkv.Slice(from: 0, count: (heads + kvHeads) * headDim)
        self._q = _qkv.Slice(from: 0, count: heads * headDim)
        self._k = _qkv.Slice(from: heads * headDim, count: kvHeads * headDim)
        self._v = _qkv.Slice(from: (heads + kvHeads) * headDim, count: kvHeads * headDim)
        self._o = try d.CreateBuffer(of: float32.self, count: heads * headDim)
    }

    /// Fused is an Attention of separate Q, K and V weights, stacked: one
    /// product makes all three, and no weight is copied.
    public static func Fused(q: Linear, k: Linear, v: Linear, o: Linear, heads: int, kvHeads: int, rope: Rope) throws -> Attention {
        return try Attention(qkv: try Stacked([q, k, v]), o: o, heads: heads, kvHeads: kvHeads, headDim: q.Out / heads, rope: rope)
    }

    /// Forward attends n tokens -- x's rows, one or more runs of
    /// sequences -- each to itself and the tokens before it in its run's
    /// cache, adds them to the caches, and writes the output projection of
    /// each into its row of y, or with accumulate adds it.
    public func Forward(_ x: gpu.Buffer<float32>, rows n: int, runs: [Run], into y: gpu.Buffer<float32>, accumulate: bool = false) async throws {
        if n == 1 && runs.count == 1 {
            try await Forward(x, position: runs[0].Position, cache: runs[0].Cache, into: y, accumulate: accumulate)
            return
        }
        if HeadDim % 2 != 0 {
            throw LayerError.unsupported("a batch through heads of odd dimension \(HeadDim)")
        }
        if _rows < n {
            try _grow(n)
        }
        let qkv = _batchQKV!, qh = _batchQ!, oh = _batchO!, merged = _batchMerged!
        try await nn.Forward(QKV, x, rows: n, into: qkv)
        let hd = Heads * HeadDim
        for r in runs {
            let cache = r.Cache
            if r.Count == 0 { continue }
            if r.Row < 0 || r.Row + r.Count > n || r.Position < 0 || r.Position + r.Count > cache.Capacity
                || cache.KVHeads != KVHeads || cache.HeadDim != HeadDim {
                throw LayerError.unsupported("a run of \(r.Count) tokens at row \(r.Row) and position \(r.Position) in a batch of \(n) and a cache of \(cache.Capacity)")
            }
            if Table == nil || !Table!.Fits(dim: HeadDim, positions: cache.Capacity, base: Rope.Base) {
                Table = try await neural.RopeTable.Make(on: _qkv.Device, dim: HeadDim, positions: cache.Capacity, base: Rope.Base)
            }
            let t = Table!
            let q = qh.Slice(from: r.Row * hd, count: r.Count * hd)
            let o = oh.Slice(from: r.Row * hd, count: r.Count * hd)
            try await _ropeStoreRows.Launch(qkv, t.Cos, t.Sin, cache.K, cache.V, q, Heads, KVHeads, HeadDim, r.Row, r.Count,
                                            r.Position, cache.Capacity, Rope.Layout == .halves ? 1 : 0,
                                            over: ((HeadDim / 2 + 31) / 32 * 32, Heads + 2 * KVHeads, r.Count), workgroup: (32, 1, 1))
            let shape = attention.Shape(heads: Heads, kvHeads: KVHeads, queries: r.Count, keys: r.Position + r.Count,
                                        headDim: HeadDim, keyCapacity: cache.Capacity)
            try await attention.Forward(q: q, k: cache.K, v: cache.V, into: o, shape, mask: .Causal)
            try await neural.MergeHeads(o, into: merged.Slice(from: r.Row * hd, count: r.Count * hd), rows: r.Count, heads: Heads, dim: HeadDim)
        }
        try await O.Forward(merged, rows: n, into: y, accumulate: accumulate)
    }

    // _grow makes the batch buffers room for n rows: apart from Forward
    // because the compiler's SIL verifier rejects CreateBuffer specialized
    // inside a branch of an async function.
    func _grow(_ n: int) throws {
        let d = _qkv.Device
        let hd = Heads * HeadDim
        _batchQKV = try d.CreateBuffer(of: float32.self, count: n * (hd + 2 * KVHeads * HeadDim))
        _batchQ = try d.CreateBuffer(of: float32.self, count: n * hd)
        _batchO = try d.CreateBuffer(of: float32.self, count: n * hd)
        _batchMerged = try d.CreateBuffer(of: float32.self, count: n * hd)
        _rows = n
    }

    /// Forward attends x, the token at position, to itself and the tokens
    /// before it in cache, adds it to the cache, and writes the output
    /// projection into y -- or with accumulate adds it to y.
    public func Forward(_ x: gpu.Buffer<float32>, position: int, cache: Cache, into y: gpu.Buffer<float32>, accumulate: bool = false) async throws {
        if position >= cache.Capacity {
            throw LayerError.unsupported("position \(position) past a cache of \(cache.Capacity)")
        }
        try await nn.Forward(QKV, x, into: _qkv)
        if HeadDim % 2 == 0 && cache.KVHeads == KVHeads && cache.HeadDim == HeadDim {
            // One launch turns Q and K and writes K and V into the cache.
            if Table == nil || !Table!.Fits(dim: HeadDim, positions: cache.Capacity, base: Rope.Base) {
                Table = try await neural.RopeTable.Make(on: _qkv.Device, dim: HeadDim, positions: cache.Capacity, base: Rope.Base)
            }
            let t = Table!
            try await _ropeStore.Launch(_qkv, t.Cos, t.Sin, cache.K, cache.V, Heads, KVHeads, HeadDim, position, cache.Capacity,
                                        Rope.Layout == .halves ? 1 : 0,
                                        over: ((HeadDim / 2 + 31) / 32 * 32, Heads + 2 * KVHeads, 1), workgroup: (32, 1, 1))
        } else {
            // Q's heads and K's lie next to each other: one rotation for both.
            try await neural.RoPE(_qk, position: position, heads: Heads + KVHeads, dim: HeadDim, base: Rope.Base, layout: Rope.Layout)
            try await _store.Launch(_k, _v, cache.K, cache.V, position, cache.Capacity, HeadDim, over: _k.count)
        }
        let shape = attention.Shape(heads: Heads, kvHeads: KVHeads, queries: 1, keys: position + 1,
                                    headDim: HeadDim, keyCapacity: cache.Capacity)
        try await attention.Forward(q: _q, k: cache.K, v: cache.V, into: _o, shape, mask: .Causal)
        try await O.Forward(_o, into: y, accumulate: accumulate)
    }
}
