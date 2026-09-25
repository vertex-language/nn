// Attention, with the KV cache decoding keeps between tokens.
package nn

import "gpu"
import "gpu/attention"
import "gpu/neural"
import "tensor"

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

/// Attention is a transformer's self-attention for one token at a time:
/// the Q, K and V projections -- one Linear, QKV, their rows one after the
/// other, so one product makes all three -- rotary position embeddings,
/// the keys and values so far from a Cache, and the output projection.
/// Heads a multiple of KVHeads is grouped-query attention.
public final class Attention {
    public let QKV: Linear
    public let O: Linear
    public let Heads: int
    public let KVHeads: int
    public let HeadDim: int
    public let RopeBase: float32
    let _qkv: gpu.Buffer<float32>
    let _qk: gpu.Buffer<float32>
    let _q: gpu.Buffer<float32>
    let _k: gpu.Buffer<float32>
    let _v: gpu.Buffer<float32>
    let _o: gpu.Buffer<float32>

    public init(qkv: Linear, o: Linear, heads: int, kvHeads: int, headDim: int, ropeBase: float32) throws {
        if heads == 0 || kvHeads == 0 || heads % kvHeads != 0 || qkv.Out != (heads + 2 * kvHeads) * headDim || o.In != heads * headDim {
            throw LayerError.unsupported("attention of \(heads) heads and \(kvHeads) KV heads of \(headDim), QKV to \(qkv.Out), O from \(o.In)")
        }
        self.QKV = qkv
        self.O = o
        self.Heads = heads
        self.KVHeads = kvHeads
        self.HeadDim = headDim
        self.RopeBase = ropeBase
        let d = qkv.Weight.Device
        self._qkv = try d.CreateBuffer(of: float32.self, count: qkv.Out)
        self._qk = _qkv.Slice(from: 0, count: (heads + kvHeads) * headDim)
        self._q = _qkv.Slice(from: 0, count: heads * headDim)
        self._k = _qkv.Slice(from: heads * headDim, count: kvHeads * headDim)
        self._v = _qkv.Slice(from: (heads + kvHeads) * headDim, count: kvHeads * headDim)
        self._o = try d.CreateBuffer(of: float32.self, count: heads * headDim)
    }

    /// Fused is an Attention of separate Q, K and V weights, fused.
    public static func Fused(q: Linear, k: Linear, v: Linear, o: Linear, heads: int, kvHeads: int, ropeBase: float32) async throws -> Attention {
        let qkv = try Linear(try await tensor.ConcatRows([q.Weight, k.Weight, v.Weight]))
        return try Attention(qkv: qkv, o: o, heads: heads, kvHeads: kvHeads, headDim: q.Out / heads, ropeBase: ropeBase)
    }

    /// Forward attends x, the token at position, to itself and the tokens
    /// before it in cache, adds it to the cache, and writes the output
    /// projection into y -- or with accumulate adds it to y.
    public func Forward(_ x: gpu.Buffer<float32>, position: int, cache: Cache, into y: gpu.Buffer<float32>, accumulate: bool = false) async throws {
        if position >= cache.Capacity {
            throw LayerError.unsupported("position \(position) past a cache of \(cache.Capacity)")
        }
        try await QKV.Forward(x, into: _qkv)
        // Q's heads and K's lie next to each other: one rotation for both.
        try await neural.RoPE(_qk, position: position, heads: Heads + KVHeads, dim: HeadDim, base: RopeBase)
        try await _store.Launch(_k, _v, cache.K, cache.V, position, cache.Capacity, HeadDim, over: _k.count)
        let shape = attention.Shape(heads: Heads, kvHeads: KVHeads, queries: 1, keys: position + 1,
                                    headDim: HeadDim, keyCapacity: cache.Capacity)
        try await attention.Forward(q: _q, k: cache.K, v: cache.V, into: _o, shape, mask: .Causal)
        try await O.Forward(_o, into: y, accumulate: accumulate)
    }
}
