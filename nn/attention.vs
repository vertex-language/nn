// Attention, with the KV cache decoding keeps between tokens.
package nn

import "gpu"
import "gpu/attention"
import "gpu/neural"

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
/// the Q, K and V projections, rotary position embeddings, the keys and
/// values so far from a Cache, and the output projection. Heads a multiple
/// of KVHeads is grouped-query attention.
public final class Attention {
    public let Q: Linear
    public let K: Linear
    public let V: Linear
    public let O: Linear
    public let Heads: int
    public let KVHeads: int
    public let HeadDim: int
    public let RopeBase: float32
    let _q: gpu.Buffer<float32>
    let _k: gpu.Buffer<float32>
    let _v: gpu.Buffer<float32>
    let _o: gpu.Buffer<float32>
    let _position: gpu.Buffer<int32>

    public init(q: Linear, k: Linear, v: Linear, o: Linear, heads: int, kvHeads: int, ropeBase: float32) throws {
        if heads == 0 || kvHeads == 0 || heads % kvHeads != 0 || q.Out % heads != 0 || k.Out != kvHeads * (q.Out / heads) {
            throw LayerError.unsupported("attention of \(heads) heads, \(kvHeads) KV heads, Q to \(q.Out), K to \(k.Out)")
        }
        self.Q = q
        self.K = k
        self.V = v
        self.O = o
        self.Heads = heads
        self.KVHeads = kvHeads
        self.HeadDim = q.Out / heads
        self.RopeBase = ropeBase
        let d = q.Weight.Device
        self._q = try d.CreateBuffer(of: float32.self, count: q.Out)
        self._k = try d.CreateBuffer(of: float32.self, count: k.Out)
        self._v = try d.CreateBuffer(of: float32.self, count: v.Out)
        self._o = try d.CreateBuffer(of: float32.self, count: q.Out)
        self._position = try d.CreateBuffer(of: int32.self, count: 1)
    }

    /// Forward attends x, the token at position, to itself and the tokens
    /// before it in cache, adds it to the cache, and writes the output
    /// projection into y.
    public func Forward(_ x: gpu.Buffer<float32>, position: int, cache: Cache, into y: gpu.Buffer<float32>) async throws {
        if position >= cache.Capacity {
            throw LayerError.unsupported("position \(position) past a cache of \(cache.Capacity)")
        }
        try await Q.Forward(x, into: _q)
        try await K.Forward(x, into: _k)
        try await V.Forward(x, into: _v)
        try await _position.Fill(int32(position))
        try await neural.RoPE(_q, positions: _position, heads: Heads, dim: HeadDim, base: RopeBase)
        try await neural.RoPE(_k, positions: _position, heads: KVHeads, dim: HeadDim, base: RopeBase)
        try await _store.Launch(_k, _v, cache.K, cache.V, position, cache.Capacity, HeadDim, over: _k.count)
        let shape = attention.Shape(heads: Heads, kvHeads: KVHeads, queries: 1, keys: position + 1,
                                    headDim: HeadDim, keyCapacity: cache.Capacity)
        try await attention.Forward(q: _q, k: cache.K, v: cache.V, into: _o, shape, mask: .Causal)
        try await O.Forward(_o, into: y)
    }
}
