// nn on every device: each layer against the same computation written
// plainly on the host, and Attention with its cache, a token at a time,
// against one causal pass over all the tokens with no cache.
package main

import "gpu"
import "gpu/attention"
import "gpu/linalg"
import "gpu/neural"
import "gpu/gputest"
import "nn"
import "tensor"

@_silgen_name("exp") func cExp(_ x: float64) -> float64

var rng = gputest.Random(seed: 11)

func floats(_ n: int, _ scale: float32 = 1) -> [float32] {
    var out: [float32] = []
    for _ in 0..<n { out.append((rng.Float32() * 2 - 1) * scale) }
    return out
}

func weight(_ d: gpu.Device, _ values: [float32], _ shape: [int]) async throws -> tensor.Tensor {
    let b = try await d.Upload(values)
    return try tensor.Tensor(shape: shape, dtype: .F32, storage: b.View(as: uint8.self))
}

func matvec(_ w: [float32], _ x: [float32], _ out: int) -> [float64] {
    let n = x.count
    var y: [float64] = []
    for r in 0..<out {
        var s: float64 = 0
        for c in 0..<n { s += float64(w[r * n + c]) * float64(x[c]) }
        y.append(s)
    }
    return y
}

func q8Blocks(_ rows: int, _ k: int) -> ([uint8], [float32]) {
    var bytes: [uint8] = [], dense: [float32] = []
    for _ in 0..<(rows * k / 32) {
        let d = float16(rng.Float32() * 0.02 + 0.001)
        bytes.append(uint8(truncatingIfNeeded: d.bitPattern))
        bytes.append(uint8(truncatingIfNeeded: d.bitPattern >> 8))
        for _ in 0..<32 {
            let q = int8(truncatingIfNeeded: rng.Uint32())
            bytes.append(uint8(bitPattern: q))
            dense.append(float32(q) * float32(d))
        }
    }
    return (bytes, dense)
}

for d in gputest.Devices() {
    // Linear, float32 and q8_0.
    let w = floats(40 * 64), x = floats(64)
    let lin = try nn.Linear(try await weight(d, w, [40, 64]))
    let y = try d.CreateBuffer(of: float32.self, count: 40)
    try await lin.Forward(try await d.Upload(x), into: y)
    gputest.Near("Linear f32 40x64", d, try await y.Download(), matvec(w, x, 40), bound: [float64](repeating: 1e-4, count: 40))
    let (qb, qd) = q8Blocks(40, 64)
    let qlin = try nn.Linear(try tensor.Tensor(shape: [40, 64], dtype: .Q8_0, storage: try await d.Upload(qb)))
    try await qlin.Forward(try await d.Upload(x), into: y)
    gputest.Near("Linear q8_0 40x64", d, try await y.Download(), matvec(qd, x, 40), bound: [float64](repeating: 1e-4, count: 40))

    // Embedding: a row, float32 and q8_0.
    let emb = try nn.Embedding(try await weight(d, w, [40, 64]))
    let row = try d.CreateBuffer(of: float32.self, count: 64)
    try await emb.Lookup(7, into: row)
    gputest.Equal("Embedding f32 row 7", d, try await row.Download(), Array(w[(7 * 64)..<(8 * 64)]))
    let qemb = try nn.Embedding(qlin.Weight)
    try await qemb.Lookup(39, into: row)
    gputest.Equal("Embedding q8_0 row 39", d, try await row.Download(), Array(qd[(39 * 64)..<(40 * 64)]))

    // RMSNorm.
    let g = floats(64)
    let norm = nn.RMSNorm(try await weight(d, g, [64]), eps: 1e-5)
    try await norm.Forward(try await d.Upload(x), into: row)
    var ms: float64 = 0
    for v in x { ms += float64(v) * float64(v) }
    let r = 1 / (ms / 64 + 1e-5).squareRoot()
    gputest.Near("RMSNorm 64", d, try await row.Download(), (0..<64).map { float64(x[$0]) * r * float64(g[$0]) }, bound: [float64](repeating: 1e-5, count: 64))

    // GatedMLP: Down(silu(Gate x) * Up x).
    let wg = floats(96 * 64), wu = floats(96 * 64), wd = floats(64 * 96)
    let mlp = try await nn.GatedMLP.Fused(gate: try nn.Linear(try await weight(d, wg, [96, 64])), up: try nn.Linear(try await weight(d, wu, [96, 64])),
                                          down: try nn.Linear(try await weight(d, wd, [64, 96])), activation: .SiLU)
    try await mlp.Forward(try await d.Upload(x), into: row)
    let gx = matvec(wg, x, 96), ux = matvec(wu, x, 96)
    var h: [float32] = []
    for i in 0..<96 { h.append(float32(gx[i] / (1 + cExp(-gx[i])) * ux[i])) }
    let want = matvec(wd, h, 64)
    gputest.Near("GatedMLP 64-96-64", d, try await row.Download(), want, bound: want.map { 1e-5 * ($0.magnitude + 1) })
    // Accumulating: the MLP added to what the output holds, a residual.
    let residual = try await d.Upload(x)
    try await mlp.Forward(try await d.Upload(x), into: residual, accumulate: true)
    gputest.Near("GatedMLP accumulating", d, try await residual.Download(), (0..<64).map { want[$0] + float64(x[$0]) },
                 bound: want.map { 1e-5 * ($0.magnitude + 1) })

    // Attention: 8 heads, 4 KV heads, head dim 8, six tokens a token at a
    // time with a cache, against one causal pass over all six.
    let dim = 64, heads = 8, kvHeads = 4, hd = 8, n = 6
    let wq = floats(dim * dim, 0.3), wk = floats(kvHeads * hd * dim, 0.3), wv = floats(kvHeads * hd * dim, 0.3), wo = floats(dim * dim, 0.3)
    let att = try await nn.Attention.Fused(q: try nn.Linear(try await weight(d, wq, [dim, dim])), k: try nn.Linear(try await weight(d, wk, [kvHeads * hd, dim])),
                                           v: try nn.Linear(try await weight(d, wv, [kvHeads * hd, dim])), o: try nn.Linear(try await weight(d, wo, [dim, dim])),
                                           heads: heads, kvHeads: kvHeads, ropeBase: 10000)
    let cache = try nn.Cache(on: d, kvHeads: kvHeads, headDim: hd, capacity: 16)
    let xs = floats(n * dim)
    var stepped: [float32] = []
    for t in 0..<n {
        try await att.Forward(try await d.Upload(Array(xs[(t * dim)..<((t + 1) * dim)])), position: t, cache: cache, into: row)
        stepped += try await row.Download()
    }
    // The direct pass: Q, K, V for every token, RoPE at each position,
    // [heads, tokens, d] layouts, attention, then O.
    var q: [float32] = [], k: [float32] = [], v: [float32] = []
    for t in 0..<n {
        let xt = Array(xs[(t * dim)..<((t + 1) * dim)])
        q += matvec(wq, xt, dim).map { float32($0) }
        k += matvec(wk, xt, kvHeads * hd).map { float32($0) }
        v += matvec(wv, xt, kvHeads * hd).map { float32($0) }
    }
    let qb2 = try await d.Upload(q), kb2 = try await d.Upload(k)
    let positions = try await d.Upload((0..<n).map { int32($0) })
    try await neural.RoPE(qb2, positions: positions, heads: heads, dim: hd)
    try await neural.RoPE(kb2, positions: positions, heads: kvHeads, dim: hd)
    func headsFirst(_ a: [float32], _ hs: int) -> [float32] {
        var out = [float32](repeating: 0, count: a.count)
        for t in 0..<n { for hh in 0..<hs { for c in 0..<hd { out[(hh * n + t) * hd + c] = a[(t * hs + hh) * hd + c] } } }
        return out
    }
    let qh = headsFirst(try await qb2.Download(), heads), kh = headsFirst(try await kb2.Download(), kvHeads), vh = headsFirst(v, kvHeads)
    let o = try d.CreateBuffer(of: float32.self, count: n * dim)
    try await attention.Forward(q: try await d.Upload(qh), k: try await d.Upload(kh), v: try await d.Upload(vh), into: o,
                                attention.Shape(heads: heads, kvHeads: kvHeads, queries: n, keys: n, headDim: hd), mask: .Causal)
    let oh = try await o.Download()
    var direct: [float64] = []
    for t in 0..<n {
        var ot: [float32] = []
        for hh in 0..<heads { for c in 0..<hd { ot.append(oh[(hh * n + t) * hd + c]) } }
        direct += matvec(wo, ot, dim)
    }
    gputest.Near("Attention with a cache, 6 tokens, GQA 8/4", d, stepped, direct, bound: direct.map { 1e-5 * ($0.magnitude + 1) })
}

gputest.Done()
