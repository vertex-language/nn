// Package nn is the layers models are made of, over tensor: each holds its
// weights as Tensors whatever their format, and runs on the device they
// are on.
//
// This is the first cut, grown from what decoding a llama one token at a
// time needs: Linear (dense or block-quantized behind one interface),
// Embedding, RMSNorm, GatedMLP, and Attention with its KV Cache. A layer
// takes and writes float32 activations in gpu buffers, and keeps what it
// needs between its steps, so that a token's forward pass allocates
// nothing. Batches, prefill and training come with the models that need
// them (proposed_ai_packages.md §6.2).
package nn

import (
    "gpu"
    "gpu/dtype"
    "gpu/linalg"
    "gpu/neural"
    "tensor"
)

/// LayerError is a layer given what it cannot take.
public enum LayerError: Error {
    case unsupported(string)

    public var Message: string {
        switch self {
        case .unsupported(let why): return "nn: " + why
        }
    }
}

/// Linear is y = W·x, W a [out, in] weight: float32, float16 or bfloat16,
/// or q4_0, q8_0, q4_K or q6_K blocks, decoded in place as the product is
/// taken. W may be stacked from
/// up to three weights of the same format and input -- a model's Q, K and V
/// projections, or gate and up -- which one product computes together,
/// with no copy of the weights made.
///
/// Each part may have a bias, added to its outputs, and a scale it is
/// multiplied by: y = s·(W·x) + b, or with BiasFirst y = s·(W·x + b) --
/// llama.cpp applies a weight's `.scale` and `.bias` in the first order to
/// attention's projections and the second to an MLP's.
public struct Linear {
    /// Parts are W's weights, stacked by rows: one, for a plain Linear.
    public let Parts: [tensor.Tensor]
    /// Biases are the parts' biases, float32, nil for a part without one.
    public let Biases: [gpu.Buffer<float32>?]
    /// Scales are the parts' scales, 1 for a part without one.
    public let Scales: [float32]
    /// BiasFirst is whether the bias is added before the scale multiplies.
    public let BiasFirst: bool
    let _scratch: _Scratch

    public init(_ weight: tensor.Tensor, bias: gpu.Buffer<float32>? = nil, scale: float32 = 1, biasFirst: bool = false) throws {
        try self.init(stacking: [weight], biases: [bias], scales: [scale], biasFirst: biasFirst)
    }

    public init(stacking parts: [tensor.Tensor], biases: [gpu.Buffer<float32>?]? = nil, scales: [float32]? = nil,
                biasFirst: bool = false) throws {
        if parts.isEmpty || parts.count > 3 {
            throw LayerError.unsupported("a Linear of \(parts.count) stacked weights; 1 to 3")
        }
        for p in parts {
            if p.Shape.count != 2 || p.Shape[1] != parts[0].Shape[1] || p.DType != parts[0].DType {
                throw LayerError.unsupported("a Linear's weights are [out, in] of one dtype and in, not \(p.Shape) \(p.DType.Name)")
            }
        }
        var none: [gpu.Buffer<float32>?] = []
        for _ in parts { none.append(nil) }
        let b = biases ?? none
        let sc = scales ?? [float32](repeating: 1, count: parts.count)
        if b.count != parts.count || sc.count != parts.count {
            throw LayerError.unsupported("a Linear of \(parts.count) parts with \(b.count) biases and \(sc.count) scales")
        }
        for i in 0..<parts.count {
            if let bb = b[i], bb.count != parts[i].Shape[0] {
                throw LayerError.unsupported("a bias of \(bb.count) for \(parts[i].Shape[0]) outputs")
            }
        }
        self.Parts = parts
        self.Biases = b
        self.Scales = sc
        self.BiasFirst = biasFirst
        self._scratch = _Scratch()
    }

    /// Plain is whether no part has a bias or a scale: the product alone.
    public var Plain: bool {
        for i in 0..<Parts.count where Biases[i] != nil || Scales[i] != 1 {
            return false
        }
        return true
    }

    // _affine writes the scaled and biased product of n rows -- t, packed,
    // Out a row -- into y's rows of stride floats from column, or with
    // accumulate adds it.
    func _affine(_ t: gpu.Buffer<float32>, rows n: int, into y: gpu.Buffer<float32>, accumulate: bool, stride: int, column: int) async throws {
        let out = Out
        let ldy = stride == 0 ? out : stride
        var off = 0
        for i in 0..<Parts.count {
            let m = Parts[i].Shape[0]
            let b = Biases[i]
            try await _affineRows.Launch(t, y, b ?? t, b != nil, BiasFirst, Scales[i], n, out, off, m, ldy, column + off, accumulate,
                                         over: ((m + 63) / 64 * 64, n, 1), workgroup: (64, 1, 1))
            off += m
        }
    }

    /// Weight is the first part: the whole weight of a plain Linear.
    public var Weight: tensor.Tensor { return Parts[0] }
    public var Device: gpu.Device { return Parts[0].Device }
    public var In: int { return Parts[0].Shape[1] }
    public var Out: int {
        var n = 0
        for p in Parts { n += p.Shape[0] }
        return n
    }

    /// Forward writes the outputs of n tokens into y: x is n rows of In,
    /// y n rows of stride floats (0: Out), this Linear's outputs from
    /// column. With accumulate it adds them to what y holds. One token is
    /// a matrix-vector product; more are one matrix product, each tile of
    /// weights read once for 32 tokens.
    public func Forward(_ x: gpu.Buffer<float32>, rows n: int, into y: gpu.Buffer<float32>, accumulate: bool = false,
                        stride: int = 0, column: int = 0) async throws {
        if n == 1 && column == 0 {
            try await Forward(x, into: y, accumulate: accumulate)
            return
        }
        if !Plain {
            let t = try _scratch.Floats(Device, n * Out)
            try await _gemm(x, rows: n, into: t, accumulate: false, stride: 0, column: 0)
            try await _affine(t, rows: n, into: y, accumulate: accumulate, stride: stride, column: column)
            return
        }
        try await _gemm(x, rows: n, into: y, accumulate: accumulate, stride: stride, column: column)
    }

    func _gemm(_ x: gpu.Buffer<float32>, rows n: int, into y: gpu.Buffer<float32>, accumulate: bool, stride: int, column: int) async throws {
        let rows = Parts.map { $0.Shape[0] }
        let w = Parts.map { $0.Storage }
        switch Weight.DType {
        case .F32:
            try await linalg.Gemm(w, rows: rows, dtype.F32(), x, into: y, n: n, k: In, accumulate: accumulate, stride: stride, column: column)
        case .F16:
            try await linalg.Gemm(w, rows: rows, dtype.F16(), x, into: y, n: n, k: In, accumulate: accumulate, stride: stride, column: column)
        case .BF16:
            try await linalg.Gemm(w, rows: rows, dtype.BF16(), x, into: y, n: n, k: In, accumulate: accumulate, stride: stride, column: column)
        case .Q4_0:
            try await linalg.Gemm(w, rows: rows, dtype.Q4_0(), x, into: y, n: n, k: In, accumulate: accumulate, stride: stride, column: column)
        case .Q8_0:
            try await linalg.Gemm(w, rows: rows, dtype.Q8_0(), x, into: y, n: n, k: In, accumulate: accumulate, stride: stride, column: column)
        case .Q4_K:
            try await linalg.Gemm(w, rows: rows, dtype.Q4_K(), x, into: y, n: n, k: In, accumulate: accumulate, stride: stride, column: column)
        case .Q6_K:
            try await linalg.Gemm(w, rows: rows, dtype.Q6_K(), x, into: y, n: n, k: In, accumulate: accumulate, stride: stride, column: column)
        default:
            throw LayerError.unsupported("a Linear of \(Weight.DType.Name)")
        }
    }

    /// Forward writes W·x into y: x has In elements, y Out. With
    /// accumulate it adds W·x to what y holds: a residual connection in the
    /// same pass.
    public func Forward(_ x: gpu.Buffer<float32>, into y: gpu.Buffer<float32>, accumulate: bool = false) async throws {
        if Plain {
            try await _product(x, into: y, accumulate: accumulate)
            return
        }
        let t = try _scratch.Floats(Device, Out)
        try await _product(x, into: t, accumulate: false)
        try await _affine(t, rows: 1, into: y, accumulate: accumulate, stride: 0, column: 0)
    }

    func _product(_ x: gpu.Buffer<float32>, into y: gpu.Buffer<float32>, accumulate: bool) async throws {
        let rows = Parts.map { $0.Shape[0] }
        switch Weight.DType {
        case .F32:
            var floats: [gpu.Buffer<float32>] = []
            for p in Parts { floats.append(try p.Floats()) }
            try await linalg.Gemv(floats, rows: rows, x, into: y, k: In, accumulate: accumulate)
        case .F16:
            try await linalg.Gemv(Parts.map { $0.Storage }, rows: rows, dtype.F16(), x, into: y, k: In, accumulate: accumulate)
        case .BF16:
            try await linalg.Gemv(Parts.map { $0.Storage }, rows: rows, dtype.BF16(), x, into: y, k: In, accumulate: accumulate)
        case .Q4_0:
            try await linalg.Gemv(Parts.map { $0.Storage }, rows: rows, dtype.Q4_0(), x, into: y, k: In, accumulate: accumulate)
        case .Q8_0:
            try await linalg.Gemv(Parts.map { $0.Storage }, rows: rows, dtype.Q8_0(), x, into: y, k: In, accumulate: accumulate)
        case .Q4_K:
            try await linalg.Gemv(Parts.map { $0.Storage }, rows: rows, dtype.Q4_K(), x, into: y, k: In, accumulate: accumulate)
        case .Q6_K:
            try await linalg.Gemv(Parts.map { $0.Storage }, rows: rows, dtype.Q6_K(), x, into: y, k: In, accumulate: accumulate)
        default:
            throw LayerError.unsupported("a Linear of \(Weight.DType.Name)")
        }
    }
}

/// Stacked is Linears for weights of one input, in order: neighbours of
/// one format stacked into one (one product for them), so that together
/// they write the weights' outputs one after another. Q and K of q4_K
/// with V of q6_K -- a Q4_K_M file's attention -- are two.
public func Stacked(_ weights: [Linear]) throws -> [Linear] {
    var out: [Linear] = []
    var group: [tensor.Tensor] = []
    var biases: [gpu.Buffer<float32>?] = []
    var scales: [float32] = []
    var first = false
    for w in weights {
        for i in 0..<w.Parts.count {
            let p = w.Parts[i]
            if !group.isEmpty && (group[0].DType != p.DType || group.count == 3 || first != w.BiasFirst) {
                out.append(try Linear(stacking: group, biases: biases, scales: scales, biasFirst: first))
                group = []
                biases = []
                scales = []
            }
            if group.isEmpty { first = w.BiasFirst }
            group.append(p)
            biases.append(w.Biases[i])
            scales.append(w.Scales[i])
        }
    }
    if !group.isEmpty {
        out.append(try Linear(stacking: group, biases: biases, scales: scales, biasFirst: first))
    }
    return out
}

func _affineRows(_ t: gpu.Span<float32>, _ y: gpu.MutableSpan<float32>, _ b: gpu.Span<float32>, _ hasBias: bool, _ biasFirst: bool, _ scale: float32,
                 _ n: int, _ out: int, _ off: int, _ m: int, _ ldy: int, _ col: int, _ accumulate: bool) kernel {
    // Row r, element j of a part: s·t[r, off + j] + b[j] (or s·(t + b)),
    // into y[r, col + j].
    let j = gpu.Index.x
    let r = gpu.Index.y
    if j < m && r < n {
        var v = t[r * out + off + j]
        if hasBias && biasFirst { v = v + b[j] }
        v = scale * v
        if hasBias && !biasFirst { v = v + b[j] }
        let at = r * ldy + col + j
        y[at] = accumulate ? y[at] + v : v
    }
}

/// _Scratch is a Linear's buffer for its product before the bias and scale,
/// grown to the most it has needed. It is apart from the methods that use
/// it because the compiler's SIL verifier rejects CreateBuffer specialized
/// inside a branch of an async function.
final class _Scratch {
    var buf: gpu.Buffer<float32>? = nil

    func Floats(_ d: gpu.Device, _ n: int) throws -> gpu.Buffer<float32> {
        if buf == nil || buf!.count < n {
            buf = try d.CreateBuffer(of: float32.self, count: n)
        }
        return buf!.Slice(from: 0, count: n)
    }
}

/// Forward runs Linears of one input into consecutive slices of y: what
/// Stacked makes of several projections, as one.
public func Forward(_ linears: [Linear], _ x: gpu.Buffer<float32>, into y: gpu.Buffer<float32>) async throws {
    var at = 0
    for l in linears {
        try await l.Forward(x, into: linears.count == 1 ? y : y.Slice(from: at, count: l.Out))
        at += l.Out
    }
}

/// Forward runs Linears of one input over n tokens: each token's outputs
/// the Linears' one after another in its row of y, as Forward(_:_:into:)
/// writes one token's.
public func Forward(_ linears: [Linear], _ x: gpu.Buffer<float32>, rows n: int, into y: gpu.Buffer<float32>) async throws {
    if n == 1 {
        try await Forward(linears, x, into: y)
        return
    }
    var total = 0
    for l in linears { total += l.Out }
    var at = 0
    for l in linears {
        try await l.Forward(x, rows: n, into: y, stride: total, column: at)
        at += l.Out
    }
}

func _gatherRow(_ table: gpu.Span<float32>, _ rows: gpu.Span<int32>, _ index: int, _ tableRows: int, _ y: gpu.MutableSpan<float32>, _ n: int) kernel {
    // Row rows[index] of a float32 table into y: a row past the table is
    // row 0.
    let i = gpu.Index.x
    if i < n {
        var r = int(rows[index])
        if r < 0 || r >= tableRows { r = 0 }
        y[i] = table[r * n + i]
    }
}

/// Embedding is a table of vectors, one a token: a [count, dim] weight,
/// float32, a half type or block-quantized.
public struct Embedding {
    public let Weight: tensor.Tensor

    public init(_ weight: tensor.Tensor) throws {
        if weight.Shape.count != 2 {
            throw LayerError.unsupported("an Embedding's weight is [count, dim], not \(weight.Shape)")
        }
        self.Weight = weight
    }

    public var Dim: int { return Weight.Shape[1] }

    /// Lookup writes the vector of token tokens[index] into x, the token
    /// read on the device: a token the device chose -- greedy decoding's
    /// next -- looked up with nothing read back. A token outside the table
    /// is token 0.
    public func Lookup(tokens: gpu.Buffer<int32>, at index: int, into x: gpu.Buffer<float32>) async throws {
        let rb = Weight.RowBytes
        switch Weight.DType {
        case .F32:
            if index < 0 || index >= tokens.count || x.count < Dim {
                throw LayerError.unsupported("a lookup of token \(index) of \(tokens.count), into \(x.count) of \(Dim)")
            }
            try await _gatherRow.Launch(try Weight.Floats(), tokens, index, Weight.Shape[0], x, Dim, over: Dim)
        case .F16:
            try await dtype.DequantizeRow(Weight.Storage, dtype.F16(), rows: tokens, index: index, rowBytes: rb, count: Dim, into: x)
        case .BF16:
            try await dtype.DequantizeRow(Weight.Storage, dtype.BF16(), rows: tokens, index: index, rowBytes: rb, count: Dim, into: x)
        case .Q4_0:
            try await dtype.DequantizeRow(Weight.Storage, dtype.Q4_0(), rows: tokens, index: index, rowBytes: rb, count: Dim, into: x)
        case .Q8_0:
            try await dtype.DequantizeRow(Weight.Storage, dtype.Q8_0(), rows: tokens, index: index, rowBytes: rb, count: Dim, into: x)
        case .Q4_K:
            try await dtype.DequantizeRow(Weight.Storage, dtype.Q4_K(), rows: tokens, index: index, rowBytes: rb, count: Dim, into: x)
        case .Q6_K:
            try await dtype.DequantizeRow(Weight.Storage, dtype.Q6_K(), rows: tokens, index: index, rowBytes: rb, count: Dim, into: x)
        default:
            throw LayerError.unsupported("an Embedding of \(Weight.DType.Name)")
        }
    }

    /// Lookup writes the vectors of tokens[first..<first + count] into x,
    /// one after another: a batch's embeddings in one launch. A token
    /// outside the table is token 0.
    public func Lookup(tokens: gpu.Buffer<int32>, first: int, count: int, into x: gpu.Buffer<float32>) async throws {
        let rb = Weight.RowBytes
        let s = Weight.Storage
        switch Weight.DType {
        case .F32: try await dtype.DequantizeRows(s, dtype.F32(), rows: tokens, first: first, count: count, rowBytes: rb, n: Dim, into: x)
        case .F16: try await dtype.DequantizeRows(s, dtype.F16(), rows: tokens, first: first, count: count, rowBytes: rb, n: Dim, into: x)
        case .BF16: try await dtype.DequantizeRows(s, dtype.BF16(), rows: tokens, first: first, count: count, rowBytes: rb, n: Dim, into: x)
        case .Q4_0: try await dtype.DequantizeRows(s, dtype.Q4_0(), rows: tokens, first: first, count: count, rowBytes: rb, n: Dim, into: x)
        case .Q8_0: try await dtype.DequantizeRows(s, dtype.Q8_0(), rows: tokens, first: first, count: count, rowBytes: rb, n: Dim, into: x)
        case .Q4_K: try await dtype.DequantizeRows(s, dtype.Q4_K(), rows: tokens, first: first, count: count, rowBytes: rb, n: Dim, into: x)
        case .Q6_K: try await dtype.DequantizeRows(s, dtype.Q6_K(), rows: tokens, first: first, count: count, rowBytes: rb, n: Dim, into: x)
        default:
            throw LayerError.unsupported("an Embedding of \(Weight.DType.Name)")
        }
    }

    /// Lookup writes token's vector into x.
    public func Lookup(_ token: int, into x: gpu.Buffer<float32>) async throws {
        let at = token * Weight.RowBytes
        switch Weight.DType {
        case .F32:
            try await x.Copy(from: try Weight.Floats().Slice(from: token * Dim, count: Dim))
        case .F16:
            try await dtype.Dequantize(Weight.Storage, dtype.F16(), at: at, count: Dim, into: x)
        case .BF16:
            try await dtype.Dequantize(Weight.Storage, dtype.BF16(), at: at, count: Dim, into: x)
        case .Q4_0:
            try await dtype.Dequantize(Weight.Storage, dtype.Q4_0(), at: at, count: Dim, into: x)
        case .Q8_0:
            try await dtype.Dequantize(Weight.Storage, dtype.Q8_0(), at: at, count: Dim, into: x)
        case .Q4_K:
            try await dtype.Dequantize(Weight.Storage, dtype.Q4_K(), at: at, count: Dim, into: x)
        case .Q6_K:
            try await dtype.Dequantize(Weight.Storage, dtype.Q6_K(), at: at, count: Dim, into: x)
        default:
            throw LayerError.unsupported("an Embedding of \(Weight.DType.Name)")
        }
    }
}

/// RMSNorm scales x to unit root-mean-square, then by a float32 weight.
public struct RMSNorm {
    public let Weight: tensor.Tensor
    public let Eps: float32

    public init(_ weight: tensor.Tensor, eps: float32) {
        self.Weight = weight
        self.Eps = eps
    }

    /// Forward normalizes rows rows of x into y.
    public func Forward(_ x: gpu.Buffer<float32>, rows: int = 1, into y: gpu.Buffer<float32>) async throws {
        try await neural.RMSNorm(x, weight: try Weight.Floats(), into: y, rows: rows, cols: Weight.Count, eps: Eps)
    }
}

func _gatedRows(_ gu: gpu.Span<float32>, _ y: gpu.MutableSpan<float32>, _ a: int32, _ hidden: int, _ n: int) kernel {
    // Row t of gu is [gate, up], hidden each; row t of y is a(gate)·up.
    let i = gpu.Index.x
    let t = gpu.Index.y
    if i < hidden && t < n {
        let at = t * 2 * hidden + i
        y[t * hidden + i] = gu[at + hidden] * neural.Apply(a, gu[at])
    }
}

/// GatedMLP is Down(a(Gate·x) ⊙ Up·x): SwiGLU with .SiLU, the MLP of
/// Llama, Mistral, Qwen and most decoders since. Gate and Up are one
/// Linear, GateUp, their rows one after the other: one product makes both.
public final class GatedMLP: FFN {
    /// GateUp is the gate and up projections, as Stacked makes them.
    public let GateUp: [Linear]
    public let Down: Linear
    public let Activation: neural.Activation
    let _gateUp: gpu.Buffer<float32>
    let _gate: gpu.Buffer<float32>
    let _up: gpu.Buffer<float32>
    var _batch: gpu.Buffer<float32>? = nil

    public init(gateUp: [Linear], down: Linear, activation: neural.Activation) throws {
        var out = 0
        for l in gateUp { out += l.Out }
        if out % 2 != 0 || out / 2 != down.In {
            throw LayerError.unsupported("a GatedMLP's gate and up of \(out) rows, and down from \(down.In)")
        }
        self.GateUp = gateUp
        self.Down = down
        self.Activation = activation
        let hidden = out / 2
        self._gateUp = try down.Device.CreateBuffer(of: float32.self, count: out)
        self._gate = _gateUp.Slice(from: 0, count: hidden)
        self._up = _gateUp.Slice(from: hidden, count: hidden)
    }

    /// Fused is a GatedMLP of separate gate and up weights, stacked: one
    /// product makes both, and no weight is copied.
    public static func Fused(gate: Linear, up: Linear, down: Linear, activation: neural.Activation) throws -> GatedMLP {
        return try GatedMLP(gateUp: try Stacked([gate, up]), down: down, activation: activation)
    }

    // _grow makes the batch buffer count floats: apart from Forward because
    // the compiler's SIL verifier rejects CreateBuffer specialized inside a
    // branch of an async function.
    func _grow(_ count: int) throws {
        _batch = try Down.Device.CreateBuffer(of: float32.self, count: count)
    }

    /// Forward writes the MLP of n tokens' rows of x into y's, or with
    /// accumulate adds it.
    public func Forward(_ x: gpu.Buffer<float32>, rows n: int, into y: gpu.Buffer<float32>, accumulate: bool = false) async throws {
        if n == 1 {
            try await Forward(x, into: y, accumulate: accumulate)
            return
        }
        let hidden = Down.In
        if _batch == nil || _batch!.count < n * 3 * hidden {
            try _grow(n * 3 * hidden)
        }
        let gu = _batch!.Slice(from: 0, count: n * 2 * hidden)
        let act = _batch!.Slice(from: n * 2 * hidden, count: n * hidden)
        try await nn.Forward(GateUp, x, rows: n, into: gu)
        // Each row's gate and up halves, a(gate)·up, into act's row.
        try await _gatedRows.Launch(gu, act, Activation._code, hidden, n, over: ((hidden + 63) / 64 * 64, n, 1), workgroup: (64, 1, 1))
        try await Down.Forward(act, rows: n, into: y, accumulate: accumulate)
    }

    /// Forward writes the MLP of x into y, or with accumulate adds it.
    public func Forward(_ x: gpu.Buffer<float32>, into y: gpu.Buffer<float32>, accumulate: bool = false) async throws {
        try await nn.Forward(GateUp, x, into: _gateUp)
        try await neural.Gated(_up, gate: _gate, Activation, into: _up)
        try await Down.Forward(_up, into: y, accumulate: accumulate)
    }
}
