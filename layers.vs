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
public struct Linear {
    /// Parts are W's weights, stacked by rows: one, for a plain Linear.
    public let Parts: [tensor.Tensor]

    public init(_ weight: tensor.Tensor) throws {
        try self.init(stacking: [weight])
    }

    public init(stacking parts: [tensor.Tensor]) throws {
        if parts.isEmpty || parts.count > 3 {
            throw LayerError.unsupported("a Linear of \(parts.count) stacked weights; 1 to 3")
        }
        for p in parts {
            if p.Shape.count != 2 || p.Shape[1] != parts[0].Shape[1] || p.DType != parts[0].DType {
                throw LayerError.unsupported("a Linear's weights are [out, in] of one dtype and in, not \(p.Shape) \(p.DType.Name)")
            }
        }
        self.Parts = parts
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

    /// Forward writes W·x into y: x has In elements, y Out. With
    /// accumulate it adds W·x to what y holds: a residual connection in the
    /// same pass.
    public func Forward(_ x: gpu.Buffer<float32>, into y: gpu.Buffer<float32>, accumulate: bool = false) async throws {
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
    for w in weights {
        for p in w.Parts {
            if !group.isEmpty && (group[0].DType != p.DType || group.count == 3) {
                out.append(try Linear(stacking: group))
                group = []
            }
            group.append(p)
        }
    }
    if !group.isEmpty {
        out.append(try Linear(stacking: group))
    }
    return out
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

    public func Forward(_ x: gpu.Buffer<float32>, into y: gpu.Buffer<float32>) async throws {
        try await neural.RMSNorm(x, weight: try Weight.Floats(), into: y, rows: 1, cols: Weight.Count, eps: Eps)
    }
}

/// GatedMLP is Down(a(Gate·x) ⊙ Up·x): SwiGLU with .SiLU, the MLP of
/// Llama, Mistral, Qwen and most decoders since. Gate and Up are one
/// Linear, GateUp, their rows one after the other: one product makes both.
public final class GatedMLP {
    /// GateUp is the gate and up projections, as Stacked makes them.
    public let GateUp: [Linear]
    public let Down: Linear
    public let Activation: neural.Activation
    let _gateUp: gpu.Buffer<float32>
    let _gate: gpu.Buffer<float32>
    let _up: gpu.Buffer<float32>

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

    /// Forward writes the MLP of x into y, or with accumulate adds it.
    public func Forward(_ x: gpu.Buffer<float32>, into y: gpu.Buffer<float32>, accumulate: bool = false) async throws {
        try await nn.Forward(GateUp, x, into: _gateUp)
        try await neural.Gated(_up, gate: _gate, Activation, into: _up)
        try await Down.Forward(_up, into: y, accumulate: accumulate)
    }
}
