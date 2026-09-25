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

import "gpu"
import "gpu/dtype"
import "gpu/linalg"
import "gpu/neural"
import "tensor"

/// LayerError is a layer given what it cannot take.
public enum LayerError: Error {
    case unsupported(string)

    public var Message: string {
        switch self {
        case .unsupported(let why): return "nn: " + why
        }
    }
}

/// Linear is y = W·x, W a [out, in] weight: float32, or q4_0 or q8_0
/// blocks decoded in place as the product is taken.
public struct Linear {
    public let Weight: tensor.Tensor

    public init(_ weight: tensor.Tensor) throws {
        if weight.Shape.count != 2 {
            throw LayerError.unsupported("a Linear's weight is [out, in], not \(weight.Shape)")
        }
        self.Weight = weight
    }

    public var In: int { return Weight.Shape[1] }
    public var Out: int { return Weight.Shape[0] }

    /// Forward writes W·x into y: x has In elements, y Out.
    public func Forward(_ x: gpu.Buffer<float32>, into y: gpu.Buffer<float32>) async throws {
        switch Weight.DType {
        case .F32:
            try await linalg.Gemv(try Weight.Floats(), x, into: y, m: Out, k: In)
        case .Q4_0:
            try await linalg.Gemv(Weight.Storage, dtype.Q4_0(), x, into: y, m: Out, k: In)
        case .Q8_0:
            try await linalg.Gemv(Weight.Storage, dtype.Q8_0(), x, into: y, m: Out, k: In)
        default:
            throw LayerError.unsupported("a Linear of \(Weight.DType.Name)")
        }
    }
}

/// Embedding is a table of vectors, one a token: a [count, dim] weight,
/// float32 or block-quantized.
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
        case .Q4_0:
            try await dtype.Dequantize(Weight.Storage, dtype.Q4_0(), at: at, count: Dim, into: x)
        case .Q8_0:
            try await dtype.Dequantize(Weight.Storage, dtype.Q8_0(), at: at, count: Dim, into: x)
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
/// Llama, Mistral, Qwen and most decoders since.
public final class GatedMLP {
    public let Gate: Linear
    public let Up: Linear
    public let Down: Linear
    public let Activation: neural.Activation
    let _gate: gpu.Buffer<float32>
    let _up: gpu.Buffer<float32>

    public init(gate: Linear, up: Linear, down: Linear, activation: neural.Activation) throws {
        self.Gate = gate
        self.Up = up
        self.Down = down
        self.Activation = activation
        let d = gate.Weight.Device
        self._gate = try d.CreateBuffer(of: float32.self, count: gate.Out)
        self._up = try d.CreateBuffer(of: float32.self, count: up.Out)
    }

    public func Forward(_ x: gpu.Buffer<float32>, into y: gpu.Buffer<float32>) async throws {
        try await Gate.Forward(x, into: _gate)
        try await Up.Forward(x, into: _up)
        try await neural.Gated(_up, gate: _gate, Activation, into: _up)
        try await Down.Forward(_up, into: y)
    }
}
