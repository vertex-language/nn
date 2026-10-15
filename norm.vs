// Norms and MLPs beyond Llama's: LayerNorm with or without its weight and
// bias, and an MLP of one projection up, an activation, and one down.
package nn

import (
    "gpu"
    "gpu/neural"
)

/// NormKind is how a Norm scales a row: by its root mean square, or to
/// zero mean and unit variance.
public enum NormKind: Equatable {
    case rms
    case layer
}

/// Norm normalizes each row of x -- RMSNorm or LayerNorm -- then scales it
/// by a weight and shifts it by a bias where it has them.
public struct Norm {
    public let Kind: NormKind
    /// Weight is the scale: ones for a norm without one (OLMo's).
    public let Weight: gpu.Buffer<float32>
    public let Bias: gpu.Buffer<float32>?
    public let Eps: float32

    public init(_ kind: NormKind, weight: gpu.Buffer<float32>, bias: gpu.Buffer<float32>? = nil, eps: float32) {
        self.Kind = kind
        self.Weight = weight
        self.Bias = bias
        self.Eps = eps
    }

    /// Plain is a norm of count elements a row with no weight or bias.
    public static func Plain(_ kind: NormKind, count: int, on d: gpu.Device, eps: float32) async throws -> Norm {
        return Norm(kind, weight: try await d.Upload([float32](repeating: 1, count: count)), eps: eps)
    }

    /// Count is how many elements a row has.
    public var Count: int { return Weight.count }

    /// Weights is the scale, as kernels take it.
    public func Weights() throws -> gpu.Buffer<float32> {
        return Weight
    }

    /// Forward normalizes rows rows of x into y.
    public func Forward(_ x: gpu.Buffer<float32>, rows: int = 1, into y: gpu.Buffer<float32>) async throws {
        let cols = Count
        if Kind == .rms && Bias == nil {
            try await neural.RMSNorm(x, weight: Weight, into: y, rows: rows, cols: cols, eps: Eps)
            return
        }
        if rows == 0 { return }
        try await y.Slice(from: 0, count: rows * cols).Copy(from: x.Slice(from: 0, count: rows * cols))
        try await _segmentNorm.Launch(y, Weight, Bias ?? Weight, Bias != nil, Kind == .layer, cols, 0, cols, 1, true, Eps,
                                      over: rows * 64, workgroup: 64)
    }
}

/// FFN is a transformer layer's feed-forward network: what GatedMLP and
/// MLP are to a block.
public protocol FFN: AnyObject {
    /// Forward writes the network's output for n rows of x into y's, or
    /// with accumulate adds it.
    func Forward(_ x: gpu.Buffer<float32>, rows n: int, into y: gpu.Buffer<float32>, accumulate: bool) async throws
}



/// MLP is Down(a(Up·x)): GPT-2's, Phi-2's, StarCoder 2's, Nemotron's.
public final class MLP: FFN {
    public let Up: Linear
    public let Down: Linear
    public let Activation: neural.Activation
    let _scratch: _Scratch

    public init(up: Linear, down: Linear, activation: neural.Activation) throws {
        if up.Out != down.In {
            throw LayerError.unsupported("an MLP up to \(up.Out) and down from \(down.In)")
        }
        self.Up = up
        self.Down = down
        self.Activation = activation
        self._scratch = _Scratch()
    }

    public func Forward(_ x: gpu.Buffer<float32>, rows n: int, into y: gpu.Buffer<float32>, accumulate: bool) async throws {
        let h = try _scratch.Floats(Up.Device, n * Up.Out)
        try await Up.Forward(x, rows: n, into: h)
        try await neural.Activate(h, Activation, into: h)
        try await Down.Forward(h, rows: n, into: y, accumulate: accumulate)
    }
}
