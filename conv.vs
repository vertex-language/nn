// Layers over sequences, as speech and audio networks use them: Dense
// (a Linear with a bias, over many rows at once), Conv1d and
// ConvTranspose1d, their weight normalization, a bidirectional LSTM, and
// the normalizations style enters through (InstanceNorm1d, AdaIN1d,
// ChannelNorm, AdaLayerNorm).
//
// Activations are float32 [channels, length] row-major buffers --
// PyTorch's NCL at batch 1 -- except where a layer says it takes rows of
// features ([length, features]: Dense, LSTM). Each layer is its
// torch.nn namesake, and loads the same weights in the same shapes.
package nn

import (
    "gpu"
    "gpu/linalg"
    "gpu/neural"
    "tensor"
)

func floats(_ t: tensor.Tensor, _ what: string) throws -> gpu.Buffer<float32> {
    if t.DType != .F32 {
        throw LayerError.unsupported("\(what) of \(t.DType.Name); convert it to float32 first (tensor.ToF32)")
    }
    return try t.Floats()
}

/// Dense is torch.nn.Linear: y = x·Wᵀ + b over rows of x, W [out, in]
/// float32 and b [out] or nil.
public final class Dense {
    public let Weight: gpu.Buffer<float32>
    public let Bias: gpu.Buffer<float32>?
    public let In: int
    public let Out: int

    public init(weight: tensor.Tensor, bias: tensor.Tensor? = nil) throws {
        if weight.Shape.count != 2 {
            throw LayerError.unsupported("a Dense weight is [out, in], not \(weight.Shape)")
        }
        self.Weight = try floats(weight, "a Dense weight")
        self.Out = weight.Shape[0]
        self.In = weight.Shape[1]
        if let b = bias {
            if b.Count != weight.Shape[0] {
                throw LayerError.unsupported("a Dense bias of \(b.Count) for \(weight.Shape[0]) outputs")
            }
            self.Bias = try floats(b, "a Dense bias")
        } else {
            self.Bias = nil
        }
    }

    /// Forward writes rows x Out into y from rows x In of x.
    public func Forward(_ x: gpu.Buffer<float32>, rows: int = 1, into y: gpu.Buffer<float32>) async throws {
        try await linalg.Matmul(x, Weight, into: y, linalg.Shape(m: rows, n: Out, k: In, transposeB: true),
                                linalg.Epilogue<float32>(scale: 1, bias: Bias, residual: nil, activation: .None))
    }
}

/// WeightNormed is the weight torch.nn.utils.weight_norm keeps as
/// weight_g and weight_v: g · v / ‖v‖, over v's dimensions after the
/// first. Checkpoints store the two; a layer is built from this.
public func WeightNormed(g: tensor.Tensor, v: tensor.Tensor) async throws -> tensor.Tensor {
    let rows = v.Shape[0]
    if g.Count != rows {
        throw LayerError.unsupported("weight_g of \(g.Shape) for weight_v of \(v.Shape)")
    }
    let d = v.Device
    let w = try d.CreateBuffer(of: float32.self, count: v.Count)
    try await neural.WeightNorm(g: try floats(g, "weight_g"), v: try floats(v, "weight_v"), into: w, rows: rows)
    return try tensor.Tensor(shape: v.Shape, dtype: .F32, storage: w.View(as: uint8.self))
}

/// Conv1d is torch.nn.Conv1d: weight [out, in/groups, kernel], bias [out]
/// or nil.
public final class Conv1d {
    public let Weight: gpu.Buffer<float32>
    public let Bias: gpu.Buffer<float32>?
    public let In: int
    public let Out: int
    public let Kernel: int
    public let Stride: int
    public let Padding: int
    public let Dilation: int
    public let Groups: int

    public init(weight: tensor.Tensor, bias: tensor.Tensor? = nil, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1) throws {
        if weight.Shape.count != 3 {
            throw LayerError.unsupported("a Conv1d weight is [out, in/groups, kernel], not \(weight.Shape)")
        }
        self.Weight = try floats(weight, "a Conv1d weight")
        self.Bias = bias == nil ? nil : try floats(bias!, "a Conv1d bias")
        self.Out = weight.Shape[0]
        self.In = weight.Shape[1] * groups
        self.Kernel = weight.Shape[2]
        self.Stride = stride
        self.Padding = padding
        self.Dilation = dilation
        self.Groups = groups
        if Out % groups != 0 || (bias != nil && bias!.Count != Out) {
            throw LayerError.unsupported("a Conv1d of \(weight.Shape) in \(groups) groups with a bias of \(bias?.Count ?? 0)")
        }
    }

    /// OutLength is how long the output of length samples is.
    public func OutLength(_ length: int) -> int {
        return neural.Conv1dLength(length, kernel: Kernel, stride: Stride, padding: Padding, dilation: Dilation)
    }

    /// Forward writes [Out, OutLength(length)] into y from [In, length].
    public func Forward(_ x: gpu.Buffer<float32>, length: int, into y: gpu.Buffer<float32>) async throws {
        try await neural.Conv1d(x, weight: Weight, bias: Bias, into: y,
                                neural.Conv1dShape(in: In, out: Out, length: length, kernel: Kernel, stride: Stride, padding: Padding, dilation: Dilation, groups: Groups))
    }
}

/// ConvTranspose1d is torch.nn.ConvTranspose1d: weight [in, out/groups,
/// kernel], bias [out] or nil.
public final class ConvTranspose1d {
    public let Weight: gpu.Buffer<float32>
    public let Bias: gpu.Buffer<float32>?
    public let In: int
    public let Out: int
    public let Kernel: int
    public let Stride: int
    public let Padding: int
    public let OutputPadding: int
    public let Groups: int

    public init(weight: tensor.Tensor, bias: tensor.Tensor? = nil, stride: int = 1, padding: int = 0, outputPadding: int = 0, groups: int = 1) throws {
        if weight.Shape.count != 3 {
            throw LayerError.unsupported("a ConvTranspose1d weight is [in, out/groups, kernel], not \(weight.Shape)")
        }
        self.Weight = try floats(weight, "a ConvTranspose1d weight")
        self.Bias = bias == nil ? nil : try floats(bias!, "a ConvTranspose1d bias")
        self.In = weight.Shape[0]
        self.Out = weight.Shape[1] * groups
        self.Kernel = weight.Shape[2]
        self.Stride = stride
        self.Padding = padding
        self.OutputPadding = outputPadding
        self.Groups = groups
        if In % groups != 0 || (bias != nil && bias!.Count != Out) {
            throw LayerError.unsupported("a ConvTranspose1d of \(weight.Shape) in \(groups) groups with a bias of \(bias?.Count ?? 0)")
        }
    }

    public func OutLength(_ length: int) -> int {
        return neural.ConvTranspose1dLength(length, kernel: Kernel, stride: Stride, padding: Padding, outputPadding: OutputPadding)
    }

    /// Forward writes [Out, OutLength(length)] into y from [In, length].
    public func Forward(_ x: gpu.Buffer<float32>, length: int, into y: gpu.Buffer<float32>) async throws {
        try await neural.ConvTranspose1d(x, weight: Weight, bias: Bias, into: y,
                                         neural.Conv1dShape(in: In, out: Out, length: length, kernel: Kernel, stride: Stride, padding: Padding,
                                                            groups: Groups, outputPadding: OutputPadding))
    }
}

/// LSTM is torch.nn.LSTM, one layer, batch_first, forward only or
/// bidirectional: weights as PyTorch names them (weight_ih_l0 [4H, in],
/// weight_hh_l0 [4H, H], bias_ih_l0 and bias_hh_l0 [4H], and the same
/// with _reverse). The gates are input, forget, cell, output.
public final class LSTM {
    public let Input: int
    public let Hidden: int
    /// Directions is 1, or 2 for a bidirectional LSTM.
    public let Directions: int
    let _ih: [gpu.Buffer<float32>]
    let _hh: [gpu.Buffer<float32>]
    let _bih: [gpu.Buffer<float32>]
    let _bhh: [gpu.Buffer<float32>]
    // b_ih + b_hh, summed on the first Forward (vsc cannot yet lower an
    // async initializer; vsc_TODO.md).
    var _bias: [gpu.Buffer<float32>] = []

    /// init takes the forward direction's weights, and the reverse
    /// direction's for a bidirectional LSTM.
    public init(weightIH: tensor.Tensor, weightHH: tensor.Tensor, biasIH: tensor.Tensor, biasHH: tensor.Tensor,
                reverse: [tensor.Tensor] = []) throws {
        if weightIH.Shape.count != 2 || weightHH.Shape.count != 2 || weightIH.Shape[0] % 4 != 0 {
            throw LayerError.unsupported("LSTM weights of \(weightIH.Shape) and \(weightHH.Shape)")
        }
        let hidden = weightIH.Shape[0] / 4
        if weightHH.Shape[0] != 4 * hidden || weightHH.Shape[1] != hidden || biasIH.Count != 4 * hidden || biasHH.Count != 4 * hidden {
            throw LayerError.unsupported("LSTM weights of \(weightIH.Shape), \(weightHH.Shape) and biases of \(biasIH.Count) and \(biasHH.Count)")
        }
        if !reverse.isEmpty && reverse.count != 4 {
            throw LayerError.unsupported("a reverse LSTM direction is its four weights")
        }
        self.Input = weightIH.Shape[1]
        self.Hidden = hidden
        self.Directions = reverse.isEmpty ? 1 : 2
        var ih: [gpu.Buffer<float32>] = []
        var hh: [gpu.Buffer<float32>] = []
        var bih: [gpu.Buffer<float32>] = []
        var bhh: [gpu.Buffer<float32>] = []
        for dir in 0..<(reverse.isEmpty ? 1 : 2) {
            let w = dir == 0 ? [weightIH, weightHH, biasIH, biasHH] : reverse
            if w[0].Shape != weightIH.Shape || w[1].Shape != weightHH.Shape || w[2].Count != 4 * hidden || w[3].Count != 4 * hidden {
                throw LayerError.unsupported("an LSTM's reverse weights are its forward weights' shapes")
            }
            ih.append(try floats(w[0], "an LSTM weight"))
            hh.append(try floats(w[1], "an LSTM weight"))
            bih.append(try floats(w[2], "an LSTM bias"))
            bhh.append(try floats(w[3], "an LSTM bias"))
        }
        self._ih = ih
        self._hh = hh
        self._bih = bih
        self._bhh = bhh
    }

    /// Out is how many features a step's output has: Hidden a direction.
    public var Out: int { return Hidden * Directions }

    /// Forward runs the LSTM over steps rows of x (steps x Input) from a
    /// zero state, writing steps x Out into y: the forward direction's
    /// output, then the reverse's, in each row.
    public func Forward(_ x: gpu.Buffer<float32>, steps: int, into y: gpu.Buffer<float32>) async throws {
        if steps == 0 {
            return
        }
        let h4 = 4 * Hidden
        let d = x.Device
        if _bias.isEmpty {
            for dir in 0..<Directions {
                let b = try d.CreateBuffer(of: float32.self, count: h4)
                try await neural.Axpby(1, _bih[dir], 1, _bhh[dir], into: b)
                _bias.append(b)
            }
        }
        let gates = try d.CreateBuffer(of: float32.self, count: steps * h4)
        let cell = try d.CreateBuffer(of: float32.self, count: Hidden)
        for dir in 0..<Directions {
            // Every step's input projection at once; the recurrence adds
            // W_hh·h to each in turn.
            try await linalg.Matmul(x, _ih[dir], into: gates, linalg.Shape(m: steps, n: h4, k: Input, transposeB: true),
                                    linalg.Epilogue<float32>(scale: 1, bias: _bias[dir], residual: nil, activation: .None))
            var prev = -1
            for s in 0..<steps {
                let t = dir == 0 ? s : steps - 1 - s
                let g = gates.Slice(from: t * h4, count: h4)
                if prev >= 0 {
                    try await linalg.Gemv(_hh[dir], y.Slice(from: prev * Out + dir * Hidden, count: Hidden), into: g,
                                          m: h4, k: Hidden, accumulate: true)
                }
                try await neural.LSTMCell(gates: g, cell: cell, into: y.Slice(from: t * Out + dir * Hidden, count: Hidden),
                                          hidden: Hidden, first: prev < 0)
                prev = t
            }
        }
    }
}

/// InstanceNorm1d is torch.nn.InstanceNorm1d: each channel normalized
/// over its length, then scaled and shifted by weight and bias if it has
/// them (affine).
public final class InstanceNorm1d {
    public let Weight: gpu.Buffer<float32>?
    public let Bias: gpu.Buffer<float32>?
    public let Eps: float32

    public init(weight: tensor.Tensor? = nil, bias: tensor.Tensor? = nil, eps: float32 = 1e-5) throws {
        self.Weight = weight == nil ? nil : try floats(weight!, "an InstanceNorm1d weight")
        self.Bias = bias == nil ? nil : try floats(bias!, "an InstanceNorm1d bias")
        self.Eps = eps
    }

    public func Forward(_ x: gpu.Buffer<float32>, channels: int, length: int, into y: gpu.Buffer<float32>) async throws {
        try await neural.InstanceNorm(x, weight: Weight, bias: Bias, into: y, rows: channels, cols: length, eps: Eps)
    }
}

/// AdaIN1d is adaptive instance norm (StyleTTS2's): a style vector's
/// Dense makes a gamma and beta a channel, and the instance-normalized
/// input becomes (1 + gamma) · x + beta.
public final class AdaIN1d {
    public let FC: Dense
    public let Norm: InstanceNorm1d

    /// init takes the style Dense, and the norm's affine weights if the
    /// checkpoint has them (Kokoro's have none: PyTorch's defaults, 1 and 0).
    public init(fc: Dense, norm: InstanceNorm1d? = nil) throws {
        self.FC = fc
        self.Norm = norm ?? (try InstanceNorm1d())
    }

    public var Channels: int { return FC.Out / 2 }

    /// Forward writes [Channels, length] into y from x of that shape and
    /// the style vector (FC.In).
    public func Forward(_ x: gpu.Buffer<float32>, style: gpu.Buffer<float32>, length: int, into y: gpu.Buffer<float32>) async throws {
        let h = try x.Device.CreateBuffer(of: float32.self, count: FC.Out)
        try await FC.Forward(style, into: h)
        try await Norm.Forward(x, channels: Channels, length: length, into: y)
        try await neural.Modulate(y, gamma: h.Slice(from: 0, count: Channels), beta: h.Slice(from: Channels, count: Channels), into: y, cols: length)
    }
}

/// ChannelNorm is layer norm across the channels of [channels, length]
/// (StyleTTS2's LayerNorm module: F.layer_norm over the channel axis),
/// scaled by gamma and shifted by beta a channel when it has them.
public final class ChannelNorm {
    public let Gamma: gpu.Buffer<float32>?
    public let Beta: gpu.Buffer<float32>?
    public let Eps: float32

    public init(gamma: tensor.Tensor? = nil, beta: tensor.Tensor? = nil, eps: float32 = 1e-5) throws {
        self.Gamma = gamma == nil ? nil : try floats(gamma!, "a ChannelNorm gamma")
        self.Beta = beta == nil ? nil : try floats(beta!, "a ChannelNorm beta")
        self.Eps = eps
    }

    public func Forward(_ x: gpu.Buffer<float32>, channels: int, length: int, into y: gpu.Buffer<float32>) async throws {
        try await neural.ChannelNorm(x, weight: Gamma, bias: Beta, into: y, rows: channels, cols: length, eps: Eps)
    }
}

/// AdaLayerNorm is StyleTTS2's adaptive layer norm: ChannelNorm without
/// its own scale, then a style's gamma and beta, (1 + gamma) · x + beta.
public final class AdaLayerNorm {
    public let FC: Dense
    public let Eps: float32

    public init(fc: Dense, eps: float32 = 1e-5) {
        self.FC = fc
        self.Eps = eps
    }

    public var Channels: int { return FC.Out / 2 }

    public func Forward(_ x: gpu.Buffer<float32>, style: gpu.Buffer<float32>, length: int, into y: gpu.Buffer<float32>) async throws {
        let h = try x.Device.CreateBuffer(of: float32.self, count: FC.Out)
        try await FC.Forward(style, into: h)
        try await neural.ChannelNorm(x, weight: nil, bias: nil, into: y, rows: Channels, cols: length, eps: Eps)
        try await neural.Modulate(y, gamma: h.Slice(from: 0, count: Channels), beta: h.Slice(from: Channels, count: Channels), into: y, cols: length)
    }
}
