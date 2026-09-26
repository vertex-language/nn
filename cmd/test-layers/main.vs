// nn's sequence layers and gpu/neural's audio kernels against torch:
// testdata/layers.safetensors holds each case's input, weights and what
// torch.nn computed from them (testdata/oracle/layers.py), at the shapes,
// strides and paddings Kokoro uses. Every case runs on every device.
package main

import (
    "fs"
    "gpu"
    "gpu/gputest"
    "gpu/linalg"
    "gpu/neural"
    "model/safetensors"
    "nn"
    "tensor"
)

let file = try safetensors.Open(fs.Path("testdata/layers.safetensors"))

func host(_ name: string) -> [float32] {
    guard let t = file.Tensor(name) else { fatalError("no \(name) in the fixture") }
    let p = file.Bytes(t)
    var out = [float32](repeating: 0, count: t.Count)
    for i in 0..<t.Count {
        let b = 4 * i
        out[i] = float32(bitPattern: uint32(p[b]) | uint32(p[b + 1]) << 8 | uint32(p[b + 2]) << 16 | uint32(p[b + 3]) << 24)
    }
    return out
}

func shape(_ name: string) -> [int] { return file.Tensor(name)!.Shape }

func upload(_ d: gpu.Device, _ name: string) async throws -> gpu.Buffer<float32> {
    return try await d.Upload(host(name))
}

func weight(_ d: gpu.Device, _ name: string) async throws -> tensor.Tensor {
    let b = try await upload(d, name)
    return try tensor.Tensor(shape: shape(name), dtype: .F32, storage: b.View(as: uint8.self))
}

// near checks got against the fixture's want, each element within
// tol · (1 + |want|), and says the worst error it saw.
func near(_ what: string, _ d: gpu.Device, _ got: gpu.Buffer<float32>, _ want: string, tol: float64 = 1e-5) async throws {
    let g = try await got.Download()
    let w = host(want).map { float64($0) }
    var worst: float64 = 0
    if g.count == w.count {
        for i in 0..<g.count { worst = max(worst, abs(float64(g[i]) - w[i]) / (1 + abs(w[i]))) }
    }
    print("      \(what) on \(d.Name): worst \(worst)")
    gputest.Near(what, d, g, w, bound: w.map { tol * (1 + abs($0)) })
}

for d in gputest.Devices() {
    // Conv1d, at each kind Kokoro has.
    let convs: [(string, int, int, int, int, bool)] = [  // name, stride, padding, dilation, groups, bias
        ("conv.k3p1", 1, 1, 1, 1, true), ("conv.dil3", 1, 3, 3, 1, true), ("conv.s2", 2, 1, 1, 1, true),
        ("conv.noise", 6, 3, 1, 1, true), ("conv.groups", 1, 2, 1, 3, false), ("conv.1x1", 1, 0, 1, 1, false)]
    for (name, stride, pad, dil, groups, bias) in convs {
        let c = try nn.Conv1d(weight: try await weight(d, name + ".w"), bias: bias ? try await weight(d, name + ".b") : nil,
                              stride: stride, padding: pad, dilation: dil, groups: groups)
        let t = shape(name + ".x")[1]
        let y = try d.CreateBuffer(of: float32.self, count: c.Out * c.OutLength(t))
        try await c.Forward(try await upload(d, name + ".x"), length: t, into: y)
        try await near("Conv1d " + name, d, y, name + ".y")
    }

    // ConvTranspose1d: the generator's upsamplers and the depthwise pool.
    let convTs: [(string, int, int, int, int)] = [("convT.up10", 10, 5, 0, 1), ("convT.up6", 6, 3, 0, 1), ("convT.pool", 2, 1, 1, 10)]
    for (name, stride, pad, opad, groups) in convTs {
        let c = try nn.ConvTranspose1d(weight: try await weight(d, name + ".w"), bias: try await weight(d, name + ".b"),
                                       stride: stride, padding: pad, outputPadding: opad, groups: groups)
        let t = shape(name + ".x")[1]
        let y = try d.CreateBuffer(of: float32.self, count: c.Out * c.OutLength(t))
        try await c.Forward(try await upload(d, name + ".x"), length: t, into: y)
        try await near("ConvTranspose1d " + name, d, y, name + ".y")
    }

    // weight_norm.
    let wn = try await nn.WeightNormed(g: try await weight(d, "wn.g"), v: try await weight(d, "wn.v"))
    try await near("WeightNormed", d, try wn.Floats(), "wn.w")

    // A bidirectional LSTM.
    let lstm = try nn.LSTM(weightIH: try await weight(d, "lstm.weight_ih_l0"), weightHH: try await weight(d, "lstm.weight_hh_l0"),
                                 biasIH: try await weight(d, "lstm.bias_ih_l0"), biasHH: try await weight(d, "lstm.bias_hh_l0"),
                                 reverse: [try await weight(d, "lstm.weight_ih_l0_reverse"), try await weight(d, "lstm.weight_hh_l0_reverse"),
                                           try await weight(d, "lstm.bias_ih_l0_reverse"), try await weight(d, "lstm.bias_hh_l0_reverse")])
    let steps = shape("lstm.x")[0]
    let ly = try d.CreateBuffer(of: float32.self, count: steps * lstm.Out)
    try await lstm.Forward(try await upload(d, "lstm.x"), steps: steps, into: ly)
    try await near("LSTM bidirectional", d, ly, "lstm.y")

    // Norms, with a style and without.
    let x12 = try await upload(d, "inorm.x")
    let y12 = try d.CreateBuffer(of: float32.self, count: 12 * 33)
    try await nn.InstanceNorm1d(weight: try await weight(d, "inorm.w"), bias: try await weight(d, "inorm.b")).Forward(x12, channels: 12, length: 33, into: y12)
    try await near("InstanceNorm1d affine", d, y12, "inorm.y")
    let fc = try nn.Dense(weight: try await weight(d, "adain.fc.w"), bias: try await weight(d, "adain.fc.b"))
    let style = try await upload(d, "adain.s")
    try await nn.AdaIN1d(fc: fc).Forward(x12, style: style, length: 33, into: y12)
    try await near("AdaIN1d", d, y12, "adain.y")
    let xc = try await upload(d, "chnorm.x")
    let yc = try d.CreateBuffer(of: float32.self, count: 12 * 20)
    try await nn.ChannelNorm(gamma: try await weight(d, "chnorm.g"), beta: try await weight(d, "chnorm.b")).Forward(xc, channels: 12, length: 20, into: yc)
    try await near("ChannelNorm", d, yc, "chnorm.y")
    try await nn.AdaLayerNorm(fc: fc).Forward(xc, style: style, length: 20, into: yc)
    try await near("AdaLayerNorm", d, yc, "adaln.y")

    // Activations.
    let sy = try d.CreateBuffer(of: float32.self, count: 300)
    try await neural.Snake(try await upload(d, "snake.x"), alpha: try await upload(d, "snake.a"), into: sy, cols: 50)
    try await near("Snake", d, sy, "snake.y")
    let lk = try d.CreateBuffer(of: float32.self, count: 200)
    try await neural.LeakyReLU(try await upload(d, "leaky.x"), slope: 0.2, into: lk)
    try await near("LeakyReLU 0.2", d, lk, "leaky.y", tol: 0)

    // Resampling along the length, as interpolate(scale_factor:) does.
    let ups: [(string, int, float64, neural.Resize)] = [("up.n2", 3, 2, .nearest), ("up.n300", 1, 300, .nearest),
                                                        ("up.ldown", 9, 1.0 / 300, .linear), ("up.lup", 9, 300, .linear)]
    for (name, rows, factor, mode) in ups {
        let tin = shape(name + ".x")[1]
        let uy = try d.CreateBuffer(of: float32.self, count: rows * neural.UpsampleLength(tin, factor: factor))
        try await neural.Upsample(try await upload(d, name + ".x"), into: uy, rows: rows, length: tin, factor: factor, mode: mode)
        try await near("Upsample " + name, d, uy, name + ".y", tol: 0)
    }

    let cs = try d.CreateBuffer(of: float32.self, count: 9 * 400)
    try await neural.CumSum(try await upload(d, "cumsum.x"), into: cs, rows: 9, cols: 400)
    try await near("CumSum", d, cs, "cumsum.y", tol: 1e-6)
    let pd = try d.CreateBuffer(of: float32.self, count: 4 * 11)
    try await neural.Pad(try await upload(d, "pad.x"), into: pd, rows: 4, cols: 10, left: 1, right: 0, reflect: true)
    try await near("ReflectionPad1d (1, 0)", d, pd, "pad.y", tol: 0)

    // STFT: the magnitude everywhere, the angle where the magnitude says
    // there is one (wrapped: π and -π are one angle).
    let frames = neural.STFTFrames(1000, hop: 5)
    let mag = try d.CreateBuffer(of: float32.self, count: 11 * frames)
    let phase = try d.CreateBuffer(of: float32.self, count: 11 * frames)
    try await neural.STFT(try await upload(d, "stft.x"), length: 1000, n: 20, hop: 5, magnitude: mag, phase: phase)
    try await near("STFT magnitude", d, mag, "stft.mag")
    let gm = try await mag.Download(), gp = try await phase.Download()
    let wm = host("stft.mag"), wp = host("stft.phase")
    var worstPhase: float64 = 0
    var exactZeros = true
    for i in 0..<gp.count {
        if wm[i] > 1e-3 {
            var e = abs(float64(gp[i]) - float64(wp[i]))
            e = min(e, 2 * 3.14159265358979 - e)
            worstPhase = max(worstPhase, e)
        }
        // DC and Nyquist are real: their angle is 0 or π (torch's π is
        // a float32 below π, from its vectorized atan2), never ±π/2.
        let bin = i / frames
        if (bin == 0 || bin == 10) && gm[i] > 0 && abs(gp[i] - wp[i]) > 1e-6 { exactZeros = false }
    }
    print("      STFT phase on \(d.Name): worst \(worstPhase) where the magnitude exceeds 1e-3")
    gputest.Near("STFT phase", d, [float32(worstPhase)], [0], bound: [1e-3])
    gputest.Equal("STFT DC and Nyquist phase is 0 or π, as torch's", d, [exactZeros], [true])

    let iframes = shape("istft.mag")[1]
    let iy = try d.CreateBuffer(of: float32.self, count: neural.ISTFTLength(iframes, hop: 5))
    try await neural.ISTFT(magnitude: try await upload(d, "istft.mag"), phase: try await upload(d, "istft.phase"), frames: iframes, n: 20, hop: 5, into: iy)
    try await near("ISTFT", d, iy, "istft.y")

    // The generator's last step: exp of one half, sin of the other.
    let ea = try d.CreateBuffer(of: float32.self, count: 11 * 60)
    try await neural.Activate(try await upload(d, "istft.a"), .Exp, into: ea)
    try await near("Activate .Exp", d, ea, "istft.mag", tol: 2e-7)
    try await neural.Activate(try await upload(d, "istft.b"), .Sin, into: ea)
    try await near("Activate .Sin", d, ea, "istft.phase", tol: 2e-7)
    _ = linalg.Shape(m: 1, n: 1, k: 1)
}

gputest.Done()
