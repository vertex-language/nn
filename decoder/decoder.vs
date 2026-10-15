// The decoder skeleton every causal network shares: an embedding, blocks
// that each change the residual stream, a final norm and a head -- run
// over a batch of rows in runs of sequences, with a memory between
// batches. A family's network is its blocks and the knobs below; the
// batching, the memory and the outputs are written once, here.
package decoder

import (
    "gpu"
    "gpu/neural"
    "nn"
    "nn/cache"
)

/// DecoderError is a batch or a shape a network cannot run.
public enum DecoderError: Error, CustomStringConvertible {
    case invalid(string)

    public var description: string {
        switch self {
        case .invalid(let why): return "decoder: " + why
        }
    }
}

/// Input is what a batch's rows start from: tokens, or embeddings that
/// stand in for them (an image's or audio's, from an adapter).
public enum Input {
    case tokens([int32])
    case embeddings(gpu.Buffer<float32>)
}

/// Block is one layer of a decoder: what it does to the residual stream of
/// a batch's rows, attending through runs of sequences, each to its own
/// cache for this layer.
public protocol Block: AnyObject {
    func Forward(_ x: gpu.Buffer<float32>, rows n: int, runs: [nn.Run]) async throws
}

/// Standard is the transformer layer most decoders are, in the forms
/// llama.cpp's graphs take:
///
///     h = AttnNorm(x);  a = Attention(h);  a = PostAttnNorm(a)
///     x = x + ResidualScale·a
///     f = FFN(FFNNorm(x));  f = PostFFNNorm(f)
///     x = x + ResidualScale·f
///
/// with Parallel, the FFN reads FFNNorm(x) of the layer's input -- or AttnNorm's
/// h where FFNNorm is nil -- and both are added at once (GPT-J's and
/// Phi-2's, Command R's). A norm left nil is not applied.
public final class Standard: Block {
    public var AttnNorm: nn.Norm?
    public var Attention: nn.Attention
    public var PostAttnNorm: nn.Norm? = nil
    public var FFNNorm: nn.Norm?
    public var FFN: nn.FFN
    public var PostFFNNorm: nn.Norm? = nil
    public var Parallel: bool = false
    public var ResidualScale: float32 = 1
    let _ws: Workspace

    public init(attnNorm: nn.Norm?, attention: nn.Attention, ffnNorm: nn.Norm?, ffn: nn.FFN) {
        self.AttnNorm = attnNorm
        self.Attention = attention
        self.FFNNorm = ffnNorm
        self.FFN = ffn
        self._ws = Workspace()
    }

    public func Forward(_ x: gpu.Buffer<float32>, rows n: int, runs: [nn.Run]) async throws {
        let dim = Attention.O.Out
        let d = Attention.O.Device
        let h = try _ws.Floats(d, 0, n * dim)
        let direct = PostAttnNorm == nil && PostFFNNorm == nil && ResidualScale == 1
        if let norm = AttnNorm {
            try await norm.Forward(x, rows: n, into: h)
        } else {
            try await h.Copy(from: x.Slice(from: 0, count: n * dim))
        }
        if Parallel {
            // Both branches from the layer's input, added together.
            let f = try _ws.Floats(d, 1, n * dim)
            if let norm = FFNNorm {
                try await norm.Forward(x, rows: n, into: f)
            } else {
                try await f.Copy(from: h)
            }
            let a = try _ws.Floats(d, 2, n * dim)
            try await Attention.Forward(h, rows: n, runs: runs, into: a)
            try await _finish(a, PostAttnNorm, x, n, dim)
            try await FFN.Forward(f, rows: n, into: h, accumulate: false)
            try await _finish(h, PostFFNNorm, x, n, dim)
            return
        }
        if direct {
            // The residual added by the projections that make each branch.
            try await Attention.Forward(h, rows: n, runs: runs, into: x, accumulate: true)
        } else {
            let a = try _ws.Floats(d, 2, n * dim)
            try await Attention.Forward(h, rows: n, runs: runs, into: a)
            try await _finish(a, PostAttnNorm, x, n, dim)
        }
        if let norm = FFNNorm {
            try await norm.Forward(x, rows: n, into: h)
        } else {
            try await h.Copy(from: x.Slice(from: 0, count: n * dim))
        }
        if direct {
            try await FFN.Forward(h, rows: n, into: x, accumulate: true)
        } else {
            let f = try _ws.Floats(d, 1, n * dim)
            try await FFN.Forward(h, rows: n, into: f, accumulate: false)
            try await _finish(f, PostFFNNorm, x, n, dim)
        }
    }

    // _finish adds a branch's output to the residual stream: through its
    // post norm, if it has one, and times ResidualScale.
    func _finish(_ b: gpu.Buffer<float32>, _ post: nn.Norm?, _ x: gpu.Buffer<float32>, _ n: int, _ dim: int) async throws {
        if let norm = post {
            try await norm.Forward(b, rows: n, into: b)
        }
        try await neural.Axpby(ResidualScale, b, 1, x.Slice(from: 0, count: n * dim), into: x.Slice(from: 0, count: n * dim))
    }
}

/// Network is a decoder's weights on a device and what decoding needs
/// besides: an embedding (times EmbedScale), blocks, a final norm, and a
/// head (its logits times LogitScale), over a memory of sequences.
public final class Network {
    public let Device: gpu.Device
    public let Dim: int
    /// Context is how many positions a sequence may hold.
    public let Context: int
    public let Embed: nn.Embedding
    public var EmbedScale: float32 = 1
    public let Blocks: [Block]
    /// Attentions are each block's attention, for the memory's shape:
    /// KVHeads and HeadDim of each layer's cache.
    public let KVHeads: int
    public let HeadDim: int
    public let Norm: nn.Norm?
    public let Output: nn.Linear
    public var LogitScale: float32 = 1
    let _ws: Workspace
    var _tokens: gpu.Buffer<int32>? = nil

    public init(on d: gpu.Device, dim: int, context: int, embed: nn.Embedding, blocks: [Block], kvHeads: int, headDim: int,
                norm: nn.Norm?, output: nn.Linear) {
        self.Device = d
        self.Dim = dim
        self.Context = context
        self.Embed = embed
        self.Blocks = blocks
        self.KVHeads = kvHeads
        self.HeadDim = headDim
        self.Norm = norm
        self.Output = output
        self._ws = Workspace()
    }

    /// Vocab is how many logits a row has.
    public var Vocab: int { return Output.Out }

    /// Memory is a memory for this network: sequences sequences of
    /// capacity positions each (the network's context when nil).
    public func Memory(sequences: int = 1, capacity: int? = nil) -> cache.KV {
        return cache.KV(on: Device, layers: Blocks.count, kvHeads: KVHeads, headDim: HeadDim,
                        capacity: capacity ?? Context, sequences: sequences)
    }

    /// Tokens is a buffer of count token ids on the network's device: where
    /// a decoding loop the device runs keeps its sequence.
    public func Tokens(count: int) throws -> gpu.Buffer<int32> {
        // A float's bytes viewed as ints: CreateBuffer(of: int32.self)
        // specialized here fails the compiler's SIL verifier.
        return try Device.CreateBuffer(of: float32.self, count: count).View(as: int32.self)
    }

    /// Forward runs a batch through the network: input's rows, in runs of
    /// sequences, each added to its sequence's memory (dropping what the
    /// sequence held from the run's position on). It is the wanted rows'
    /// outputs, one after another: logits, or with hidden the final
    /// normalized hidden states. The buffer is the network's, good until
    /// the next Forward or Step.
    public func Forward(_ input: Input, runs: [cache.Run], wanted: [int], hidden: bool = false,
                        memory: cache.KV) async throws -> gpu.Buffer<float32> {
        var n = 0
        for r in runs {
            if r.Row != n {
                throw DecoderError.invalid("runs that do not cover the batch's rows in order")
            }
            if r.Position > memory.Length(r.Sequence) {
                throw DecoderError.invalid("sequence \(r.Sequence) at position \(r.Position) past the \(memory.Length(r.Sequence)) it holds")
            }
            if r.Position + r.Count > memory.Capacity {
                throw DecoderError.invalid("sequence \(r.Sequence) past a memory of \(memory.Capacity) positions")
            }
            n += r.Count
        }
        for w in wanted where w < 0 || w >= n {
            throw DecoderError.invalid("row \(w) wanted of a batch of \(n)")
        }
        let dim = Dim
        let x = try _ws.Floats(Device, 0, max(n, 1) * dim)
        switch input {
        case .tokens(let t):
            if t.count != n { throw DecoderError.invalid("\(t.count) tokens for \(n) rows") }
            if n == 1 {
                try await Embed.Lookup(int(t[0]), into: x)
            } else if n > 1 {
                let ids = try _ids(n)
                try await ids.Upload(t)
                try await Embed.Lookup(tokens: ids, first: 0, count: n, into: x)
            }
        case .embeddings(let e):
            if e.count < n * dim { throw DecoderError.invalid("embeddings of \(e.count) floats for \(n) rows of \(dim)") }
            try await x.Copy(from: e.Slice(from: 0, count: n * dim))
        }
        if n == 0 {
            return x.Slice(from: 0, count: 0)
        }
        if EmbedScale != 1 {
            switch input {
            case .tokens(_): try await neural.Axpby(EmbedScale, x, 0, x, into: x)
            default: break
            }
        }
        for r in runs {
            memory.Truncate(r.Sequence, from: r.Position)
        }
        try await _blocks(x, n, runs, memory)
        for r in runs {
            try memory.Wrote(r.Sequence, through: r.Position + r.Count)
        }
        return try await _outputs(x, n, wanted, hidden)
    }

    /// Step runs one token through the network that the device chose:
    /// tokens[index], at position of sequence -- greedy decoding's next
    /// token, looked up with nothing read back -- and is its logits.
    public func Step(tokens: gpu.Buffer<int32>, at index: int, position: int, sequence: int = 0,
                     memory: cache.KV) async throws -> gpu.Buffer<float32> {
        if position > memory.Length(sequence) || position >= memory.Capacity {
            throw DecoderError.invalid("sequence \(sequence) at position \(position) of \(memory.Length(sequence)) held")
        }
        memory.Truncate(sequence, from: position)
        let x = try _ws.Floats(Device, 0, Dim)
        try await Embed.Lookup(tokens: tokens, at: index, into: x)
        if EmbedScale != 1 {
            try await neural.Axpby(EmbedScale, x, 0, x, into: x)
        }
        let runs = [cache.Run(row: 0, count: 1, position: position, sequence: sequence)]
        try await _blocks(x, 1, runs, memory)
        try memory.Wrote(sequence, through: position + 1)
        return try await _outputs(x, 1, [0], false)
    }

    func _blocks(_ x: gpu.Buffer<float32>, _ n: int, _ runs: [cache.Run], _ memory: cache.KV) async throws {
        for i in 0..<Blocks.count {
            var nr: [nn.Run] = []
            for r in runs {
                nr.append(nn.Run(row: r.Row, count: r.Count, position: r.Position, cache: try memory.Layer(i, sequence: r.Sequence)))
            }
            try await Blocks[i].Forward(x, rows: n, runs: nr)
        }
    }

    // _outputs is the wanted rows of x through the final norm, and the
    // head unless hidden.
    func _outputs(_ x: gpu.Buffer<float32>, _ n: int, _ wanted: [int], _ hidden: bool) async throws -> gpu.Buffer<float32> {
        let dim = Dim
        let w = wanted.count
        var all = w == n
        for j in 0..<w where wanted[j] != j {
            all = false
        }
        var rows = x.Slice(from: 0, count: w * dim)
        if !all {
            let g = try _ws.Floats(Device, 1, max(w, 1) * dim)
            for j in 0..<w {
                try await g.Slice(from: j * dim, count: dim).Copy(from: x.Slice(from: wanted[j] * dim, count: dim))
            }
            rows = g.Slice(from: 0, count: w * dim)
        }
        let normed = try _ws.Floats(Device, 2, max(w, 1) * dim).Slice(from: 0, count: w * dim)
        if let norm = Norm {
            try await norm.Forward(rows, rows: w, into: normed)
        } else {
            try await normed.Copy(from: rows)
        }
        if hidden {
            return normed
        }
        let out = try _ws.Floats(Device, 3, max(w, 1) * Vocab).Slice(from: 0, count: w * Vocab)
        try await Output.Forward(normed, rows: w, into: out)
        if LogitScale != 1 {
            try await neural.Axpby(LogitScale, out, 0, out, into: out)
        }
        return out
    }

    func _ids(_ n: int) throws -> gpu.Buffer<int32> {
        if _tokens == nil || _tokens!.count < n {
            _tokens = try Tokens(count: n)
        }
        return _tokens!.Slice(from: 0, count: n)
    }
}

/// Workspace is a layer's or a network's buffers between its steps, by
/// slot, each grown to the most it has needed. Its allocations are apart
/// from the async methods that use them because the compiler's SIL
/// verifier rejects CreateBuffer specialized inside a branch of an async
/// function.
public final class Workspace {
    var _bufs: [gpu.Buffer<float32>] = []
    var _have: [bool] = []

    public init() {}

    /// Floats is slot's buffer, count floats long (a view of what it has).
    public func Floats(_ d: gpu.Device, _ slot: int, _ count: int) throws -> gpu.Buffer<float32> {
        while _have.count <= slot {
            _have.append(false)
            _bufs.append(try d.CreateBuffer(of: float32.self, count: 1))
        }
        if !_have[slot] || _bufs[slot].count < count {
            _bufs[slot] = try d.CreateBuffer(of: float32.self, count: max(count, 1))
            _have[slot] = true
        }
        return _bufs[slot].Slice(from: 0, count: count)
    }
}
