// Package cache is what a model keeps between the batches it decodes:
// for attention, the keys and values of each sequence's tokens so far.
//
// A Memory holds several sequences, each its own positions from 0 --
// the conversations a server decodes side by side, or a prompt and its
// speculative continuations -- and every operation on it is a
// sequence's: how long it is, and dropping its positions from one on (a
// rejected draft, an edited prompt). KV is the memory of full attention;
// windowed, recurrent and latent memories join it as families need them.
package cache

import (
    "gpu"
    "nn"
)

/// MemoryError is a sequence or position a memory does not hold.
public enum MemoryError: Error, CustomStringConvertible {
    case unknownSequence(int, int)
    case full(int, int)

    public var description: string {
        switch self {
        case .unknownSequence(let s, let n): return "cache: sequence \(s) of a memory of \(n)"
        case .full(let p, let c): return "cache: position \(p) past a memory of \(c)"
        }
    }
}

/// MaxSequences is the most sequences a memory holds.
public let MaxSequences = 256

/// Memory is a model's state between batches, by sequence.
public protocol Memory: AnyObject {
    /// Sequences is how many sequences it holds, numbered from 0.
    var Sequences: int { get }
    /// Capacity is how many positions a sequence may hold.
    var Capacity: int { get }
    /// Length is how many positions sequence s holds: the position its
    /// next token takes.
    func Length(_ s: int) -> int
    /// Truncate drops sequence s's positions from `from` on.
    func Truncate(_ s: int, from: int)
    /// Clear drops every sequence's positions.
    func Clear()
}

/// KV is full attention's memory: for each layer and sequence, a cache of
/// the keys and values of its positions. A sequence's caches are made when
/// it is first written, so a memory of many sequences costs only those
/// used; sequences beyond those it was made for are added as they are
/// written, up to MaxSequences.
public final class KV: Memory {
    public let Device: gpu.Device
    public let Layers: int
    public let KVHeads: int
    public let HeadDim: int
    public let Capacity: int
    public var Sequences: int { return _lengths.count }
    var _caches: [[nn.Cache]]
    var _lengths: [int]

    public init(on d: gpu.Device, layers: int, kvHeads: int, headDim: int, capacity: int, sequences: int = 1) {
        self.Device = d
        self.Layers = layers
        self.KVHeads = kvHeads
        self.HeadDim = headDim
        self.Capacity = capacity
        self._caches = []
        for _ in 0..<sequences {
            self._caches.append([])
        }
        self._lengths = [int](repeating: 0, count: sequences)
    }

    public func Length(_ s: int) -> int {
        return s >= 0 && s < Sequences ? _lengths[s] : 0
    }

    public func Truncate(_ s: int, from: int) {
        if s >= 0 && s < Sequences && from < _lengths[s] {
            _lengths[s] = max(0, from)
        }
    }

    public func Clear() {
        for s in 0..<Sequences {
            _lengths[s] = 0
        }
    }

    /// Layer is sequence s's cache for layer l, made if it is not yet.
    public func Layer(_ l: int, sequence s: int) throws -> nn.Cache {
        if s < 0 || s >= MaxSequences {
            throw MemoryError.unknownSequence(s, MaxSequences)
        }
        while s >= _caches.count {
            _caches.append([])
            _lengths.append(0)
        }
        if _caches[s].isEmpty {
            var layers: [nn.Cache] = []
            for _ in 0..<Layers {
                layers.append(try nn.Cache(on: Device, kvHeads: KVHeads, headDim: HeadDim, capacity: Capacity))
            }
            _caches[s] = layers
        }
        return _caches[s][l]
    }

    /// Wrote records that sequence s now holds positions up to end.
    public func Wrote(_ s: int, through end: int) throws {
        if s < 0 || s >= Sequences {
            throw MemoryError.unknownSequence(s, Sequences)
        }
        if end > Capacity {
            throw MemoryError.full(end, Capacity)
        }
        _lengths[s] = end
    }
}

/// Run is one sequence's part of a batch: Count rows from row Row, at
/// positions from Position.
public struct Run {
    public var Row: int
    public var Count: int
    public var Position: int
    public var Sequence: int

    public init(row: int, count: int, position: int, sequence: int) {
        self.Row = row
        self.Count = count
        self.Position = position
        self.Sequence = sequence
    }
}
