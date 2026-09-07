/**
 * Unit tests for the matrix core, anchored by an independent exact-arithmetic
 * oracle (/tmp/anchors-linalg.py, Python fractions). Every anchor value below
 * was produced by that oracle, not hand-computed.
 */

import { describe, expect, it } from 'vitest'
import {
  determinant,
  inverse,
  multiply,
  parseMatrix,
  rref,
  solve,
  trace,
  transpose,
  type Matrix,
} from '../src/matrix.ts'

const ROUND = 10
const MAX = 20

function assertCloseMatrix(actual: Matrix | undefined, expected: Matrix | undefined): void {
  expect(actual).toBeDefined()
  expect(expected).toBeDefined()
  expect(actual?.length).toBe(expected?.length)
  for (let i = 0; i < (actual?.length ?? 0); i++) {
    expect(actual?.[i]?.length).toBe(expected?.[i]?.length)
    for (let j = 0; j < (actual?.[i]?.length ?? 0); j++) {
      expect(actual?.[i]?.[j]).toBeCloseTo(expected?.[i]?.[j] as number, 8)
    }
  }
}

function assertCloseVector(actual: number[] | undefined, expected: number[] | undefined): void {
  expect(actual).toBeDefined()
  expect(expected).toBeDefined()
  expect(actual?.length).toBe(expected?.length)
  for (let i = 0; i < (actual?.length ?? 0); i++) {
    expect(actual?.[i]).toBeCloseTo(expected?.[i] as number, 8)
  }
}

function assertNoUndefined(node: unknown, path = 'root'): void {
  if (node === null) return
  if (Array.isArray(node)) {
    node.forEach((item, i) => assertNoUndefined(item, `${path}[${i}]`))
    return
  }
  if (typeof node === 'object') {
    for (const [key, value] of Object.entries(node as Record<string, unknown>)) {
      expect(value, `${path}.${key} must not be undefined`).not.toBeUndefined()
      assertNoUndefined(value, `${path}.${key}`)
    }
  }
}

describe('parseMatrix', () => {
  it('accepts a valid rectangular matrix', () => {
    const out = parseMatrix([[1, 2], [3, 4]], MAX, ROUND)
    expect('matrix' in out).toBe(true)
    if ('matrix' in out) expect(out.matrix).toEqual([[1, 2], [3, 4]])
  })

  it('rejects non-rectangular input', () => {
    const out = parseMatrix([[1, 2], [3]], MAX, ROUND)
    expect('error' in out).toBe(true)
  })

  it('rejects non-finite entries', () => {
    expect('error' in parseMatrix([[1, Number.NaN]], MAX, ROUND)).toBe(true)
    expect('error' in parseMatrix([[1, Number.POSITIVE_INFINITY]], MAX, ROUND)).toBe(true)
  })

  it('rejects empty and oversized input', () => {
    expect('error' in parseMatrix([], MAX, ROUND)).toBe(true)
    const wide = Array.from({ length: MAX + 1 }, () => 1)
    expect('error' in parseMatrix([wide], MAX, ROUND)).toBe(true)
  })
})

describe('multiply', () => {
  it('2x3 · 3x2 = 2x2 (oracle anchor)', () => {
    const out = multiply([[1, 2, 3], [4, 5, 6]], [[7, 8], [9, 10], [11, 12]])
    assertCloseMatrix(out, [[58, 64], [139, 154]])
  })

  it('1x1 product', () => {
    assertCloseMatrix(multiply([[3]], [[-7]]), [[-21]])
  })

  it('identity multiplication preserves the matrix', () => {
    const a = [[2, 0, 1], [1, 3, 0], [0, 1, 4]]
    assertCloseMatrix(multiply(a, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]), a)
  })

  it('returns null on incompatible dimensions', () => {
    expect(multiply([[1, 2]], [[1]] as unknown as Matrix)).toBeNull()
  })
})

describe('determinant', () => {
  it('2x2 oracle anchor: det([[1,2],[3,4]]) = -2', () => {
    expect(determinant([[1, 2], [3, 4]], ROUND)).toBeCloseTo(-2, 8)
  })

  it('3x3 oracle anchor = -1', () => {
    expect(determinant([[2, 1, 1], [1, 3, 2], [1, 0, 0]], ROUND)).toBeCloseTo(-1, 8)
  })

  it('4x4 requiring partial pivoting = 12', () => {
    const m = [[0, 1, 0, 3], [2, 0, 0, 1], [0, 0, 1, 0], [0, 2, 0, 0]]
    expect(determinant(m, ROUND)).toBeCloseTo(12, 8)
  })

  it('singular matrix gives 0', () => {
    expect(determinant([[1, 2], [2, 4]], ROUND)).toBe(0)
  })

  it('fractional entries: 1/30 ≈ 0.0166666667', () => {
    expect(determinant([[0.5, 1 / 3], [0.25, 0.2]], ROUND)).toBeCloseTo(0.0166666667, 8)
  })

  it('returns null for non-square input', () => {
    expect(determinant([[1, 2, 3], [4, 5, 6]], ROUND)).toBeNull()
  })
})

describe('inverse', () => {
  it('2x2 oracle anchor: inv([[4,7],[2,6]])', () => {
    assertCloseMatrix(inverse([[4, 7], [2, 6]], ROUND), [[0.6, -0.7], [-0.2, 0.4]])
  })

  it('3x3 oracle anchor', () => {
    const m = [[1, 2, 3], [0, 1, 4], [5, 6, 0]]
    assertCloseMatrix(inverse(m, ROUND), [[-24, 18, 5], [20, -15, -4], [-5, 4, 1]])
  })

  it('fractional 2x2 oracle anchor', () => {
    assertCloseMatrix(inverse([[0.5, 1 / 3], [0.25, 0.2]], ROUND), [[12, -20], [-15, 30]])
  })

  it('singular matrix returns null', () => {
    expect(inverse([[1, 2], [2, 4]], ROUND)).toBeNull()
  })

  it('non-square matrix returns null', () => {
    expect(inverse([[1, 2, 3], [4, 5, 6]], ROUND)).toBeNull()
  })

  it('inverse times original = identity (self-consistency)', () => {
    const m = [[4, 7], [2, 6]]
    const inv = inverse(m, ROUND)
    const prod = multiply(m, inv as Matrix)
    assertCloseMatrix(prod, [[1, 0], [0, 1]])
  })
})

describe('trace and transpose', () => {
  it('trace of 3x3 = 15', () => {
    expect(trace([[1, 2, 3], [4, 5, 6], [7, 8, 9]])).toBeCloseTo(15, 8)
  })

  it('trace of non-square is null', () => {
    expect(trace([[1, 2], [3, 4], [5, 6]])).toBeNull()
  })

  it('transpose of 2x3', () => {
    assertCloseMatrix(transpose([[1, 2, 3], [4, 5, 6]]), [[1, 4], [2, 5], [3, 6]])
  })
})

describe('rref', () => {
  it('rank-2 3x3 oracle anchor', () => {
    const m = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assertCloseMatrix(rref(m, ROUND), [[1, 0, -1], [0, 1, 2], [0, 0, 0]])
  })

  it('linearly dependent rows collapse to zero rows', () => {
    const m = [[1, 2, 3, 4], [2, 4, 6, 8]]
    assertCloseMatrix(rref(m, ROUND), [[1, 2, 3, 4], [0, 0, 0, 0]])
  })

  it('pivot swap required', () => {
    const m = [[0, 2, 4], [1, 2, 3]]
    assertCloseMatrix(rref(m, ROUND), [[1, 0, -1], [0, 1, 2]])
  })
})

describe('solve', () => {
  it('unique 2x2: 2x+3y=8, x−y=1 → (2.2, 1.2)', () => {
    const out = solve([[2, 3], [1, -1]], [8, 1], ROUND)
    if ('error' in out) throw new Error(`unexpected error: ${out.error}`)
    expect(out.kind).toBe('unique')
    assertCloseVector(out.solution, [2.2, 1.2])
  })

  it('unique 3x3: (2, 3, -1)', () => {
    const a = [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]]
    const out = solve(a, [8, -11, -3], ROUND)
    if ('error' in out) throw new Error(`unexpected error: ${out.error}`)
    expect(out.kind).toBe('unique')
    assertCloseVector(out.solution, [2, 3, -1])
  })

  it('infinite: 2 equations, 3 variables', () => {
    const a = [[1, 1, 1], [1, 2, 3]]
    const out = solve(a, [6, 14], ROUND)
    if ('error' in out) throw new Error(`unexpected error: ${out.error}`)
    expect(out.kind).toBe('infinite')
    assertCloseVector(out.particular, [-2, 8, 0])
    expect(out.nullspaceBasis?.length).toBe(1)
    assertCloseVector(out.nullspaceBasis?.[0], [1, -2, 1])
    expect(out.freeVariableCount).toBe(1)
    assertCloseMatrix(out.rref, [[1, 0, -1, -2], [0, 1, 2, 8]])
  })

  it('infinite: 1 equation, 3 variables → 2 basis vectors', () => {
    const out = solve([[1, 2, 3]], [6], ROUND)
    if ('error' in out) throw new Error(`unexpected error: ${out.error}`)
    expect(out.kind).toBe('infinite')
    assertCloseVector(out.particular, [6, 0, 0])
    expect(out.nullspaceBasis?.length).toBe(2)
    assertCloseVector(out.nullspaceBasis?.[0], [-2, 1, 0])
    assertCloseVector(out.nullspaceBasis?.[1], [-3, 0, 1])
  })

  it('none: inconsistent 2x2', () => {
    const out = solve([[1, 1], [2, 2]], [3, 8], ROUND)
    if ('error' in out) throw new Error(`unexpected error: ${out.error}`)
    expect(out.kind).toBe('none')
    expect(out.message).toContain('inconsistent')
  })

  it('none: inconsistent 3x3', () => {
    const a = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    const out = solve(a, [1, 2, 3], ROUND)
    if ('error' in out) throw new Error(`unexpected error: ${out.error}`)
    expect(out.kind).toBe('none')
  })

  it('unique from overdetermined consistent system (3x2)', () => {
    const a = [[1, 2], [3, 4], [5, 6]]
    const out = solve(a, [5, 11, 17], ROUND)
    if ('error' in out) throw new Error(`unexpected error: ${out.error}`)
    expect(out.kind).toBe('unique')
    assertCloseVector(out.solution, [1, 2])
  })

  it('b length mismatch is an error', () => {
    const out = solve([[1, 2]], [1, 2, 3], ROUND)
    expect('error' in out).toBe(true)
  })

  it('solution verifies Ax = b (self-consistency)', () => {
    const a = [[2, 3], [1, -1]]
    const b = [8, 1]
    const out = solve(a, b, ROUND)
    if ('error' in out || out.kind !== 'unique') throw new Error('expected unique')
    const prod = multiply(a, (out.solution as number[]).map((v) => [v]))
    assertCloseVector(prod?.map((row) => row[0] ?? 0), b)
  })

  it('no undefined keys anywhere in infinite outcome', () => {
    const out = solve([[1, 1, 1], [1, 2, 3]], [6, 14], ROUND)
    assertNoUndefined(out)
  })
})
