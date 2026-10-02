/**
 * Unit tests for the matrix core, anchored by an independent exact-arithmetic
 * oracle (test/oracle/anchors.py — Python `fractions` Gaussian elimination and
 * closed-form spectra). Every anchor value below was produced by that oracle.
 *
 * ORACLE: test/oracle/anchors.py
 *
 * Run `python3 test/oracle/anchors.py --check` to regenerate and self-check.
 */

import { describe, expect, it } from 'vitest'
import {
  determinant,
  eigenSymmetric,
  inverse,
  multiply,
  parseMatrix,
  power,
  rankOf,
  rref,
  solve,
  trace,
  transpose,
  type Matrix,
} from '../src/matrix.ts'

const ROUND = 10
const MAX = 20

function assertCloseMatrix(actual: Matrix | null | undefined, expected: Matrix | null | undefined): void {
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

describe('rankOf', () => {
  it('rank-2 3x3 (oracle anchor)', () => {
    expect(rankOf([[1, 2, 3], [4, 5, 6], [7, 8, 9]], ROUND)).toBe(2)
  })

  it('dependent 2x2 has rank 1 (oracle anchor)', () => {
    expect(rankOf([[1, 2], [2, 4]], ROUND)).toBe(1)
  })

  it('zero matrix has rank 0 (oracle anchor)', () => {
    expect(rankOf([[0, 0], [0, 0]], ROUND)).toBe(0)
  })

  it('full-rank rectangular 2x3 has rank 2 (oracle anchor)', () => {
    expect(rankOf([[1, 2, 3], [4, 5, 6]], ROUND)).toBe(2)
  })

  it('identity 3x3 has rank 3 (oracle anchor)', () => {
    expect(rankOf([[1, 0, 0], [0, 1, 0], [0, 0, 1]], ROUND)).toBe(3)
  })

  it('rank-1 outer product 4x4 has rank 1 (oracle anchor)', () => {
    expect(rankOf([[1, 2, 3, 4], [2, 4, 6, 8], [3, 6, 9, 12], [4, 8, 12, 16]], ROUND)).toBe(1)
  })

  it('rank never exceeds the smaller dimension', () => {
    const wide = [[1, 2, 3, 4], [5, 6, 7, 8]]
    expect(rankOf(wide, ROUND)).toBeLessThanOrEqual(2)
  })
})

describe('power', () => {
  it('nilpotent shift cubed (oracle anchor)', () => {
    const out = power([[1, 1], [0, 1]], 3, ROUND)
    expect('matrix' in out).toBe(true)
    if ('matrix' in out) assertCloseMatrix(out.matrix, [[1, 3], [0, 1]])
  })

  it('exponent 0 gives the identity (oracle anchor)', () => {
    const out = power([[4, 7], [2, 6]], 0, ROUND)
    if (!('matrix' in out)) throw new Error('expected a matrix')
    assertCloseMatrix(out.matrix, [[1, 0], [0, 1]])
  })

  it('squared 2x2 (oracle anchor)', () => {
    const out = power([[4, 7], [2, 6]], 2, ROUND)
    if (!('matrix' in out)) throw new Error('expected a matrix')
    assertCloseMatrix(out.matrix, [[30, 70], [20, 50]])
  })

  it('negative exponent inverts first (oracle anchor)', () => {
    const out = power([[4, 7], [2, 6]], -1, ROUND)
    if (!('matrix' in out)) throw new Error('expected a matrix')
    assertCloseMatrix(out.matrix, [[0.6, -0.7], [-0.2, 0.4]])
  })

  it('Fibonacci matrix to the 20th power (oracle anchor)', () => {
    const out = power([[1, 1], [1, 0]], 20, ROUND)
    if (!('matrix' in out)) throw new Error('expected a matrix')
    assertCloseMatrix(out.matrix, [[10946, 6765], [6765, 4181]])
  })

  it('diagonal matrix power multiplies the diagonal (oracle anchor)', () => {
    const out = power([[2, 0], [0, 3]], 6, ROUND)
    if (!('matrix' in out)) throw new Error('expected a matrix')
    assertCloseMatrix(out.matrix, [[64, 0], [0, 729]])
  })

  it('A³ · A⁻³ = I (self-consistency)', () => {
    const m = [[4, 7], [2, 6]]
    const cube = power(m, 3, ROUND)
    const inverseCube = power(m, -3, ROUND)
    if (!('matrix' in cube) || !('matrix' in inverseCube)) throw new Error('expected matrices')
    assertCloseMatrix(multiply(cube.matrix, inverseCube.matrix), [[1, 0], [0, 1]])
  })

  it('rejects non-square matrices', () => {
    const out = power([[1, 2, 3]], 2, ROUND)
    expect('error' in out).toBe(true)
  })

  it('rejects a negative exponent on a singular matrix', () => {
    const out = power([[1, 2], [2, 4]], -1, ROUND)
    expect('error' in out).toBe(true)
    if ('error' in out) expect(out.error).toContain('singular')
  })

  it('rejects a non-integer exponent', () => {
    const out = power([[1, 2], [3, 4]], 0.5, ROUND)
    expect('error' in out).toBe(true)
    if ('error' in out) expect(out.error).toContain('integer')
  })

  it('rejects an exponent outside the supported range', () => {
    expect('error' in power([[1, 0], [0, 1]], 65, ROUND)).toBe(true)
    expect('error' in power([[1, 0], [0, 1]], -65, ROUND)).toBe(true)
  })
})

describe('eigenSymmetric', () => {
  /** V·Vᵀ must be the identity when the eigenvectors are orthonormal. */
  function orthonormalityError(vectors: Matrix): number {
    const n = vectors.length
    let worst = 0
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        let dot = 0
        for (let k = 0; k < n; k++) dot += (vectors[i]?.[k] ?? 0) * (vectors[j]?.[k] ?? 0)
        worst = Math.max(worst, Math.abs(dot - (i === j ? 1 : 0)))
      }
    }
    return worst
  }

  /** max |A·v − λ·v| recomputed from the reported numbers. */
  function residual(matrix: Matrix, eigenvalues: number[], vectors: Matrix): number {
    let worst = 0
    for (let i = 0; i < eigenvalues.length; i++) {
      const prod = multiply(matrix, (vectors[i] ?? []).map((v) => [v]))
      for (let r = 0; r < matrix.length; r++) {
        worst = Math.max(worst, Math.abs((prod?.[r]?.[0] ?? 0) - (eigenvalues[i] ?? 0) * (vectors[i]?.[r] ?? 0)))
      }
    }
    return worst
  }

  it('2x2 oracle anchor: [[2,1],[1,2]] → 3, 1', () => {
    const out = eigenSymmetric([[2, 1], [1, 2]], ROUND)
    if ('error' in out) throw new Error(out.error)
    expect(out.eigenvalues[0]).toBeCloseTo(3, 8)
    expect(out.eigenvalues[1]).toBeCloseTo(1, 8)
    assertCloseMatrix(out.eigenvectors, [[0.7071067812, 0.7071067812], [0.7071067812, -0.7071067812]])
    expect(out.traceSum).toBeCloseTo(4, 8)
    expect(out.rotations).toBe(1)
    expect(out.maxResidual).toBeCloseTo(0, 8)
  })

  it('diagonal oracle anchor: diag(5,−2,3) → 5, 3, −2 with permuted identity vectors', () => {
    const out = eigenSymmetric([[5, 0, 0], [0, -2, 0], [0, 0, 3]], ROUND)
    if ('error' in out) throw new Error(out.error)
    expect(out.eigenvalues).toEqual([5, 3, -2])
    assertCloseMatrix(out.eigenvectors, [[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    expect(out.traceSum).toBeCloseTo(6, 8)
    expect(out.rotations).toBe(0)
  })

  it('constant-diagonal oracle anchor: [[4,1,1],[1,4,1],[1,1,4]] → 6, 3, 3', () => {
    const out = eigenSymmetric([[4, 1, 1], [1, 4, 1], [1, 1, 4]], ROUND)
    if ('error' in out) throw new Error(out.error)
    expect(out.eigenvalues[0]).toBeCloseTo(6, 8)
    expect(out.eigenvalues[1]).toBeCloseTo(3, 8)
    expect(out.eigenvalues[2]).toBeCloseTo(3, 8)
    expect(out.traceSum).toBeCloseTo(12, 8)
    expect(orthonormalityError(out.eigenvectors)).toBeLessThan(1e-9)
    expect(residual([[4, 1, 1], [1, 4, 1], [1, 1, 4]], out.eigenvalues, out.eigenvectors)).toBeLessThan(1e-8)
  })

  it('irrational oracle anchor: [[1,2,0],[2,1,2],[0,2,1]] → 1+2√2, 1, 1−2√2', () => {
    const matrix = [[1, 2, 0], [2, 1, 2], [0, 2, 1]]
    const out = eigenSymmetric(matrix, ROUND)
    if ('error' in out) throw new Error(out.error)
    expect(out.eigenvalues[0]).toBeCloseTo(3.8284271247, 8)
    expect(out.eigenvalues[1]).toBeCloseTo(1, 8)
    expect(out.eigenvalues[2]).toBeCloseTo(-1.8284271247, 8)
    assertCloseMatrix(out.eigenvectors, [[0.5, 0.7071067812, 0.5], [0.7071067812, 0, -0.7071067812], [0.5, -0.7071067812, 0.5]])
    expect(out.traceSum).toBeCloseTo(3, 8)
    expect(residual(matrix, out.eigenvalues, out.eigenvectors)).toBeLessThan(1e-8)
  })

  it('tridiagonal oracle anchor: 2 − 2cos(kπ/5) with analytic eigenvectors', () => {
    const matrix = [[2, -1, 0, 0], [-1, 2, -1, 0], [0, -1, 2, -1], [0, 0, -1, 2]]
    const out = eigenSymmetric(matrix, ROUND)
    if ('error' in out) throw new Error(out.error)
    expect(out.eigenvalues[0]).toBeCloseTo(3.6180339887, 8)
    expect(out.eigenvalues[1]).toBeCloseTo(2.6180339887, 8)
    expect(out.eigenvalues[2]).toBeCloseTo(1.3819660113, 8)
    expect(out.eigenvalues[3]).toBeCloseTo(0.3819660113, 8)
    assertCloseMatrix(out.eigenvectors, [
      [0.3717480345, -0.601500955, 0.601500955, -0.3717480345],
      [0.601500955, -0.3717480345, -0.3717480345, 0.601500955],
      [0.601500955, 0.3717480345, -0.3717480345, -0.601500955],
      [0.3717480345, 0.601500955, 0.601500955, 0.3717480345],
    ])
    expect(out.traceSum).toBeCloseTo(8, 8)
    expect(orthonormalityError(out.eigenvectors)).toBeLessThan(1e-9)
    expect(residual(matrix, out.eigenvalues, out.eigenvectors)).toBeLessThan(1e-8)
  })

  it('integer spectrum oracle anchor: H·diag(8,4,0,−8)·Hᵀ → 8, 4, 0, −8', () => {
    const matrix = [[1, 3, 5, -1], [3, 1, -1, 5], [5, -1, 1, 3], [-1, 5, 3, 1]]
    const out = eigenSymmetric(matrix, ROUND)
    if ('error' in out) throw new Error(out.error)
    expect(out.eigenvalues.map((v) => Math.round(v))).toEqual([8, 4, 0, -8])
    expect(out.eigenvalues[0]).toBeCloseTo(8, 8)
    expect(out.eigenvalues[2]).toBeCloseTo(0, 8)
    expect(out.traceSum).toBeCloseTo(4, 8)
    expect(residual(matrix, out.eigenvalues, out.eigenvectors)).toBeLessThan(1e-8)
  })

  it('1x1 oracle anchor', () => {
    const out = eigenSymmetric([[7]], ROUND)
    if ('error' in out) throw new Error(out.error)
    expect(out.eigenvalues).toEqual([7])
    assertCloseMatrix(out.eigenvectors, [[1]])
    expect(out.rotations).toBe(0)
    expect(out.traceSum).toBe(7)
  })

  it('zero matrix oracle anchor: both eigenvalues are 0', () => {
    const out = eigenSymmetric([[0, 0], [0, 0]], ROUND)
    if ('error' in out) throw new Error(out.error)
    expect(out.eigenvalues).toEqual([0, 0])
    expect(out.maxResidual).toBe(0)
  })

  it('eigenvalues are sorted in descending order (5x5 tridiagonal)', () => {
    const matrix = [[2, -1, 0, 0, 0], [-1, 2, -1, 0, 0], [0, -1, 2, -1, 0], [0, 0, -1, 2, -1], [0, 0, 0, -1, 2]]
    const out = eigenSymmetric(matrix, ROUND)
    if ('error' in out) throw new Error(out.error)
    const expected = [3.7320508076, 3, 2, 1, 0.2679491924]
    for (let i = 0; i < expected.length; i++) expect(out.eigenvalues[i]).toBeCloseTo(expected[i] as number, 8)
    for (let i = 1; i < out.eigenvalues.length; i++) {
      expect((out.eigenvalues[i - 1] as number) >= (out.eigenvalues[i] as number)).toBe(true)
    }
    expect(orthonormalityError(out.eigenvectors)).toBeLessThan(1e-9)
  })

  it('leading non-zero component is positive (sign convention)', () => {
    const out = eigenSymmetric([[1, 2, 0], [2, 1, 2], [0, 2, 1]], ROUND)
    if ('error' in out) throw new Error(out.error)
    for (const vector of out.eigenvectors) {
      const lead = vector.find((v) => Math.abs(v) > 1e-8) ?? 0
      expect(lead).toBeGreaterThan(0)
    }
  })

  it('rejects non-square input', () => {
    const out = eigenSymmetric([[1, 2, 3]], ROUND)
    if (!('error' in out)) throw new Error('expected an error')
    expect(out.error).toContain('square')
  })

  it('rejects a non-symmetric matrix and names the offending entry', () => {
    const out = eigenSymmetric([[1, 2], [3, 4]], ROUND)
    if (!('error' in out)) throw new Error('expected an error')
    expect(out.error).toContain('not symmetric')
    expect(out.error).toContain('(0,1)')
  })

  it('accepts rounding-level asymmetry inside the tolerance', () => {
    const out = eigenSymmetric([[2, 1 + 1e-12], [1, 2]], ROUND)
    if ('error' in out) throw new Error(out.error)
    expect(out.eigenvalues[0]).toBeCloseTo(3, 8)
    expect(out.eigenvalues[1]).toBeCloseTo(1, 8)
  })
})
