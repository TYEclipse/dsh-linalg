/**
 * Tests for the v0.4.0 `matrix_svd` tool: singular values, numerical rank,
 * condition number, orthonormal factors with measured residuals, and the
 * optional Moore–Penrose pseudo-inverse.
 *
 * Every anchor value below was produced by test/oracle/anchors.py (section E),
 * which derives singular values independently of the implementation: the
 * eigenvalues of AᵀA through a symmetric Jacobi iteration (a different matrix
 * and a different iteration than the plugin's one-sided Jacobi on A), the
 * closed form σ² = (‖A‖_F² ± sqrt(‖A‖_F⁴ − 4·det²))/2 for every 2x2 case, and
 * exact `fractions` arithmetic for the pseudo-inverse and every rank.
 *
 * ORACLE: test/oracle/anchors.py
 *
 * Run `python3 test/oracle/anchors.py --check` to regenerate and self-check.
 */

import { describe, expect, it } from 'vitest'
import { svdDecompose, type Matrix, type SvdOutcome } from '../src/matrix.ts'
import { resolveConfig } from '../src/index.ts'
import { buildLinalgTools } from '../src/tools.ts'

const ROUND = 10

type Exec = (args: unknown) => Promise<unknown>

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

/** Narrow the union returned by svdDecompose, failing the test on an error. */
function expectSvd(m: Matrix, pseudoInverse = false): SvdOutcome {
  const out = svdDecompose(m, ROUND, pseudoInverse)
  if ('error' in out) throw new Error(`unexpected error: ${out.error}`)
  return out
}

function expectMatrix(actual: Matrix | undefined, expected: Matrix): void {
  expect(actual).toBeDefined()
  expect(actual?.length).toBe(expected.length)
  for (let i = 0; i < expected.length; i++) {
    expect(actual?.[i]?.length).toBe(expected[i]?.length)
    for (let j = 0; j < (expected[i]?.length ?? 0); j++) {
      expect(actual?.[i]?.[j]).toBeCloseTo(expected[i]?.[j] as number, 8)
    }
  }
}

/** Columns must be orthonormal: UᵀU = I (checked from the reported factors, not hand-typed). */
function expectOrthonormal(matrix: Matrix): void {
  const count = matrix[0]?.length ?? 0
  for (let a = 0; a < count; a++) {
    for (let b = 0; b < count; b++) {
      let dot = 0
      for (let i = 0; i < matrix.length; i++) dot += (matrix[i]?.[a] ?? 0) * (matrix[i]?.[b] ?? 0)
      expect(dot).toBeCloseTo(a === b ? 1 : 0, 10)
    }
  }
}

/** U·Σ·Vᵀ must reproduce the input to reporting precision. */
function expectReconstruction(out: SvdOutcome, input: Matrix): void {
  for (let i = 0; i < input.length; i++) {
    for (let j = 0; j < (input[i]?.length ?? 0); j++) {
      let sum = 0
      for (let t = 0; t < out.singularValues.length; t++) {
        if (t >= (out.u[0]?.length ?? 0)) continue
        sum += (out.u[i]?.[t] ?? 0) * (out.singularValues[t] as number) * (out.v[j]?.[t] ?? 0)
      }
      expect(sum).toBeCloseTo(input[i]?.[j] as number, 8)
    }
  }
}

describe('svdDecompose (oracle section E)', () => {
  it('orders a diagonal matrix descending and reports the condition number', () => {
    // oracle: diag_3_4 -> s = [4.0, 3.0], rank 2/2, condition = 1.3333333333
    const out = expectSvd([[3, 0], [0, 4]])
    expect(out.singularValues).toEqual([4, 3])
    expect(out.rank).toBe(2)
    expect(out.singular).toBe(false)
    expect(out.conditionNumber).toBeCloseTo(1.3333333333, 9)
    expect(out.residual).toBe(0)
    expect(out.orthogonality).toBe(0)
    expect(out.sweeps).toBeGreaterThanOrEqual(1)
    // shape: 2 columns (rank 2), and the factors are orthonormal
    expect(out.u).toHaveLength(2)
    expect(out.u[0]).toHaveLength(2)
    expect(out.v[0]).toHaveLength(2)
    expectOrthonormal(out.u)
    expectOrthonormal(out.v)
    expectReconstruction(out, [[3, 0], [0, 4]])
  })

  it('collapses a rank-1 matrix to one direction and withholds the condition number', () => {
    // oracle: ones_2x2 -> s = [2.0, 0.0], rank 1/2, condition = None (rank-deficient)
    const out = expectSvd([[1, 1], [1, 1]])
    expect(out.singularValues).toEqual([2, 0])
    expect(out.rank).toBe(1)
    expect(out.singular).toBe(true)
    expect(out.conditionNumber).toBeUndefined()
    expect(out.note).toContain('rank 1 of 2')
    // one direction only, unit length, and it reproduces the matrix
    expect(out.u[0]).toHaveLength(1)
    expect(out.v[0]).toHaveLength(1)
    expectOrthonormal(out.u)
    expectOrthonormal(out.v)
    expectReconstruction(out, [[1, 1], [1, 1]])
    expect(out.residual).toBe(0)
  })

  it('reports condition number 1 for a rotation (all singular values equal)', () => {
    // oracle: rotation_90 -> s = [1.0, 1.0], condition = 1.0
    const out = expectSvd([[0, -1], [1, 0]])
    expect(out.singularValues).toEqual([1, 1])
    expect(out.conditionNumber).toBe(1)
    expect(out.residual).toBe(0)
    expect(out.orthogonality).toBe(0)
  })

  it('handles a tall matrix with distinct singular values', () => {
    // oracle: tall_3x2 -> s = [9.5255180916, 0.5143005807], condition = 18.5213053413
    const out = expectSvd([[1, 2], [3, 4], [5, 6]])
    expect(out.singularValues).toHaveLength(2)
    expect(out.singularValues[0]).toBeCloseTo(9.5255180916, 9)
    expect(out.singularValues[1]).toBeCloseTo(0.5143005807, 9)
    expect(out.rank).toBe(2)
    expect(out.conditionNumber).toBeCloseTo(18.5213053413, 8)
    expect(out.u).toHaveLength(3)
    expect(out.v).toHaveLength(2)
    expect(out.residual).toBe(0)
    expect(out.orthogonality).toBe(0)
  })

  it('sees rank 1 through an outer product even though every entry is non-zero', () => {
    // oracle: rank1_3x2 -> s = [8.3666002653, 0.0], rank 1/2
    const out = expectSvd([[1, 2], [2, 4], [3, 6]])
    expect(out.singularValues[0]).toBeCloseTo(8.3666002653, 9)
    expect(out.singularValues[1]).toBe(0)
    expect(out.rank).toBe(1)
    expect(out.singular).toBe(true)
    expect(out.conditionNumber).toBeUndefined()
  })

  it('flags a singular square matrix instead of dividing by a zero singular value', () => {
    // oracle: square_singular -> s = [5.0, 0.0], rank 1/2
    const out = expectSvd([[1, 2], [2, 4]])
    expect(out.singularValues).toEqual([5, 0])
    expect(out.rank).toBe(1)
    expect(out.singular).toBe(true)
    expect(out.conditionNumber).toBeUndefined()
  })

  it('measures the condition number of an ill-conditioned Hilbert matrix', () => {
    // oracle: hilbert_3 -> s = [1.4083189271, 0.1223270659, 0.0026873404], condition = 524.0567775872
    const out = expectSvd([[1, 0.5, 1 / 3], [0.5, 1 / 3, 0.25], [1 / 3, 0.25, 0.2]])
    expect(out.singularValues[0]).toBeCloseTo(1.4083189271, 9)
    expect(out.singularValues[1]).toBeCloseTo(0.1223270659, 9)
    expect(out.singularValues[2]).toBeCloseTo(0.0026873404, 9)
    expect(out.rank).toBe(3)
    expect(out.conditionNumber).toBeCloseTo(524.0567775872, 6)
    expect(out.residual).toBe(0)
    expect(out.orthogonality).toBe(0)
  })

  it('treats the zero matrix as rank 0 with no directions and no condition number', () => {
    // oracle: zero_2x2 -> s = [0.0, 0.0], rank 0/2, condition = None
    const out = expectSvd([[0, 0], [0, 0]])
    expect(out.singularValues).toEqual([0, 0])
    expect(out.rank).toBe(0)
    expect(out.singular).toBe(true)
    expect(out.conditionNumber).toBeUndefined()
    expect(out.u).toEqual([[], []])
    expect(out.v).toEqual([[], []])
    expect(out.residual).toBe(0)
    expect(out.note).toContain('zero matrix')
  })

  it('handles a single row (rank 1 of min(1,3) = full rank)', () => {
    // oracle: row_1x3 -> s = [3.7416573868, 0.0, 0.0], rank 1/1, condition = 1.0
    const out = expectSvd([[1, 2, 3]])
    expect(out.singularValues[0]).toBeCloseTo(3.7416573868, 9)
    expect(out.singularValues[1]).toBe(0)
    expect(out.singularValues[2]).toBe(0)
    expect(out.rank).toBe(1)
    expect(out.singular).toBe(false)
    expect(out.conditionNumber).toBe(1)
  })

  it('handles a wide matrix, padding the singular value list to the column count', () => {
    // oracle: wide_2x3 -> s = [9.5080320007, 0.7728696357, 0.0], rank 2/2, condition = 12.3022455041
    const out = expectSvd([[1, 2, 3], [4, 5, 6]])
    expect(out.singularValues).toHaveLength(3)
    expect(out.singularValues[0]).toBeCloseTo(9.5080320007, 9)
    expect(out.singularValues[1]).toBeCloseTo(0.7728696357, 9)
    expect(out.singularValues[2]).toBe(0)
    expect(out.rank).toBe(2)
    expect(out.conditionNumber).toBeCloseTo(12.3022455041, 8)
  })

  it('reports rank 2 for a 4x3 matrix whose rows are an arithmetic progression', () => {
    // oracle: rect_4x3 -> s = [25.462407436, 1.2906616758, 1.031e-07], exact rank 2/3
    // (σ₃ is exactly 0: the Gram route can only bound it below 1.0e-06, so the
    // exact rank from Fractions is the authority on the third value)
    const out = expectSvd([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    expect(out.singularValues).toHaveLength(3)
    expect(out.singularValues[0]).toBeCloseTo(25.462407436, 9)
    expect(out.singularValues[1]).toBeCloseTo(1.2906616758, 9)
    expect(out.singularValues[2]).toBe(0)
    expect(out.rank).toBe(2)
    expect(out.singular).toBe(true)
    expect(out.conditionNumber).toBeUndefined()
    expect(out.residual).toBe(0)
  })

  it('skips a zero column without dividing by zero', () => {
    // oracle: zero_column -> s = [1.0, 0.0], rank 1/2
    const out = expectSvd([[1, 0], [0, 0]])
    expect(out.singularValues).toEqual([1, 0])
    expect(out.rank).toBe(1)
    expect(out.conditionNumber).toBeUndefined()
    expectReconstruction(out, [[1, 0], [0, 0]])
  })

  it('rejects the empty matrix instead of inventing a result', () => {
    const out = svdDecompose([], ROUND)
    expect('error' in out && out.error).toContain('non-empty')
  })

  it('returns the Moore–Penrose pseudo-inverse with the identity checked', () => {
    // oracle: pinv(tall_3x2) as rows = [[-1.3333333333, -0.3333333333, 0.6666666667],
    //                                  [1.0833333333, 0.3333333333, -0.4166666667]]
    const out = expectSvd([[1, 2], [3, 4], [5, 6]], true)
    expect(out.pseudoInverse).toBeDefined()
    expectMatrix(out.pseudoInverse, [
      [-1.3333333333, -0.3333333333, 0.6666666667],
      [1.0833333333, 0.3333333333, -0.4166666667],
    ])
    expect(out.pseudoInverseResidual).toBe(0)
  })

  it('returns the pseudo-inverse of a wide matrix and of a full-rank square matrix', () => {
    // oracle: pinv(wide_2x3) as rows = [[-0.9444444444, 0.4444444444], [-0.1111111111, 0.1111111111], [0.7222222222, -0.2222222222]]
    const wide = expectSvd([[1, 2, 3], [4, 5, 6]], true)
    expectMatrix(wide.pseudoInverse, [
      [-0.9444444444, 0.4444444444],
      [-0.1111111111, 0.1111111111],
      [0.7222222222, -0.2222222222],
    ])
    expect(wide.pseudoInverseResidual).toBe(0)
    // oracle: pinv(diag_2_4) as rows = [[0.5, 0.0], [0.0, 0.25]]
    const square = expectSvd([[2, 0], [0, 4]], true)
    expectMatrix(square.pseudoInverse, [[0.5, 0], [0, 0.25]])
    expect(square.pseudoInverseResidual).toBe(0)
  })

  it('refuses non-rectangular input through the shared parser', async () => {
    const tools = buildLinalgTools(resolveConfig({}))
    const run = tools.matrix_svd.execute as unknown as Exec
    const out = (await run({ matrix: [[1, 2], [3]] })) as { valid: boolean; error?: string }
    expect(out.valid).toBe(false)
    expect(out.error).toContain('rectangular')
  })
})

describe('matrix_svd tool', () => {
  const tools = buildLinalgTools(resolveConfig({}))
  const run = tools.matrix_svd.execute as unknown as Exec

  it('returns the full report for a tall matrix', async () => {
    const out = (await run({ matrix: [[1, 2], [3, 4], [5, 6]] })) as Record<string, unknown>
    expect(out.valid).toBe(true)
    expect(out.rows).toBe(3)
    expect(out.columns).toBe(2)
    expect(out.rank).toBe(2)
    expect(out.singular).toBe(false)
    expect(out.conditionNumber).toBeCloseTo(18.5213053413, 8)
    expect(out.residual).toBe(0)
    expect(out.orthogonality).toBe(0)
    assertNoUndefined(out)
  })

  it('omits the condition number (and says why) for a singular matrix', async () => {
    const out = (await run({ matrix: [[1, 2], [2, 4]] })) as Record<string, unknown>
    expect(out.valid).toBe(true)
    expect(out.conditionNumber).toBeUndefined()
    expect(out.singular).toBe(true)
    expect(String(out.note)).toContain('no condition number is reported')
    assertNoUndefined(out)
  })

  it('exposes the pseudo-inverse only when asked', async () => {
    const without = (await run({ matrix: [[1, 2], [3, 4], [5, 6]] })) as Record<string, unknown>
    expect(without.pseudoInverse).toBeUndefined()
    const withPinv = (await run({ matrix: [[1, 2], [3, 4], [5, 6]], pseudoInverse: true })) as Record<string, unknown>
    expect(withPinv.pseudoInverse).toBeDefined()
    expect(withPinv.pseudoInverseResidual).toBe(0)
    assertNoUndefined(withPinv)
  })

  it('renders singular values, rank and residuals', async () => {
    const value = (await run({ matrix: [[1, 2], [3, 4], [5, 6]] })) as Record<string, unknown>
    const block = tools.matrix_svd.output.render({ matrix: [[1, 2], [3, 4], [5, 6]] }, value) as { type: string; text: string }[]
    expect(block).toHaveLength(1)
    expect(block[0]?.text).toContain('singular values (descending): 9.5255180916, 0.5143005807')
    expect(block[0]?.text).toContain('rank 2 of 2')
    expect(block[0]?.text).toContain('condition number 18.5213053413')
    expect(block[0]?.text).not.toContain('pseudo-inverse')
  })

  it('renders the pseudo-inverse block when it is present', async () => {
    const value = (await run({ matrix: [[1, 2], [3, 4], [5, 6]], pseudoInverse: true })) as Record<string, unknown>
    const block = tools.matrix_svd.output.render(
      { matrix: [[1, 2], [3, 4], [5, 6]], pseudoInverse: true },
      value,
    ) as { text: string }[]
    expect(block[0]?.text).toContain('pseudo-inverse A⁺ (2x3):')
    expect(block[0]?.text).toContain('max |A·A⁺·A − A| 0')
  })

  it('renders a rank-0 report without crashing on empty vectors', async () => {
    const value = (await run({ matrix: [[0, 0], [0, 0]] })) as Record<string, unknown>
    const block = tools.matrix_svd.output.render({ matrix: [[0, 0], [0, 0]] }, value) as { text: string }[]
    expect(block[0]?.text).toContain('(none: rank 0)')
    expect(block[0]?.text).toContain('condition number not reported')
  })
})
