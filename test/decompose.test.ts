/**
 * Unit tests for the v0.3.0 factorizations — LU with partial pivoting, thin QR,
 * Cholesky — plus the least-squares solver, and for the two new tools that
 * expose them.
 *
 * Every anchor value below was produced by test/oracle/anchors.py (section D),
 * which implements each factorization independently: exact `fractions`
 * Gaussian elimination for LU / Cholesky / least squares (via the normal
 * equations), and classical Gram–Schmidt for QR — a different algorithm from
 * the plugin's modified Gram–Schmidt.
 *
 * ORACLE: test/oracle/anchors.py
 *
 * Run `python3 test/oracle/anchors.py --check` to regenerate and self-check.
 */

import { describe, expect, it } from 'vitest'
import {
  choleskyDecompose,
  leastSquares,
  luDecompose,
  qrDecompose,
  type Matrix,
} from '../src/matrix.ts'
import { resolveConfig } from '../src/index.ts'
import { buildLinalgTools } from '../src/tools.ts'

const ROUND = 10
const MAX = 20

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

describe('luDecompose (oracle section D)', () => {
  it('factors [[2,1,1],[4,-6,0],[-2,7,2]] with one row swap', () => {
    const out = luDecompose([[2, 1, 1], [4, -6, 0], [-2, 7, 2]], ROUND)
    expect('error' in out).toBe(false)
    if ('error' in out) return
    expect(out.p).toEqual([1, 0, 2])
    expect(out.swaps).toBe(1)
    expect(out.sign).toBe(-1)
    expect(out.singular).toBe(false)
    expect(out.determinant).toBeCloseTo(-16, 8)
    expectMatrix(out.l, [[1, 0, 0], [0.5, 1, 0], [-0.5, 1, 1]])
    expectMatrix(out.u, [[4, -6, 0], [0, 4, 1], [0, 0, 1]])
    expect(out.residual).toBeCloseTo(0, 8)
  })

  it('factors the 4x4 pivot example and its determinant agrees with the U diagonal', () => {
    const out = luDecompose([[0, 1, 0, 3], [2, 0, 0, 1], [0, 0, 1, 0], [0, 2, 0, 0]], ROUND)
    if ('error' in out) throw new Error(out.error)
    expect(out.p).toEqual([1, 3, 2, 0])
    expect(out.swaps).toBe(2)
    expect(out.sign).toBe(1)
    expect(out.determinant).toBeCloseTo(12, 8)
    expectMatrix(out.l, [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0.5, 0, 1]])
    expectMatrix(out.u, [[2, 0, 0, 1], [0, 2, 0, 0], [0, 0, 1, 0], [0, 0, 0, 3]])
  })

  it('still factors a singular matrix but flags it and reports determinant 0', () => {
    const out = luDecompose([[1, 2], [2, 4]], ROUND)
    if ('error' in out) throw new Error(out.error)
    expect(out.p).toEqual([1, 0])
    expect(out.swaps).toBe(1)
    expect(out.singular).toBe(true)
    expect(out.determinant).toBe(0)
    expectMatrix(out.l, [[1, 0], [0.5, 1]])
    expectMatrix(out.u, [[2, 4], [0, 0]])
    expect(out.residual).toBeCloseTo(0, 8)
  })

  it('rejects a non-square matrix', () => {
    const out = luDecompose([[1, 2, 3]], ROUND)
    expect('error' in out && out.error).toContain('square matrix')
  })
})

describe('qrDecompose (oracle section D)', () => {
  it('factors [[3,0],[4,5]] into rational Q and R', () => {
    const out = qrDecompose([[3, 0], [4, 5]], ROUND)
    if ('error' in out) throw new Error(out.error)
    expectMatrix(out.q, [[0.6, -0.8], [0.8, 0.6]])
    expectMatrix(out.r, [[5, 4], [0, 3]])
    expect(out.residual).toBeCloseTo(0, 8)
    expect(out.orthogonality).toBeCloseTo(0, 8)
  })

  it('handles a tall 3x2 matrix (thin Q is 3x2)', () => {
    const out = qrDecompose([[3, 0], [4, 0], [0, 5]], ROUND)
    if ('error' in out) throw new Error(out.error)
    expect(out.q.length).toBe(3)
    expect(out.q[0]?.length).toBe(2)
    expectMatrix(out.q, [[0.6, 0], [0.8, 0], [0, 1]])
    expectMatrix(out.r, [[5, 0], [0, 5]])
  })

  it('matches the irrational case [[1,1],[1,0],[0,1]]', () => {
    const out = qrDecompose([[1, 1], [1, 0], [0, 1]], ROUND)
    if ('error' in out) throw new Error(out.error)
    expectMatrix(out.q, [[0.7071067812, 0.4082482905], [0.7071067812, -0.4082482905], [0, 0.8164965809]])
    expectMatrix(out.r, [[1.4142135624, 0.7071067812], [0, 1.2247448714]])
    expect(out.orthogonality).toBeCloseTo(0, 8)
  })

  it('rejects linearly dependent columns instead of inventing an orthonormal completion', () => {
    const out = qrDecompose([[1, 2], [2, 4]], ROUND)
    expect('error' in out && out.error).toContain('linearly dependent')
    expect('error' in out && out.error).toContain('column 2')
  })

  it('rejects rows < columns', () => {
    const out = qrDecompose([[1, 2, 3], [4, 5, 6]], ROUND)
    expect('error' in out && out.error).toContain('at least as many rows as columns')
  })
})

describe('choleskyDecompose (oracle section D)', () => {
  it('factors the textbook 3x3 SPD matrix exactly', () => {
    const out = choleskyDecompose([[4, 12, -16], [12, 37, -43], [-16, -43, 98]], ROUND)
    if ('error' in out) throw new Error(out.error)
    expectMatrix(out.l, [[2, 0, 0], [6, 1, 0], [-8, 5, 3]])
    expect(out.determinant).toBeCloseTo(36, 8)
    expect(out.residual).toBeCloseTo(0, 8)
  })

  it('rejects an indefinite matrix and names the pivot', () => {
    const out = choleskyDecompose([[1, 2], [2, 1]], ROUND)
    expect('error' in out && out.error).toContain('not positive definite')
    expect('error' in out && out.error).toContain('-3')
  })

  it('rejects an asymmetric matrix and names the entry', () => {
    const out = choleskyDecompose([[1, 2], [3, 4]], ROUND)
    expect('error' in out && out.error).toContain('symmetric')
    expect('error' in out && out.error).toContain('(0,1)')
  })

  it('rejects a non-square matrix', () => {
    const out = choleskyDecompose([[1, 2, 3]], ROUND)
    expect('error' in out && out.error).toContain('square matrix')
  })
})

describe('leastSquares (oracle section D)', () => {
  it('fits a line through [[1,1],[1,2],[1,3]] → b = [1,2,2]', () => {
    const out = leastSquares([[1, 1], [1, 2], [1, 3]], [1, 2, 2], ROUND)
    if ('error' in out) throw new Error(out.error)
    expect(out.solution[0]).toBeCloseTo(0.6666666667, 8)
    expect(out.solution[1]).toBeCloseTo(0.5, 8)
    expect(out.residualNorm).toBeCloseTo(0.4082482905, 8)
    expect(out.normalEquationsResidual).toBeCloseTo(0, 8)
    expect(out.rows).toBe(3)
    expect(out.columns).toBe(2)
  })

  it('reproduces an exact system with zero residual', () => {
    const out = leastSquares([[1, 0], [0, 1], [1, 1]], [2, 3, 5], ROUND)
    if ('error' in out) throw new Error(out.error)
    expect(out.solution).toEqual([2, 3])
    expect(out.residualNorm).toBe(0)
    expect(out.normalEquationsResidual).toBe(0)
  })

  it('refuses a rank-deficient system (the minimizer is not unique)', () => {
    const out = leastSquares([[1, 2], [2, 4], [3, 6]], [1, 2, 3], ROUND)
    expect('error' in out && out.error).toContain('rank 1')
    expect('error' in out && out.error).toContain('not unique')
  })

  it('refuses an underdetermined system', () => {
    const out = leastSquares([[1, 2, 3]], [6], ROUND)
    expect('error' in out && out.error).toContain('at least as many rows as columns')
  })

  it('rejects a b of the wrong length', () => {
    const out = leastSquares([[1, 2], [3, 4], [5, 6]], [1, 2], ROUND)
    expect('error' in out && out.error).toContain('b has 2 entries; expected 3')
  })
})

describe('matrix_decompose tool', () => {
  const tools = buildLinalgTools(resolveConfig({}))
  const run = tools.matrix_decompose.execute as unknown as Exec

  it('is registered under its canonical name with a schema and executable', () => {
    expect(tools.matrix_decompose.name).toBe('matrix_decompose')
    expect(typeof tools.matrix_decompose.execute).toBe('function')
    expect(tools.matrix_decompose.output.schema).toBeDefined()
  })

  it('returns LU factors for kind=lu', async () => {
    const out = (await run({ matrix: [[2, 1, 1], [4, -6, 0], [-2, 7, 2]], kind: 'lu' })) as {
      valid: boolean
      kind: string
      p: number[]
      l: Matrix
      u: Matrix
      swaps: number
      singular: boolean
      determinant: number
    }
    expect(out.valid).toBe(true)
    expect(out.kind).toBe('lu')
    expect(out.p).toEqual([1, 0, 2])
    expect(out.swaps).toBe(1)
    expect(out.singular).toBe(false)
    expect(out.determinant).toBeCloseTo(-16, 8)
    expectMatrix(out.l, [[1, 0, 0], [0.5, 1, 0], [-0.5, 1, 1]])
    expectMatrix(out.u, [[4, -6, 0], [0, 4, 1], [0, 0, 1]])
    assertNoUndefined(out)
  })

  it('returns QR factors for kind=qr', async () => {
    const out = (await run({ matrix: [[3, 0], [4, 5]], kind: 'qr' })) as {
      valid: boolean
      q: Matrix
      r: Matrix
      residual: number
      orthogonality: number
    }
    expect(out.valid).toBe(true)
    expectMatrix(out.q, [[0.6, -0.8], [0.8, 0.6]])
    expectMatrix(out.r, [[5, 4], [0, 3]])
    expect(out.residual).toBeCloseTo(0, 8)
    expect(out.orthogonality).toBeCloseTo(0, 8)
    assertNoUndefined(out)
  })

  it('returns the Cholesky factor for kind=cholesky', async () => {
    const out = (await run({ matrix: [[4, 12, -16], [12, 37, -43], [-16, -43, 98]], kind: 'cholesky' })) as {
      valid: boolean
      l: Matrix
      determinant: number
    }
    expect(out.valid).toBe(true)
    expectMatrix(out.l, [[2, 0, 0], [6, 1, 0], [-8, 5, 3]])
    expect(out.determinant).toBeCloseTo(36, 8)
    assertNoUndefined(out)
  })

  it('reports a rejected factorization without throwing', async () => {
    const out = (await run({ matrix: [[1, 2], [2, 1]], kind: 'cholesky' })) as { valid: boolean; error: string }
    expect(out.valid).toBe(false)
    expect(out.error).toContain('not positive definite')
    assertNoUndefined(out)
  })

  it('reports malformed matrices and rejects unknown kinds at the schema layer', async () => {
    const ragged = (await run({ matrix: [[1, 2], [3]], kind: 'lu' })) as { valid: boolean; error: string }
    expect(ragged.valid).toBe(false)
    expect(ragged.error).toContain('rectangular')
    assertNoUndefined(ragged)
    // The schema enum is the first line of defence: an unknown kind never reaches execute.
    await expect(run({ matrix: [[1]], kind: 'svd' })).rejects.toThrow(/must be one of/)
  })

  it('renders a human-readable summary for each kind', () => {
    const render = tools.matrix_decompose.output.render as unknown as (args: unknown, value: unknown) => Array<{ text: string }>
    const lu = render({ matrix: [[2, 1, 1], [4, -6, 0], [-2, 7, 2]], kind: 'lu' }, {
      valid: true, kind: 'lu', p: [1, 0, 2], l: [[1, 0, 0], [0.5, 1, 0], [-0.5, 1, 1]],
      u: [[4, -6, 0], [0, 4, 1], [0, 0, 1]], swaps: 1, sign: -1, singular: false, determinant: -16, residual: 0,
    })[0]?.text
    expect(lu).toContain('LU')
    expect(lu).toContain('[1, 0, 2]')
    const qr = render({ matrix: [[3, 0], [4, 5]], kind: 'qr' }, {
      valid: true, kind: 'qr', q: [[0.6, -0.8], [0.8, 0.6]], r: [[5, 4], [0, 3]], residual: 0, orthogonality: 0,
    })[0]?.text
    expect(qr).toContain('QᵀQ')
    const ch = render({ matrix: [[4]], kind: 'cholesky' }, { valid: true, kind: 'cholesky', l: [[2]], determinant: 4, residual: 0 })[0]?.text
    expect(ch).toContain('Cholesky')
    const bad = render({ matrix: [[1]], kind: 'qr' }, { valid: false, kind: 'qr', error: 'nope' })[0]?.text
    expect(bad).toContain('nope')
  })
})

describe('least_squares tool', () => {
  const tools = buildLinalgTools(resolveConfig({}))
  const run = tools.least_squares.execute as unknown as Exec

  it('solves the overdetermined system and reports the residual norm', async () => {
    const out = (await run({ a: [[1, 1], [1, 2], [1, 3]], b: [1, 2, 2] })) as {
      valid: boolean
      rows: number
      columns: number
      solution: number[]
      residualNorm: number
      normalEquationsResidual: number
    }
    expect(out.valid).toBe(true)
    expect(out.rows).toBe(3)
    expect(out.columns).toBe(2)
    expect(out.solution[0]).toBeCloseTo(0.6666666667, 8)
    expect(out.solution[1]).toBeCloseTo(0.5, 8)
    expect(out.residualNorm).toBeCloseTo(0.4082482905, 8)
    expect(out.normalEquationsResidual).toBeCloseTo(0, 8)
    assertNoUndefined(out)
  })

  it('rejects rank-deficient and malformed input', async () => {
    const deficient = (await run({ a: [[1, 2], [2, 4], [3, 6]], b: [1, 2, 3] })) as { valid: boolean; error: string }
    expect(deficient.valid).toBe(false)
    expect(deficient.error).toContain('rank 1')
    const ragged = (await run({ a: [[1, 2], [3]], b: [1, 2] })) as { valid: boolean; error: string }
    expect(ragged.valid).toBe(false)
    expect(ragged.error).toContain('A:')
    expect(ragged.error).toContain('rectangular')
    assertNoUndefined(ragged)
    // b is a required parameter, so the schema rejects a missing one before execute runs.
    await expect(run({ a: [[1, 1]] })).rejects.toThrow(/missing required property "b"/)
  })

  it('renders the solution and the two residuals', () => {
    const render = tools.least_squares.output.render as unknown as (args: unknown, value: unknown) => Array<{ text: string }>
    const text = render({ a: [[1, 1], [1, 2], [1, 3]], b: [1, 2, 2] }, {
      valid: true, rows: 3, columns: 2, solution: [0.6666666667, 0.5], residualNorm: 0.4082482905, normalEquationsResidual: 0,
    })[0]?.text
    expect(text).toContain('x1 = 0.6666666667')
    expect(text).toContain('0.4082482905')
    const bad = render({ a: [[1]], b: [1] }, { valid: false, error: 'nope' })[0]?.text
    expect(bad).toContain('nope')
  })
})
