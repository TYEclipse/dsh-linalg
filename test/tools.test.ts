/**
 * Tests for tool definition assembly, config resolution, execute paths,
 * schema-level enum rejection, and lossless-JSON discipline.
 */

import { describe, expect, it } from 'vitest'
import { resolveConfig } from '../src/index.ts'
import { buildLinalgTools } from '../src/tools.ts'

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

describe('resolveConfig', () => {
  it('applies every default', () => {
    expect(resolveConfig({})).toEqual({ maxDimension: 20, roundPlaces: 10 })
  })

  it('honours overrides', () => {
    const resolved = resolveConfig({ maxDimension: 5, roundPlaces: 3 })
    expect(resolved.maxDimension).toBe(5)
    expect(resolved.roundPlaces).toBe(3)
  })
})

describe('buildLinalgTools', () => {
  const tools = buildLinalgTools(resolveConfig({}))

  it('exposes all four tools under their canonical names', () => {
    expect(Object.keys(tools).sort()).toEqual(['matrix_compute', 'matrix_multiply', 'solve_linear', 'vector_ops'].sort())
  })

  it('gives every tool a name, description, schema and executable', () => {
    for (const [key, definition] of Object.entries(tools)) {
      expect(definition.name).toBe(key)
      expect(definition.description.length).toBeGreaterThan(20)
      expect(definition.parameters).toBeDefined()
      expect(definition.output.schema).toBeDefined()
      expect(typeof definition.execute).toBe('function')
    }
  })
})

describe('matrix_multiply execute', () => {
  const tools = buildLinalgTools(resolveConfig({}))
  const run = tools.matrix_multiply.execute as unknown as Exec

  it('multiplies a valid pair (oracle anchor)', async () => {
    const out = (await run({ a: [[1, 2, 3], [4, 5, 6]], b: [[7, 8], [9, 10], [11, 12]] })) as {
      valid: boolean
      rows: number
      cols: number
      product: number[][]
    }
    expect(out.valid).toBe(true)
    expect(out.rows).toBe(2)
    expect(out.cols).toBe(2)
    expect(out.product).toEqual([[58, 64], [139, 154]])
    assertNoUndefined(out)
  })

  it('reports dimension mismatch without throwing', async () => {
    const out = (await run({ a: [[1, 2]], b: [[1, 2]] })) as { valid: boolean; error: string }
    expect(out.valid).toBe(false)
    expect(out.error).toContain('dimension mismatch')
    assertNoUndefined(out)
  })

  it('reports malformed matrix', async () => {
    const out = (await run({ a: [[1, 2], [3]], b: [[1], [2]] })) as { valid: boolean; error: string }
    expect(out.valid).toBe(false)
    expect(out.error).toContain('rectangular')
  })
})

describe('matrix_compute execute', () => {
  const tools = buildLinalgTools(resolveConfig({}))
  const run = tools.matrix_compute.execute as unknown as Exec

  it('determinant anchor', async () => {
    const out = (await run({ matrix: [[1, 2], [3, 4]], op: 'determinant' })) as { valid: boolean; result: number }
    expect(out.valid).toBe(true)
    expect(out.result).toBeCloseTo(-2, 8)
    assertNoUndefined(out)
  })

  it('inverse anchor', async () => {
    const out = (await run({ matrix: [[4, 7], [2, 6]], op: 'inverse' })) as { valid: boolean; matrix: number[][] }
    expect(out.valid).toBe(true)
    expect(out.matrix).toEqual([[0.6, -0.7], [-0.2, 0.4]])
  })

  it('transpose anchor', async () => {
    const out = (await run({ matrix: [[1, 2, 3], [4, 5, 6]], op: 'transpose' })) as { valid: boolean; matrix: number[][] }
    expect(out.matrix).toEqual([[1, 4], [2, 5], [3, 6]])
  })

  it('trace anchor', async () => {
    const out = (await run({ matrix: [[1, 2, 3], [4, 5, 6], [7, 8, 9]], op: 'trace' })) as { valid: boolean; result: number }
    expect(out.result).toBeCloseTo(15, 8)
  })

  it('rref anchor', async () => {
    const out = (await run({ matrix: [[1, 2, 3], [4, 5, 6], [7, 8, 9]], op: 'rref' })) as { valid: boolean; matrix: number[][] }
    expect(out.matrix).toEqual([[1, 0, -1], [0, 1, 2], [0, 0, 0]])
  })

  it('singular inverse is a clean failure', async () => {
    const out = (await run({ matrix: [[1, 2], [2, 4]], op: 'inverse' })) as { valid: boolean; error: string }
    expect(out.valid).toBe(false)
    expect(out.error).toContain('singular')
    assertNoUndefined(out)
  })

  it('non-square determinant is a clean failure', async () => {
    const out = (await run({ matrix: [[1, 2], [3, 4], [5, 6]], op: 'determinant' })) as { valid: boolean; error: string }
    expect(out.valid).toBe(false)
    expect(out.error).toContain('square')
  })

  it('invalid enum value is rejected at the schema layer', async () => {
    await expect(run({ matrix: [[1]], op: 'eigenvalues' })).rejects.toThrow()
  })
})

describe('solve_linear execute', () => {
  const tools = buildLinalgTools(resolveConfig({}))
  const run = tools.solve_linear.execute as unknown as Exec

  it('unique solution anchor', async () => {
    const out = (await run({ a: [[2, 3], [1, -1]], b: [8, 1] })) as { valid: boolean; kind: string; solution: number[] }
    expect(out.valid).toBe(true)
    expect(out.kind).toBe('unique')
    expect(out.solution[0]).toBeCloseTo(2.2, 8)
    expect(out.solution[1]).toBeCloseTo(1.2, 8)
    assertNoUndefined(out)
  })

  it('infinite solutions with basis', async () => {
    const out = (await run({ a: [[1, 1, 1], [1, 2, 3]], b: [6, 14] })) as {
      valid: boolean
      kind: string
      particular: number[]
      nullspaceBasis: number[][]
      freeVariableCount: number
    }
    expect(out.kind).toBe('infinite')
    expect(out.particular).toEqual([-2, 8, 0])
    expect(out.nullspaceBasis).toEqual([[1, -2, 1]])
    expect(out.freeVariableCount).toBe(1)
    assertNoUndefined(out)
  })

  it('no-solution classification', async () => {
    const out = (await run({ a: [[1, 1], [2, 2]], b: [3, 8] })) as { valid: boolean; kind: string }
    expect(out.kind).toBe('none')
  })

  it('b length mismatch is a clean failure', async () => {
    const out = (await run({ a: [[1, 2]], b: [1, 2] })) as { valid: boolean; error: string }
    expect(out.valid).toBe(false)
    expect(out.error).toContain('b has')
  })
})

describe('vector_ops execute', () => {
  const tools = buildLinalgTools(resolveConfig({}))
  const run = tools.vector_ops.execute as unknown as Exec

  it('dot anchor', async () => {
    const out = (await run({ op: 'dot', a: [1, 2, 3], b: [4, 5, 6] })) as { valid: boolean; result: number }
    expect(out.valid).toBe(true)
    expect(out.result).toBeCloseTo(32, 8)
    assertNoUndefined(out)
  })

  it('cross anchor', async () => {
    const out = (await run({ op: 'cross', a: [1, 2, 3], b: [4, 5, 6] })) as { valid: boolean; vector: number[] }
    expect(out.vector).toEqual([-3, 6, -3])
  })

  it('norm without b', async () => {
    const out = (await run({ op: 'norm', a: [3, 4] })) as { valid: boolean; result: number }
    expect(out.result).toBeCloseTo(5, 8)
  })

  it('angle anchor: 90°', async () => {
    const out = (await run({ op: 'angle', a: [1, 0], b: [0, 1] })) as { valid: boolean; result: number }
    expect(out.result).toBeCloseTo(90, 8)
  })

  it('projection anchor', async () => {
    const out = (await run({ op: 'projection', a: [3, 4], b: [1, 0] })) as { valid: boolean; vector: number[] }
    expect(out.vector).toEqual([3, 0])
  })

  it('cross with 2D vectors is a clean failure', async () => {
    const out = (await run({ op: 'cross', a: [1, 2], b: [3, 4] })) as { valid: boolean; error: string }
    expect(out.valid).toBe(false)
    expect(out.error).toContain('3D')
  })

  it('missing b for dot is a clean failure', async () => {
    const out = (await run({ op: 'dot', a: [1, 2] })) as { valid: boolean; error: string }
    expect(out.valid).toBe(false)
    expect(out.error).toContain('requires a second vector')
  })

  it('invalid enum value is rejected at the schema layer', async () => {
    await expect(run({ op: 'rotate', a: [1, 2] })).rejects.toThrow()
  })
})

describe('renders', () => {
  const tools = buildLinalgTools(resolveConfig({}))

  it('renders a product as rows', () => {
    const block = tools.matrix_multiply.output.render(
      { a: [[1]], b: [[2]] },
      { valid: true, rows: 2, cols: 2, product: [[58, 64], [139, 154]] },
    )
    expect(block[0]?.text).toContain('product (2x2)')
    expect(block[0]?.text).toContain('[58, 64]')
  })

  it('renders a scalar determinant', () => {
    const block = tools.matrix_compute.output.render(
      { matrix: [[1]], op: 'determinant' },
      { valid: true, op: 'determinant', result: -2 },
    )
    expect(block[0]?.text).toBe('determinant = -2')
  })

  it('renders a unique solution', () => {
    const block = tools.solve_linear.output.render(
      { a: [[1]], b: [1] },
      { valid: true, kind: 'unique', solution: [2.2, 1.2] },
    )
    expect(block[0]?.text).toContain('unique solution: x1 = 2.2, x2 = 1.2')
  })

  it('renders an infinite solution with basis', () => {
    const block = tools.solve_linear.output.render(
      { a: [[1]], b: [1] },
      {
        valid: true,
        kind: 'infinite',
        particular: [-2, 8, 0],
        nullspaceBasis: [[1, -2, 1]],
        freeVariableCount: 1,
        message: 'm',
      },
    )
    expect(block[0]?.text).toContain('infinitely many solutions (1 free variable(s))')
    expect(block[0]?.text).toContain('[1, -2, 1]')
  })

  it('renders a cross product vector', () => {
    const block = tools.vector_ops.output.render(
      { op: 'cross', a: [1], b: [2] },
      { valid: true, op: 'cross', vector: [-3, 6, -3] },
    )
    expect(block[0]?.text).toBe('cross = [-3, 6, -3]')
  })
})
