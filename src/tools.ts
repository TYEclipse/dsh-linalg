/**
 * Tool definitions for dsh-linalg: seven pure-math tools exposed via defineTool.
 *   matrix_multiply — matrix product A·B with dimension checking
 *   matrix_compute  — transpose / determinant / inverse / trace / rref / rank / power
 *   matrix_decompose— LU / QR / Cholesky factors with measured residuals
 *   matrix_eigen    — symmetric eigen-decomposition with residuals
 *   least_squares   — minimizer of ||A·x − b|| for tall systems
 *   solve_linear    — solve Ax = b with unique / infinite / none classification
 *   vector_ops      — dot / cross / norm / projection / angle
 *
 * @module dsh-linalg/tools
 */

import { defineTool, type ToolDefinition } from '@deepseek-ai/dsh-tools'
import {
  choleskyDecompose,
  cleanMatrix,
  cleanValue,
  determinant,
  eigenSymmetric,
  inverse,
  leastSquares,
  luDecompose,
  multiply,
  parseMatrix,
  power,
  qrDecompose,
  rankOf,
  rref,
  solve,
  trace,
  transpose,
  type Matrix,
  type SolveKind,
} from './matrix.ts'
import { angleDeg, cross, dot, norm, parseVector, projection } from './vector.ts'
import type { ResolvedConfig } from './index.ts'

export interface ToolSet {
  matrix_multiply: ToolDefinition
  matrix_compute: ToolDefinition
  matrix_decompose: ToolDefinition
  matrix_eigen: ToolDefinition
  least_squares: ToolDefinition
  solve_linear: ToolDefinition
  vector_ops: ToolDefinition
}

type MatrixOp = 'transpose' | 'determinant' | 'inverse' | 'trace' | 'rref' | 'rank' | 'power'
type VectorOp = 'dot' | 'cross' | 'norm' | 'projection' | 'angle'
type DecomposeKind = 'lu' | 'qr' | 'cholesky'

const MATRIX_OPS: readonly MatrixOp[] = ['transpose', 'determinant', 'inverse', 'trace', 'rref', 'rank', 'power']
const VECTOR_OPS: readonly VectorOp[] = ['dot', 'cross', 'norm', 'projection', 'angle']
const DECOMPOSE_KINDS: readonly DecomposeKind[] = ['lu', 'qr', 'cholesky']

/** multiply result — every key present on success; only error on failure. */
interface MultiplyResult {
  valid: boolean
  error?: string
  rows?: number
  cols?: number
  product?: Matrix
}

interface ComputeResult {
  valid: boolean
  op: string
  error?: string
  /** scalar result for determinant / trace / rank */
  result?: number
  /** matrix result for transpose / inverse / rref / power */
  matrix?: Matrix
}

/** Result of a factorization: LU (p/l/u), QR (q/r) or Cholesky (l). */
interface DecomposeResult {
  valid: boolean
  kind: string
  error?: string
  /** LU only: row order of the permutation (row i of P·A is row p[i] of A). */
  p?: number[]
  /** LU and Cholesky: the lower triangular factor. */
  l?: Matrix
  /** LU only: the upper triangular factor. */
  u?: Matrix
  /** QR only: thin Q (columns orthonormal). */
  q?: Matrix
  /** QR only: upper triangular R. */
  r?: Matrix
  /** LU only: number of row swaps. */
  swaps?: number
  /** LU only: +1 / −1, the determinant sign from the swaps. */
  sign?: number
  /** LU only: true when a pivot column was numerically zero. */
  singular?: boolean
  /** LU and Cholesky: determinant implied by the factors. */
  determinant?: number
  /** max |P·A − L·U|, max |A − Q·R| or max |A − L·Lᵀ|. */
  residual?: number
  /** QR only: max |QᵀQ − I|. */
  orthogonality?: number
}

interface LeastSquaresResult {
  valid: boolean
  error?: string
  rows?: number
  columns?: number
  solution?: number[]
  residualNorm?: number
  normalEquationsResidual?: number
}

interface EigenResult {
  valid: boolean
  op: string
  error?: string
  /** Eigenvalues in descending order. */
  eigenvalues?: number[]
  /** Unit eigenvectors, one ROW per eigenvalue, same order. */
  eigenvectors?: Matrix
  /** Jacobi rotations applied. */
  rotations?: number
  /** Largest |A·v − λ·v| entry across all eigenpairs. */
  maxResidual?: number
  /** Sum of eigenvalues (equals the trace). */
  traceSum?: number
}

interface SolveResult {
  valid: boolean
  kind: SolveKind
  error?: string
  solution?: number[]
  particular?: number[]
  nullspaceBasis?: Matrix
  freeVariableCount?: number
  rref?: Matrix
  message?: string
}

interface VectorResult {
  valid: boolean
  op: string
  error?: string
  /** scalar result for dot / norm / angle */
  result?: number
  /** vector result for cross / projection */
  vector?: number[]
}

function renderMatrix(name: string, m: Matrix): string {
  return `${name} (${m.length}x${m[0]?.length ?? 0}):\n${m.map((row) => `  [${row.join(', ')}]`).join('\n')}`
}

function renderMultiply(value: unknown): string {
  const result = value as MultiplyResult
  if (!result.valid) return `matrix multiply failed: ${result.error}`
  return renderMatrix('product', result.product as Matrix)
}

function renderCompute(value: unknown): string {
  const result = value as ComputeResult
  if (!result.valid) return `${result.op} failed: ${result.error}`
  if (result.result !== undefined) return `${result.op} = ${result.result}`
  return renderMatrix(result.op, result.matrix as Matrix)
}

function renderEigen(value: unknown): string {
  const result = value as EigenResult
  if (!result.valid) return `matrix eigen failed: ${result.error}`
  const eigenvalues = `eigenvalues (descending): ${(result.eigenvalues as number[]).join(', ')}`
  const rows = (result.eigenvectors as Matrix)
    .map((vec, index) => `  lambda${index + 1} = ${(result.eigenvalues as number[])[index]}: [${vec.join(', ')}]`)
    .join('\n')
  return `${eigenvalues}\nsum = ${result.traceSum}, max residual = ${result.maxResidual}\neigenvectors (unit, one per row):\n${rows}`
}

function renderDecompose(value: unknown): string {
  const result = value as DecomposeResult
  if (!result.valid) return `matrix_decompose failed: ${result.error}`
  if (result.kind === 'lu') {
    const l = result.l as Matrix
    const u = result.u as Matrix
    return `LU: P (row order) [${(result.p as number[]).join(', ')}] — row i of P·A is row p[i] of A\n`
      + `row swaps ${result.swaps}, sign ${result.sign}, singular ${result.singular}, det from U ${result.determinant}, max |P·A − L·U| ${result.residual}\n`
      + `L (unit lower):\n${l.map((row) => `  [${row.join(', ')}]`).join('\n')}\n`
      + `U (upper):\n${u.map((row) => `  [${row.join(', ')}]`).join('\n')}`
  }
  if (result.kind === 'qr') {
    const q = result.q as Matrix
    const r = result.r as Matrix
    return `QR (thin): Q is ${q.length}x${q[0]?.length ?? 0}, R is ${r.length}x${r[0]?.length ?? 0}, R diagonal positive\n`
      + `max |A − Q·R| ${result.residual}, max |QᵀQ − I| ${result.orthogonality}\n`
      + `Q (orthonormal columns):\n${q.map((row) => `  [${row.join(', ')}]`).join('\n')}\n`
      + `R (upper):\n${r.map((row) => `  [${row.join(', ')}]`).join('\n')}`
  }
  const l = result.l as Matrix
  return `Cholesky: L (lower, positive diagonal), det from L ${result.determinant}, max |A − L·Lᵀ| ${result.residual}\n`
    + l.map((row) => `  [${row.join(', ')}]`).join('\n')
}

function renderLeastSquares(value: unknown): string {
  const result = value as LeastSquaresResult
  if (!result.valid) return `least_squares failed: ${result.error}`
  const solution = (result.solution as number[]).map((v, i) => `x${i + 1} = ${v}`).join(', ')
  return `least squares solution: ${solution}\n`
    + `A is ${result.rows}x${result.columns} (full column rank), min ||A·x − b|| = ${result.residualNorm}, `
    + `max |Aᵀ(A·x − b)| = ${result.normalEquationsResidual}`
}

function renderSolve(value: unknown): string {
  const result = value as SolveResult
  if (!result.valid) return `solve_linear failed: ${result.error}`
  if (result.kind === 'none') return `no solution: ${result.message}`
  if (result.kind === 'unique') {
    const vars = (result.solution as number[]).map((v, i) => `x${i + 1} = ${v}`).join(', ')
    return `unique solution: ${vars}`
  }
  const particular = `particular solution: [${(result.particular as number[]).join(', ')}]`
  const basis = (result.nullspaceBasis as Matrix).map((vec) => `[${vec.join(', ')}]`).join('; ')
  return `infinitely many solutions (${result.freeVariableCount} free variable(s))\n  ${particular}\n  nullspace basis: ${basis}\n  ${result.message}`
}

function renderVector(value: unknown): string {
  const result = value as VectorResult
  if (!result.valid) return `vector op failed: ${result.error}`
  if (result.result !== undefined) return `${result.op} = ${result.result}`
  return `${result.op} = [${(result.vector as number[]).join(', ')}]`
}

/** Build all four linalg tool definitions. */
export function buildLinalgTools(config: ResolvedConfig): ToolSet {
  const maxDim = config.maxDimension
  const roundPlaces = config.roundPlaces

  const matrix_multiply = defineTool({
    name: 'matrix_multiply',
    description: 'Multiply two matrices A·B with exact dimension checking. Handles any rectangular shapes that '
      + 'are compatible (columns of A must equal rows of B). Use this instead of computing matrix products by hand. '
      + 'Pure arithmetic, no external calls.',
    parameters: {
      a: { type: 'array', required: true, items: { type: 'array', items: { type: 'number' } }, description: `First matrix as rows of numbers, e.g. [[1,2],[3,4]]. Maximum dimension ${maxDim}x${maxDim}.` },
      b: { type: 'array', required: true, items: { type: 'array', items: { type: 'number' } }, description: 'Second matrix as rows of numbers.' },
    },
    output: {
      schema: {
        type: 'object',
        additionalProperties: false,
        properties: {
          valid: { type: 'boolean', required: true },
          error: { type: 'string' },
          rows: { type: 'number' },
          cols: { type: 'number' },
          product: { type: 'array', items: { type: 'array', items: { type: 'number' } } },
        },
      },
      render: (_args: { a: Matrix; b: Matrix }, value: unknown) => [{ type: 'text', text: renderMultiply(value) }],
    },
    async execute(args: { a: Matrix; b: Matrix }): Promise<MultiplyResult> {
      const pa = parseMatrix(args.a, maxDim, roundPlaces)
      if ('error' in pa) return { valid: false, error: `A: ${pa.error}` }
      const pb = parseMatrix(args.b, maxDim, roundPlaces)
      if ('error' in pb) return { valid: false, error: `B: ${pb.error}` }
      const product = multiply(pa.matrix, pb.matrix)
      if (product === null) {
        return { valid: false, error: `dimension mismatch: A is ${pa.matrix.length}x${pa.matrix[0]?.length ?? 0} but B is ${pb.matrix.length}x${pb.matrix[0]?.length ?? 0} (columns of A must equal rows of B)` }
      }
      const out: MultiplyResult = { valid: true, rows: product.length, cols: product[0]?.length ?? 0, product }
      return out
    },
  })

  const matrix_compute = defineTool({
    name: 'matrix_compute',
    description: 'Compute a single matrix operation: transpose, determinant, inverse, trace, reduced row '
      + 'echelon form (rref), rank, or an integer power (power). Determinant, inverse and power require square '
      + 'matrices; singular matrices (determinant 0) report an error instead of producing garbage. Pure '
      + 'arithmetic, no external calls.',
    parameters: {
      matrix: { type: 'array', required: true, items: { type: 'array', items: { type: 'number' } }, description: `Matrix as rows of numbers, e.g. [[1,2],[3,4]]. Maximum dimension ${maxDim}x${maxDim}.` },
      op: { type: 'string', required: true, enum: [...MATRIX_OPS], description: 'Operation to perform: transpose | determinant | inverse | trace | rref | rank | power.' },
      exponent: { type: 'number', description: 'Exponent for op="power": an integer from -64 to 64 (negative exponents invert first, 0 gives the identity). Ignored by every other operation.' },
    },
    output: {
      schema: {
        type: 'object',
        additionalProperties: false,
        properties: {
          valid: { type: 'boolean', required: true },
          op: { type: 'string', required: true },
          error: { type: 'string' },
          result: { type: 'number' },
          matrix: { type: 'array', items: { type: 'array', items: { type: 'number' } } },
        },
      },
      render: (_args: { matrix: Matrix; op: string }, value: unknown) => [{ type: 'text', text: renderCompute(value) }],
    },
    async execute(args: { matrix: Matrix; op: string; exponent?: number }): Promise<ComputeResult> {
      const op = args.op as MatrixOp
      const pm = parseMatrix(args.matrix, maxDim, roundPlaces)
      if ('error' in pm) return { valid: false, op, error: pm.error }
      const m = pm.matrix
      switch (op) {
        case 'transpose': {
          const out: ComputeResult = { valid: true, op, matrix: transpose(m) }
          return out
        }
        case 'determinant': {
          const d = determinant(m, roundPlaces)
          if (d === null) return { valid: false, op, error: `determinant requires a square matrix; got ${m.length}x${m[0]?.length ?? 0}` }
          const out: ComputeResult = { valid: true, op, result: d }
          return out
        }
        case 'inverse': {
          const inv = inverse(m, roundPlaces)
          if (inv === null) {
            const d = determinant(m, roundPlaces)
            if (d === null) return { valid: false, op, error: `inverse requires a square matrix; got ${m.length}x${m[0]?.length ?? 0}` }
            if (d === 0) return { valid: false, op, error: 'matrix is singular (determinant 0); it has no inverse' }
            return { valid: false, op, error: 'matrix is numerically singular (pivot below tolerance)' }
          }
          const out: ComputeResult = { valid: true, op, matrix: inv }
          return out
        }
        case 'trace': {
          const t = trace(m)
          if (t === null) return { valid: false, op, error: `trace requires a square matrix; got ${m.length}x${m[0]?.length ?? 0}` }
          const out: ComputeResult = { valid: true, op, result: t }
          return out
        }
        case 'rref': {
          const out: ComputeResult = { valid: true, op, matrix: rref(m, roundPlaces) }
          return out
        }
        case 'rank': {
          const out: ComputeResult = { valid: true, op, result: rankOf(m, roundPlaces) }
          return out
        }
        case 'power': {
          if (args.exponent === undefined) return { valid: false, op, error: 'power requires an integer exponent, e.g. exponent=3 for A³ or exponent=-1 for A⁻¹' }
          const raised = power(m, args.exponent, roundPlaces)
          if ('error' in raised) return { valid: false, op, error: raised.error }
          const out: ComputeResult = { valid: true, op, matrix: raised.matrix }
          return out
        }
        default:
          return { valid: false, op, error: `unknown operation: ${op}` }
      }
    },
  })

  const matrix_eigen = defineTool({
    name: 'matrix_eigen',
    description: 'Eigen-decomposition of a real symmetric matrix (cyclic Jacobi method): returns every '
      + 'eigenvalue in descending order plus a matching unit eigenvector per eigenvalue (signed so the leading '
      + 'non-zero component is positive), the rotation count and the measured residual max|A·v − λ·v|. '
      + 'Non-square or non-symmetric input, and non-convergence, are reported as errors rather than guessed. '
      + 'Use this instead of factoring characteristic polynomials by hand. Pure arithmetic, no external calls.',
    parameters: {
      matrix: { type: 'array', required: true, items: { type: 'array', items: { type: 'number' } }, description: `Symmetric matrix as rows of numbers, e.g. [[2,1],[1,2]]. Maximum dimension ${maxDim}x${maxDim}.` },
    },
    output: {
      schema: {
        type: 'object',
        additionalProperties: false,
        properties: {
          valid: { type: 'boolean', required: true },
          op: { type: 'string', required: true },
          error: { type: 'string' },
          eigenvalues: { type: 'array', items: { type: 'number' } },
          eigenvectors: { type: 'array', items: { type: 'array', items: { type: 'number' } } },
          rotations: { type: 'number' },
          maxResidual: { type: 'number' },
          traceSum: { type: 'number' },
        },
      },
      render: (_args: { matrix: Matrix }, value: unknown) => [{ type: 'text', text: renderEigen(value) }],
    },
    async execute(args: { matrix: Matrix }): Promise<EigenResult> {
      const op = 'symmetric'
      const pm = parseMatrix(args.matrix, maxDim, roundPlaces)
      if ('error' in pm) return { valid: false, op, error: pm.error }
      const outcome = eigenSymmetric(pm.matrix, roundPlaces)
      if ('error' in outcome) return { valid: false, op, error: outcome.error }
      const out: EigenResult = {
        valid: true,
        op,
        eigenvalues: outcome.eigenvalues,
        eigenvectors: outcome.eigenvectors,
        rotations: outcome.rotations,
        maxResidual: outcome.maxResidual,
        traceSum: outcome.traceSum,
      }
      return out
    },
  })

  const matrix_decompose = defineTool({
    name: 'matrix_decompose',
    description: 'Factor a matrix and get the factors plus measured residuals: LU with partial pivoting '
      + '(P·A = L·U, unit lower L, upper U, singular matrices flagged instead of silently wrong), thin QR with a '
      + 'positive diagonal (A = Q·R, Q has orthonormal columns; linearly dependent columns are an error, not a '
      + 'made-up completion), or Cholesky (A = L·Lᵀ, symmetric positive definite only — an asymmetric or '
      + 'indefinite matrix is reported with the offending entry / pivot). Each answer carries max |factor product '
      + '− A| so you can see how accurate it is. Pure arithmetic, no external calls.',
    parameters: {
      matrix: { type: 'array', required: true, items: { type: 'array', items: { type: 'number' } }, description: `Matrix as rows of numbers, e.g. [[4,12,-16],[12,37,-43],[-16,-43,98]]. Maximum dimension ${maxDim}x${maxDim}.` },
      kind: { type: 'string', required: true, enum: [...DECOMPOSE_KINDS], description: 'Factorization to compute: lu (square, P·A = L·U) | qr (rows ≥ columns, A = Q·R) | cholesky (symmetric positive definite, A = L·Lᵀ).' },
    },
    output: {
      schema: {
        type: 'object',
        additionalProperties: false,
        properties: {
          valid: { type: 'boolean', required: true },
          kind: { type: 'string', required: true },
          error: { type: 'string' },
          p: { type: 'array', items: { type: 'number' } },
          l: { type: 'array', items: { type: 'array', items: { type: 'number' } } },
          u: { type: 'array', items: { type: 'array', items: { type: 'number' } } },
          q: { type: 'array', items: { type: 'array', items: { type: 'number' } } },
          r: { type: 'array', items: { type: 'array', items: { type: 'number' } } },
          swaps: { type: 'number' },
          sign: { type: 'number' },
          singular: { type: 'boolean' },
          determinant: { type: 'number' },
          residual: { type: 'number' },
          orthogonality: { type: 'number' },
        },
      },
      render: (_args: { matrix: Matrix; kind: string }, value: unknown) => [{ type: 'text', text: renderDecompose(value) }],
    },
    async execute(args: { matrix: Matrix; kind: string }): Promise<DecomposeResult> {
      const kind = args.kind as DecomposeKind
      const pm = parseMatrix(args.matrix, maxDim, roundPlaces)
      if ('error' in pm) return { valid: false, kind, error: pm.error }
      const m = pm.matrix
      if (kind === 'lu') {
        const lu = luDecompose(m, roundPlaces)
        if ('error' in lu) return { valid: false, kind, error: lu.error }
        const out: DecomposeResult = {
          valid: true,
          kind,
          p: lu.p,
          l: lu.l,
          u: lu.u,
          swaps: lu.swaps,
          sign: lu.sign,
          singular: lu.singular,
          determinant: lu.determinant,
          residual: lu.residual,
        }
        return out
      }
      if (kind === 'qr') {
        const qr = qrDecompose(m, roundPlaces)
        if ('error' in qr) return { valid: false, kind, error: qr.error }
        const out: DecomposeResult = { valid: true, kind, q: qr.q, r: qr.r, residual: qr.residual, orthogonality: qr.orthogonality }
        return out
      }
      if (kind === 'cholesky') {
        const ch = choleskyDecompose(m, roundPlaces)
        if ('error' in ch) return { valid: false, kind, error: ch.error }
        const out: DecomposeResult = { valid: true, kind, l: ch.l, determinant: ch.determinant, residual: ch.residual }
        return out
      }
      return { valid: false, kind, error: `unknown factorization: ${kind} (expected lu | qr | cholesky)` }
    },
  })

  const least_squares = defineTool({
    name: 'least_squares',
    description: 'Solve an overdetermined linear system in the least-squares sense: for A (m rows ≥ n columns) '
      + 'and b, return the x that minimizes ||A·x − b||, the minimum residual norm itself, and '
      + 'max |Aᵀ(A·x − b)| (zero means x really is the minimizer). Rank-deficient or underdetermined input is an '
      + 'error — the answer would not be unique, so none is invented. Use this for curve-fitting and noisy '
      + 'measurement problems instead of solving the normal equations by hand. Pure arithmetic, no external calls.',
    parameters: {
      a: { type: 'array', required: true, items: { type: 'array', items: { type: 'number' } }, description: `Coefficient matrix A as rows of numbers, e.g. [[1,1],[1,2],[1,3]]. Maximum dimension ${maxDim}x${maxDim}.` },
      b: { type: 'array', required: true, items: { type: 'number' }, description: 'Constant vector b; one entry per row of A.' },
    },
    output: {
      schema: {
        type: 'object',
        additionalProperties: false,
        properties: {
          valid: { type: 'boolean', required: true },
          error: { type: 'string' },
          rows: { type: 'number' },
          columns: { type: 'number' },
          solution: { type: 'array', items: { type: 'number' } },
          residualNorm: { type: 'number' },
          normalEquationsResidual: { type: 'number' },
        },
      },
      render: (_args: { a: Matrix; b: number[] }, value: unknown) => [{ type: 'text', text: renderLeastSquares(value) }],
    },
    async execute(args: { a: Matrix; b: number[] }): Promise<LeastSquaresResult> {
      const pa = parseMatrix(args.a, maxDim, roundPlaces)
      if ('error' in pa) return { valid: false, error: `A: ${pa.error}` }
      const pb = parseVector(args.b, maxDim)
      if ('error' in pb) return { valid: false, error: `b: ${pb.error}` }
      const outcome = leastSquares(pa.matrix, pb.vector, roundPlaces)
      if ('error' in outcome) return { valid: false, error: outcome.error }
      const out: LeastSquaresResult = {
        valid: true,
        rows: outcome.rows,
        columns: outcome.columns,
        solution: outcome.solution,
        residualNorm: outcome.residualNorm,
        normalEquationsResidual: outcome.normalEquationsResidual,
      }
      return out
    },
  })

  const solve_linear = defineTool({
    name: 'solve_linear',
    description: 'Solve the linear system Ax = b. Classifies the result as a unique solution, infinitely many '
      + 'solutions (returns a particular solution plus a nullspace basis and the free-variable count), or no '
      + 'solution (inconsistent system). Handles singular and rectangular A. Pure arithmetic, no external calls.',
    parameters: {
      a: { type: 'array', required: true, items: { type: 'array', items: { type: 'number' } }, description: `Coefficient matrix A as rows of numbers. Maximum dimension ${maxDim}x${maxDim}.` },
      b: { type: 'array', required: true, items: { type: 'number' }, description: 'Constant vector b; must have one entry per row of A.' },
    },
    output: {
      schema: {
        type: 'object',
        additionalProperties: false,
        properties: {
          valid: { type: 'boolean', required: true },
          kind: { type: 'string', required: true, enum: ['unique', 'infinite', 'none'] },
          error: { type: 'string' },
          solution: { type: 'array', items: { type: 'number' } },
          particular: { type: 'array', items: { type: 'number' } },
          nullspaceBasis: { type: 'array', items: { type: 'array', items: { type: 'number' } } },
          freeVariableCount: { type: 'number' },
          rref: { type: 'array', items: { type: 'array', items: { type: 'number' } } },
          message: { type: 'string' },
        },
      },
      render: (_args: { a: Matrix; b: number[] }, value: unknown) => [{ type: 'text', text: renderSolve(value) }],
    },
    async execute(args: { a: Matrix; b: number[] }): Promise<SolveResult> {
      const pa = parseMatrix(args.a, maxDim, roundPlaces)
      if ('error' in pa) return { valid: false, kind: 'none', error: `A: ${pa.error}` }
      const pb = parseVector(args.b, maxDim)
      if ('error' in pb) return { valid: false, kind: 'none', error: `b: ${pb.error}` }
      const outcome = solve(pa.matrix, pb.vector, roundPlaces)
      if ('error' in outcome) return { valid: false, kind: 'none', error: outcome.error }
      const out: SolveResult = { valid: true, kind: outcome.kind }
      if (outcome.solution !== undefined) out.solution = outcome.solution
      if (outcome.particular !== undefined) out.particular = outcome.particular
      if (outcome.nullspaceBasis !== undefined) out.nullspaceBasis = outcome.nullspaceBasis
      if (outcome.freeVariableCount !== undefined) out.freeVariableCount = outcome.freeVariableCount
      if (outcome.rref !== undefined) out.rref = outcome.rref
      if (outcome.message !== undefined) out.message = outcome.message
      return out
    },
  })

  const vector_ops = defineTool({
    name: 'vector_ops',
    description: 'Compute a single vector operation: dot product (any equal length), cross product (3D only), '
      + 'Euclidean norm, projection of a onto b, or the angle between two vectors in degrees. '
      + 'Pure arithmetic, no external calls.',
    parameters: {
      op: { type: 'string', required: true, enum: [...VECTOR_OPS], description: 'Operation to perform: dot | cross | norm | projection | angle.' },
      a: { type: 'array', required: true, items: { type: 'number' }, description: 'First vector, e.g. [1, 2, 3].' },
      b: { type: 'array', items: { type: 'number' }, description: 'Second vector; required for dot, cross, projection and angle, ignored for norm.' },
    },
    output: {
      schema: {
        type: 'object',
        additionalProperties: false,
        properties: {
          valid: { type: 'boolean', required: true },
          op: { type: 'string', required: true },
          error: { type: 'string' },
          result: { type: 'number' },
          vector: { type: 'array', items: { type: 'number' } },
        },
      },
      render: (_args: { op: string; a: number[]; b?: number[] }, value: unknown) => [{ type: 'text', text: renderVector(value) }],
    },
    async execute(args: { op: string; a: number[]; b?: number[] }): Promise<VectorResult> {
      const op = args.op as VectorOp
      const pa = parseVector(args.a, maxDim)
      if ('error' in pa) return { valid: false, op, error: `a: ${pa.error}` }
      const a = pa.vector
      switch (op) {
        case 'norm': {
          const out: VectorResult = { valid: true, op, result: cleanValue(norm(a), roundPlaces) }
          return out
        }
        case 'dot':
        case 'cross':
        case 'projection':
        case 'angle': {
          if (args.b === undefined) return { valid: false, op, error: `${op} requires a second vector b` }
          const pb = parseVector(args.b, maxDim)
          if ('error' in pb) return { valid: false, op, error: `b: ${pb.error}` }
          const b = pb.vector
          if (op === 'dot') {
            const d = dot(a, b)
            if (d === null) return { valid: false, op, error: `dot product requires equal-length vectors; got ${a.length} and ${b.length}` }
            const out: VectorResult = { valid: true, op, result: cleanValue(d, roundPlaces) }
            return out
          }
          if (op === 'cross') {
            const c = cross(a, b)
            if (c === null) return { valid: false, op, error: `cross product requires 3D vectors; got lengths ${a.length} and ${b.length}` }
            const out: VectorResult = { valid: true, op, vector: cleanMatrix([c], roundPlaces)[0] }
            return out
          }
          if (op === 'projection') {
            const p = projection(a, b)
            if (p === null) return { valid: false, op, error: `projection requires equal-length vectors and a non-zero b; got lengths ${a.length} and ${b.length}` }
            const out: VectorResult = { valid: true, op, vector: cleanMatrix([p], roundPlaces)[0] }
            return out
          }
          const ang = angleDeg(a, b)
          if (ang === null) return { valid: false, op, error: `angle requires equal-length non-zero vectors; got lengths ${a.length} and ${b.length}` }
          const out: VectorResult = { valid: true, op, result: cleanValue(ang, roundPlaces) }
          return out
        }
        default:
          return { valid: false, op, error: `unknown operation: ${op}` }
      }
    },
  })

  return { matrix_multiply, matrix_compute, matrix_decompose, matrix_eigen, least_squares, solve_linear, vector_ops }
}
