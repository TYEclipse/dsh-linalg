/**
 * dsh-linalg matrix core: parsing, validation, and exact-ish numeric linear
 * algebra on plain JSON matrices (number[][]).
 *
 * All arithmetic uses Gaussian elimination with partial pivoting. Every
 * output number is rounded to `roundPlaces` decimals (default 10) and values
 * whose absolute magnitude falls below 5e-12 are snapped to zero, so results
 * are deterministic and free of floating-point dust.
 *
 * @module dsh-linalg/matrix
 */

export type Matrix = number[][]

export interface MatrixError {
  valid: false
  error: string
}

/** Structural validation result: either a cleaned matrix or an error message. */
export function parseMatrix(input: unknown, maxDim: number, roundPlaces: number): { matrix: Matrix } | { error: string } {
  if (!Array.isArray(input) || input.length === 0) return { error: 'matrix must be a non-empty array of rows (number[][])' }
  if (input.length > maxDim) return { error: `matrix has ${input.length} rows; maximum is ${maxDim}` }
  const rows: Matrix = []
  let cols = -1
  for (let i = 0; i < input.length; i++) {
    const row = input[i]
    if (!Array.isArray(row) || row.length === 0) return { error: `row ${i} must be a non-empty array of numbers` }
    if (cols === -1) cols = row.length
    else if (row.length !== cols) return { error: `row ${i} has ${row.length} entries; expected ${cols} (matrix must be rectangular)` }
    if (cols > maxDim) return { error: `matrix has ${cols} columns; maximum is ${maxDim}` }
    const cleanRow: number[] = []
    for (let j = 0; j < row.length; j++) {
      const v = row[j]
      if (typeof v !== 'number' || !Number.isFinite(v)) return { error: `entry at row ${i} column ${j} is not a finite number` }
      cleanRow.push(cleanValue(v, roundPlaces))
    }
    rows.push(cleanRow)
  }
  return { matrix: rows }
}

/** Round to `roundPlaces` decimals and snap near-zero values to exactly 0. */
export function cleanValue(v: number, roundPlaces: number): number {
  if (!Number.isFinite(v)) return v
  const factor = 10 ** roundPlaces
  let out = Math.round(v * factor) / factor
  if (Math.abs(out) < 5e-12) out = 0
  return out === 0 ? 0 : out // kill negative zero
}

/** Deep-clean every entry of a matrix (used after elimination steps). */
export function cleanMatrix(m: Matrix, roundPlaces: number): Matrix {
  return m.map((row) => row.map((v) => cleanValue(v, roundPlaces)))
}

export function rowsOf(m: Matrix): number {
  return m.length
}

export function colsOf(m: Matrix): number {
  return m[0]?.length ?? 0
}

/** Matrix product A·B; returns null when dimensions are incompatible. */
export function multiply(a: Matrix, b: Matrix): Matrix | null {
  const ra = rowsOf(a)
  const ca = colsOf(a)
  const cb = colsOf(b)
  if (ca !== rowsOf(b)) return null
  const out: Matrix = []
  for (let i = 0; i < ra; i++) {
    const row: number[] = []
    for (let j = 0; j < cb; j++) {
      let sum = 0
      for (let k = 0; k < ca; k++) {
        sum += (a[i]?.[k] ?? 0) * (b[k]?.[j] ?? 0)
      }
      row.push(sum)
    }
    out.push(row)
  }
  return out
}

export function transpose(m: Matrix): Matrix {
  const r = rowsOf(m)
  const c = colsOf(m)
  const out: Matrix = []
  for (let j = 0; j < c; j++) {
    const row: number[] = []
    for (let i = 0; i < r; i++) row.push(m[i]?.[j] ?? 0)
    out.push(row)
  }
  return out
}

export function trace(m: Matrix): number | null {
  if (rowsOf(m) !== colsOf(m)) return null
  let sum = 0
  for (let i = 0; i < rowsOf(m); i++) sum += m[i]?.[i] ?? 0
  return sum
}

/**
 * Determinant via Gaussian elimination with partial pivoting.
 * Returns the determinant (0 for exactly-singular matrices within pivot
 * tolerance) or null when the matrix is not square.
 */
export function determinant(m: Matrix, roundPlaces: number): number | null {
  const n = rowsOf(m)
  if (n !== colsOf(m)) return null
  const a: Matrix = m.map((row) => [...row])
  let det = 1
  for (let col = 0; col < n; col++) {
    // partial pivoting: largest magnitude in this column at/under diagonal
    let pivotRow = col
    let best = Math.abs(a[col]?.[col] ?? 0)
    for (let r = col + 1; r < n; r++) {
      const mag = Math.abs(a[r]?.[col] ?? 0)
      if (mag > best) {
        best = mag
        pivotRow = r
      }
    }
    if (pivotRow !== col) {
      const tmp = a[pivotRow]!
      a[pivotRow] = a[col]!
      a[col] = tmp
      det = -det
    }
    const pivot = a[col]?.[col] ?? 0
    if (Math.abs(pivot) < 1e-12) return 0
    det *= pivot
    for (let r = col + 1; r < n; r++) {
      const factor = (a[r]?.[col] ?? 0) / pivot
      if (factor === 0) continue
      for (let c = col; c < n; c++) {
        a[r]![c] = (a[r]?.[c] ?? 0) - factor * (a[col]?.[c] ?? 0)
      }
    }
  }
  return cleanValue(det, roundPlaces)
}

/** Inverse via Gauss–Jordan with partial pivoting; null when non-square or singular. */
export function inverse(m: Matrix, roundPlaces: number): Matrix | null {
  const n = rowsOf(m)
  if (n !== colsOf(m)) return null
  // augmented [A | I]
  const aug: Matrix = m.map((row, i) => {
    const identityRow: number[] = Array.from({ length: n }, (_, j) => (j === i ? 1 : 0))
    return [...row, ...identityRow]
  })
  for (let col = 0; col < n; col++) {
    let pivotRow = col
    let best = Math.abs(aug[col]?.[col] ?? 0)
    for (let r = col + 1; r < n; r++) {
      const mag = Math.abs(aug[r]?.[col] ?? 0)
      if (mag > best) {
        best = mag
        pivotRow = r
      }
    }
    if (pivotRow !== col) {
      const tmp = aug[pivotRow]!
      aug[pivotRow] = aug[col]!
      aug[col] = tmp
    }
    const pivot = aug[col]?.[col] ?? 0
    if (Math.abs(pivot) < 1e-12) return null // singular
    for (let c = 0; c < 2 * n; c++) aug[col]![c] = (aug[col]?.[c] ?? 0) / pivot
    for (let r = 0; r < n; r++) {
      if (r === col) continue
      const factor = aug[r]?.[col] ?? 0
      if (factor === 0) continue
      for (let c = 0; c < 2 * n; c++) {
        aug[r]![c] = (aug[r]?.[c] ?? 0) - factor * (aug[col]?.[c] ?? 0)
      }
    }
  }
  return cleanMatrix(aug.map((row) => row.slice(n)), roundPlaces)
}

/** Reduced row echelon form via Gauss–Jordan with partial pivoting. */
export function rref(m: Matrix, roundPlaces: number): Matrix {
  const a: Matrix = m.map((row) => [...row])
  const rows = rowsOf(a)
  const cols = colsOf(a)
  let pivotCol = 0
  for (let pivotRow = 0; pivotRow < rows && pivotCol < cols; pivotRow++) {
    // find best pivot in this column at/under current row
    let bestRow = pivotRow
    let best = Math.abs(a[pivotRow]?.[pivotCol] ?? 0)
    for (let r = pivotRow + 1; r < rows; r++) {
      const mag = Math.abs(a[r]?.[pivotCol] ?? 0)
      if (mag > best) {
        best = mag
        bestRow = r
      }
    }
    if (best < 1e-12) {
      pivotCol++
      pivotRow-- // no pivot in this column; stay on this row for the next column
      continue
    }
    if (bestRow !== pivotRow) {
      const tmp = a[bestRow]!
      a[bestRow] = a[pivotRow]!
      a[pivotRow] = tmp!
    }
    const pivot = a[pivotRow]?.[pivotCol] ?? 1
    for (let c = pivotCol; c < cols; c++) a[pivotRow]![c] = (a[pivotRow]?.[c] ?? 0) / pivot
    for (let r = 0; r < rows; r++) {
      if (r === pivotRow) continue
      const factor = a[r]?.[pivotCol] ?? 0
      if (factor === 0) continue
      for (let c = pivotCol; c < cols; c++) {
        a[r]![c] = (a[r]?.[c] ?? 0) - factor * (a[pivotRow]?.[c] ?? 0)
      }
    }
    pivotCol++
  }
  return cleanMatrix(a, roundPlaces)
}

/** Rank = number of non-zero rows in the reduced row echelon form. */
export function rankOf(m: Matrix, roundPlaces: number): number {
  let rank = 0
  for (const row of rref(m, roundPlaces)) {
    if (row.some((v) => v !== 0)) rank++
  }
  return rank
}

/** Outcome of an integer matrix power: either the product or a reason it is undefined. */
export type PowerOutcome = { matrix: Matrix } | { error: string }

/**
 * Integer matrix power A^k (k = 0 gives the identity, k < 0 inverts first).
 * Uses binary exponentiation; the exponent is capped by the caller's schema.
 */
export function power(m: Matrix, k: number, roundPlaces: number): PowerOutcome {
  const n = rowsOf(m)
  if (n !== colsOf(m)) return { error: `power requires a square matrix; got ${n}x${colsOf(m)}` }
  if (!Number.isInteger(k)) return { error: `exponent must be an integer; got ${k}` }
  if (Math.abs(k) > 64) return { error: `exponent must be between -64 and 64; got ${k}` }
  let base: Matrix = cleanMatrix(m, roundPlaces)
  let exp = k
  if (k < 0) {
    const inv = inverse(base, roundPlaces)
    if (inv === null) return { error: 'negative exponents need an invertible matrix; this one is singular (or numerically singular)' }
    base = inv
    exp = -k
  }
  let result: Matrix = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)))
  while (exp > 0) {
    if (exp % 2 === 1) result = multiply(result, base) as Matrix
    exp = Math.floor(exp / 2)
    if (exp > 0) base = multiply(base, base) as Matrix
  }
  for (const row of result) {
    for (const v of row) {
      if (!Number.isFinite(v)) return { error: `result overflowed to a non-finite value for exponent ${k}` }
    }
  }
  return { matrix: cleanMatrix(result, roundPlaces) }
}

export type SolveKind = 'unique' | 'infinite' | 'none'

export interface SolveOutcome {
  kind: SolveKind
  /** Unique solution vector (kind === 'unique'). */
  solution?: number[]
  /** Particular solution with free variables set to zero (kind === 'infinite'). */
  particular?: number[]
  /** Basis vectors spanning the null space (kind === 'infinite'). */
  nullspaceBasis?: number[][]
  /** Number of free variables (kind === 'infinite'). */
  freeVariableCount?: number
  /** Reduced row echelon form of the augmented matrix [A|b]. */
  rref?: Matrix
  /** Human-readable explanation for 'none' / 'infinite'. */
  message?: string
}

/**
 * Solve the linear system Ax = b.
 * Returns { error } on structurally invalid input, otherwise a SolveOutcome.
 */
export function solve(a: Matrix, b: number[], roundPlaces: number): { error: string } | SolveOutcome {
  const m = rowsOf(a)
  const n = colsOf(a)
  if (b.length !== m) return { error: `b has ${b.length} entries; expected ${m} (one per row of A)` }
  const aug: Matrix = a.map((row, i) => [...row, b[i] ?? 0])
  // forward elimination with partial pivoting, tracking pivot columns
  const pivotCols: number[] = []
  let pivotRow = 0
  for (let col = 0; col < n && pivotRow < m; col++) {
    let bestRow = pivotRow
    let best = Math.abs(aug[pivotRow]?.[col] ?? 0)
    for (let r = pivotRow + 1; r < m; r++) {
      const mag = Math.abs(aug[r]?.[col] ?? 0)
      if (mag > best) {
        best = mag
        bestRow = r
      }
    }
    if (best < 1e-12) continue // free column
    if (bestRow !== pivotRow) {
      const tmp = aug[bestRow]!
      aug[bestRow] = aug[pivotRow]!
      aug[pivotRow] = tmp!
    }
    const pivot = aug[pivotRow]?.[col] ?? 1
    for (let c = col; c <= n; c++) aug[pivotRow]![c] = (aug[pivotRow]?.[c] ?? 0) / pivot
    for (let r = pivotRow + 1; r < m; r++) {
      const factor = aug[r]?.[col] ?? 0
      if (factor === 0) continue
      for (let c = col; c <= n; c++) {
        aug[r]![c] = (aug[r]?.[c] ?? 0) - factor * (aug[pivotRow]?.[c] ?? 0)
      }
    }
    pivotCols.push(col)
    pivotRow++
  }
  const rank = pivotCols.length
  // consistency check: a row of all-zero coefficients with nonzero constant
  for (let r = rank; r < m; r++) {
    let allZero = true
    for (let c = 0; c < n; c++) {
      if (Math.abs(aug[r]?.[c] ?? 0) >= 1e-12) {
        allZero = false
        break
      }
    }
    if (allZero && Math.abs(aug[r]?.[n] ?? 0) >= 1e-12) {
      return { kind: 'none', message: `inconsistent system: row ${r} reduces to 0 = ${aug[r]?.[n]} (no solution exists)` }
    }
  }
  if (rank === n) {
    // back substitution on the first `rank` rows (raw doubles, cleaned at the end)
    const solution: number[] = Array.from({ length: n }, () => 0)
    for (let i = rank - 1; i >= 0; i--) {
      const col = pivotCols[i] ?? i
      let sum = aug[i]?.[n] ?? 0
      for (let c = col + 1; c < n; c++) sum -= (aug[i]?.[c] ?? 0) * (solution[c] ?? 0)
      solution[col] = sum / (aug[i]?.[col] ?? 1)
    }
    return { kind: 'unique', solution: cleanMatrix([solution], roundPlaces)[0] }
  }
  // infinite: compute RREF of the augmented matrix, particular + nullspace basis
  const reduced = rref(aug, roundPlaces)
  const freeCols: number[] = []
  for (let c = 0; c < n; c++) {
    if (!pivotCols.includes(c)) freeCols.push(c)
  }
  const particular: number[] = Array.from({ length: n }, () => 0)
  for (let i = 0; i < rank; i++) {
    const col = pivotCols[i] ?? 0
    particular[col] = reduced[i]?.[n] ?? 0
  }
  const basis: Matrix = []
  for (const freeCol of freeCols) {
    const vec: number[] = Array.from({ length: n }, () => 0)
    vec[freeCol] = 1
    for (let i = 0; i < rank; i++) {
      const col = pivotCols[i] ?? 0
      vec[col] = -(reduced[i]?.[freeCol] ?? 0)
    }
    basis.push(cleanMatrix([vec], roundPlaces)[0] ?? [])
  }
  const message = `system has ${rank} pivot(s) and ${freeCols.length} free variable(s); ` +
    'general solution = particular solution + any linear combination of the nullspace basis vectors'
  return { kind: 'infinite', particular, nullspaceBasis: basis, freeVariableCount: freeCols.length, rref: reduced, message }
}

/** Spectrum of a real symmetric matrix, as produced by {@link eigenSymmetric}. */
export interface EigenOutcome {
  /** Eigenvalues in descending order. */
  eigenvalues: number[]
  /** Eigenvectors as ROWS in the same order (row i belongs to eigenvalue i), each unit length. */
  eigenvectors: Matrix
  /** Number of Jacobi rotations applied. */
  rotations: number
  /** Largest |A·v − λ·v| entry over all eigenpairs (honest accuracy report). */
  maxResidual: number
  /** Sum of the eigenvalues — equals the trace, kept as a self-check. */
  traceSum: number
}

/**
 * Eigen-decomposition of a real symmetric matrix via the cyclic Jacobi method.
 *
 * Returns { error } when the matrix is not square, is not symmetric within
 * tolerance, or fails to converge — it never fabricates a spectrum.
 * `roundPlaces` only affects the reported numbers; the iteration itself runs
 * in full double precision.
 */
export function eigenSymmetric(m: Matrix, roundPlaces: number): { error: string } | EigenOutcome {
  const n = rowsOf(m)
  if (n !== colsOf(m)) return { error: `eigen-decomposition requires a square matrix; got ${n}x${colsOf(m)}` }
  let scale = 0
  for (const row of m) for (const v of row) scale = Math.max(scale, Math.abs(v))
  const symTol = 1e-9 * (1 + scale)
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const diff = Math.abs((m[i]?.[j] ?? 0) - (m[j]?.[i] ?? 0))
      if (diff > symTol) {
        return {
          error: `matrix is not symmetric: (${i},${j}) = ${m[i]?.[j]} but (${j},${i}) = ${m[j]?.[i]}, difference ${diff} exceeds tolerance ${symTol}; only real symmetric matrices have a guaranteed real spectrum here`,
        }
      }
    }
  }
  const a: Matrix = m.map((row) => [...row])
  const v: Matrix = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)))
  // symmetrise rounding-level asymmetries so every rotation stays exact
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const avg = ((a[i]?.[j] ?? 0) + (a[j]?.[i] ?? 0)) / 2
      a[i]![j] = avg
      a[j]![i] = avg
    }
  }
  let frobenius = 0
  for (const row of a) for (const x of row) frobenius += x * x
  const tol = 1e-13 * (1 + Math.sqrt(frobenius))
  const maxSweeps = 100
  let rotations = 0
  let converged = false
  const offDiagonal = (): number => {
    let off = 0
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += (a[p]?.[q] ?? 0) ** 2
    return Math.sqrt(off)
  }
  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    if (offDiagonal() <= tol) {
      converged = true
      break
    }
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = a[p]?.[q] ?? 0
        if (Math.abs(apq) <= tol / (n === 0 ? 1 : n)) continue
        const theta = ((a[q]?.[q] ?? 0) - (a[p]?.[p] ?? 0)) / (2 * apq)
        const t = theta === 0 ? 1 : Math.sign(theta) / (Math.abs(theta) + Math.sqrt(theta * theta + 1))
        const c = 1 / Math.sqrt(t * t + 1)
        const s = t * c
        for (let k = 0; k < n; k++) {
          const akp = a[k]?.[p] ?? 0
          const akq = a[k]?.[q] ?? 0
          a[k]![p] = c * akp - s * akq
          a[k]![q] = s * akp + c * akq
        }
        for (let k = 0; k < n; k++) {
          const apk = a[p]?.[k] ?? 0
          const aqk = a[q]?.[k] ?? 0
          a[p]![k] = c * apk - s * aqk
          a[q]![k] = s * apk + c * aqk
        }
        for (let k = 0; k < n; k++) {
          const vkp = v[k]?.[p] ?? 0
          const vkq = v[k]?.[q] ?? 0
          v[k]![p] = c * vkp - s * vkq
          v[k]![q] = s * vkp + c * vkq
        }
        rotations++
      }
    }
  }
  if (!converged && offDiagonal() > tol) {
    return {
      error: `Jacobi iteration did not converge within ${maxSweeps} sweeps (${rotations} rotations, off-diagonal norm ${offDiagonal()}); no spectrum reported rather than an inaccurate one`,
    }
  }
  const order = Array.from({ length: n }, (_, i) => i).sort((i, j) => (a[j]?.[j] ?? 0) - (a[i]?.[i] ?? 0))
  const rawEigenvalues: number[] = []
  const eigenvalues: number[] = []
  const eigenvectors: Matrix = []
  for (const index of order) {
    const lambda = a[index]?.[index] ?? 0
    let column = v.map((row) => row[index] ?? 0)
    let norm2 = 0
    for (const x of column) norm2 += x * x
    const length = Math.sqrt(norm2)
    if (length > 0) column = column.map((x) => x / length)
    // Deterministic sign: the leading non-negligible component is positive.
    // (A largest-magnitude rule would flip unpredictably when two components
    // tie in absolute value up to floating-point dust.)
    let lead = 0
    while (lead < column.length && Math.abs(column[lead] ?? 0) <= 1e-8) lead++
    if (lead < column.length && (column[lead] ?? 0) < 0) column = column.map((x) => -x)
    rawEigenvalues.push(lambda)
    eigenvalues.push(cleanValue(lambda, roundPlaces))
    eigenvectors.push(column.map((x) => cleanValue(x, roundPlaces)))
  }
  let maxResidual = 0
  for (let i = 0; i < n; i++) {
    const vec = eigenvectors[i] ?? []
    const lambda = rawEigenvalues[i] ?? 0
    for (let r = 0; r < n; r++) {
      let sum = 0
      for (let c = 0; c < n; c++) sum += (m[r]?.[c] ?? 0) * (vec[c] ?? 0)
      maxResidual = Math.max(maxResidual, Math.abs(sum - lambda * (vec[r] ?? 0)))
    }
  }
  let traceSum = 0
  for (const lambda of rawEigenvalues) traceSum += lambda
  return {
    eigenvalues,
    eigenvectors,
    rotations,
    maxResidual: cleanValue(maxResidual, roundPlaces),
    traceSum: cleanValue(traceSum, roundPlaces),
  }
}
