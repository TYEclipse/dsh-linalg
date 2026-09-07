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
