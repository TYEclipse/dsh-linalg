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
export type Matrix = number[][];
export interface MatrixError {
    valid: false;
    error: string;
}
/** Structural validation result: either a cleaned matrix or an error message. */
export declare function parseMatrix(input: unknown, maxDim: number, roundPlaces: number): {
    matrix: Matrix;
} | {
    error: string;
};
/** Round to `roundPlaces` decimals and snap near-zero values to exactly 0. */
export declare function cleanValue(v: number, roundPlaces: number): number;
/** Deep-clean every entry of a matrix (used after elimination steps). */
export declare function cleanMatrix(m: Matrix, roundPlaces: number): Matrix;
export declare function rowsOf(m: Matrix): number;
export declare function colsOf(m: Matrix): number;
/** Matrix product A·B; returns null when dimensions are incompatible. */
export declare function multiply(a: Matrix, b: Matrix): Matrix | null;
export declare function transpose(m: Matrix): Matrix;
export declare function trace(m: Matrix): number | null;
/**
 * Determinant via Gaussian elimination with partial pivoting.
 * Returns the determinant (0 for exactly-singular matrices within pivot
 * tolerance) or null when the matrix is not square.
 */
export declare function determinant(m: Matrix, roundPlaces: number): number | null;
/** Inverse via Gauss–Jordan with partial pivoting; null when non-square or singular. */
export declare function inverse(m: Matrix, roundPlaces: number): Matrix | null;
/** Reduced row echelon form via Gauss–Jordan with partial pivoting. */
export declare function rref(m: Matrix, roundPlaces: number): Matrix;
export type SolveKind = 'unique' | 'infinite' | 'none';
export interface SolveOutcome {
    kind: SolveKind;
    /** Unique solution vector (kind === 'unique'). */
    solution?: number[];
    /** Particular solution with free variables set to zero (kind === 'infinite'). */
    particular?: number[];
    /** Basis vectors spanning the null space (kind === 'infinite'). */
    nullspaceBasis?: number[][];
    /** Number of free variables (kind === 'infinite'). */
    freeVariableCount?: number;
    /** Reduced row echelon form of the augmented matrix [A|b]. */
    rref?: Matrix;
    /** Human-readable explanation for 'none' / 'infinite'. */
    message?: string;
}
/**
 * Solve the linear system Ax = b.
 * Returns { error } on structurally invalid input, otherwise a SolveOutcome.
 */
export declare function solve(a: Matrix, b: number[], roundPlaces: number): {
    error: string;
} | SolveOutcome;
//# sourceMappingURL=matrix.d.ts.map