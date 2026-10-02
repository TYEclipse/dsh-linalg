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
/** Rank = number of non-zero rows in the reduced row echelon form. */
export declare function rankOf(m: Matrix, roundPlaces: number): number;
/** Outcome of an integer matrix power: either the product or a reason it is undefined. */
export type PowerOutcome = {
    matrix: Matrix;
} | {
    error: string;
};
/**
 * Integer matrix power A^k (k = 0 gives the identity, k < 0 inverts first).
 * Uses binary exponentiation; the exponent is capped by the caller's schema.
 */
export declare function power(m: Matrix, k: number, roundPlaces: number): PowerOutcome;
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
/** Spectrum of a real symmetric matrix, as produced by {@link eigenSymmetric}. */
export interface EigenOutcome {
    /** Eigenvalues in descending order. */
    eigenvalues: number[];
    /** Eigenvectors as ROWS in the same order (row i belongs to eigenvalue i), each unit length. */
    eigenvectors: Matrix;
    /** Number of Jacobi rotations applied. */
    rotations: number;
    /** Largest |A·v − λ·v| entry over all eigenpairs (honest accuracy report). */
    maxResidual: number;
    /** Sum of the eigenvalues — equals the trace, kept as a self-check. */
    traceSum: number;
}
/**
 * Eigen-decomposition of a real symmetric matrix via the cyclic Jacobi method.
 *
 * Returns { error } when the matrix is not square, is not symmetric within
 * tolerance, or fails to converge — it never fabricates a spectrum.
 * `roundPlaces` only affects the reported numbers; the iteration itself runs
 * in full double precision.
 */
export declare function eigenSymmetric(m: Matrix, roundPlaces: number): {
    error: string;
} | EigenOutcome;
//# sourceMappingURL=matrix.d.ts.map