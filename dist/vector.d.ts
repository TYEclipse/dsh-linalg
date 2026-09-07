/**
 * dsh-linalg vector core: dot / cross products, norms, projection, angle.
 * Pure arithmetic on number[] vectors, zero runtime dependencies.
 *
 * @module dsh-linalg/vector
 */
export interface VectorError {
    valid: false;
    error: string;
}
/** Validate a vector: non-empty array of finite numbers. */
export declare function parseVector(input: unknown, maxLen: number): {
    vector: number[];
} | {
    error: string;
};
/** Dot product; null when lengths differ. */
export declare function dot(a: number[], b: number[]): number | null;
/** Cross product (3D only); null when either vector is not length 3. */
export declare function cross(a: number[], b: number[]): number[] | null;
/** Euclidean (L2) norm. */
export declare function norm(a: number[]): number;
/** Projection of `a` onto `b`; null when lengths differ or b is the zero vector. */
export declare function projection(a: number[], b: number[]): number[] | null;
/** Angle between `a` and `b` in degrees; null on invalid input or zero vector. */
export declare function angleDeg(a: number[], b: number[]): number | null;
//# sourceMappingURL=vector.d.ts.map