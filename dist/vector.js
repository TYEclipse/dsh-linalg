/**
 * dsh-linalg vector core: dot / cross products, norms, projection, angle.
 * Pure arithmetic on number[] vectors, zero runtime dependencies.
 *
 * @module dsh-linalg/vector
 */
/** Validate a vector: non-empty array of finite numbers. */
export function parseVector(input, maxLen) {
    if (!Array.isArray(input) || input.length === 0)
        return { error: 'vector must be a non-empty array of numbers' };
    if (input.length > maxLen)
        return { error: `vector has ${input.length} entries; maximum is ${maxLen}` };
    for (let i = 0; i < input.length; i++) {
        const v = input[i];
        if (typeof v !== 'number' || !Number.isFinite(v))
            return { error: `entry at index ${i} is not a finite number` };
    }
    return { vector: input };
}
/** Dot product; null when lengths differ. */
export function dot(a, b) {
    if (a.length !== b.length)
        return null;
    let sum = 0;
    for (let i = 0; i < a.length; i++)
        sum += (a[i] ?? 0) * (b[i] ?? 0);
    return sum;
}
/** Cross product (3D only); null when either vector is not length 3. */
export function cross(a, b) {
    if (a.length !== 3 || b.length !== 3)
        return null;
    return [
        (a[1] ?? 0) * (b[2] ?? 0) - (a[2] ?? 0) * (b[1] ?? 0),
        (a[2] ?? 0) * (b[0] ?? 0) - (a[0] ?? 0) * (b[2] ?? 0),
        (a[0] ?? 0) * (b[1] ?? 0) - (a[1] ?? 0) * (b[0] ?? 0),
    ];
}
/** Euclidean (L2) norm. */
export function norm(a) {
    let sum = 0;
    for (const v of a)
        sum += v * v;
    return Math.sqrt(sum);
}
/** Projection of `a` onto `b`; null when lengths differ or b is the zero vector. */
export function projection(a, b) {
    const d = dot(a, b);
    if (d === null)
        return null;
    const bNormSq = dot(b, b);
    if (bNormSq === null || bNormSq === 0)
        return null;
    return b.map((v) => (d / bNormSq) * v);
}
/** Angle between `a` and `b` in degrees; null on invalid input or zero vector. */
export function angleDeg(a, b) {
    const d = dot(a, b);
    if (d === null)
        return null;
    const na = norm(a);
    const nb = norm(b);
    if (na === 0 || nb === 0)
        return null;
    let cosValue = d / (na * nb);
    if (cosValue > 1)
        cosValue = 1;
    if (cosValue < -1)
        cosValue = -1;
    return (Math.acos(cosValue) * 180) / Math.PI;
}
//# sourceMappingURL=vector.js.map