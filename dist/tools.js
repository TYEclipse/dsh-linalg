/**
 * Tool definitions for dsh-linalg: four pure-math tools exposed via defineTool.
 *   matrix_multiply — matrix product A·B with dimension checking
 *   matrix_compute  — transpose / determinant / inverse / trace / rref
 *   solve_linear    — solve Ax = b with unique / infinite / none classification
 *   vector_ops      — dot / cross / norm / projection / angle
 *
 * @module dsh-linalg/tools
 */
import { defineTool } from '@deepseek-ai/dsh-tools';
import { cleanMatrix, cleanValue, determinant, inverse, multiply, parseMatrix, rref, solve, trace, transpose, } from "./matrix.js";
import { angleDeg, cross, dot, norm, parseVector, projection } from "./vector.js";
const MATRIX_OPS = ['transpose', 'determinant', 'inverse', 'trace', 'rref'];
const VECTOR_OPS = ['dot', 'cross', 'norm', 'projection', 'angle'];
function renderMatrix(name, m) {
    return `${name} (${m.length}x${m[0]?.length ?? 0}):\n${m.map((row) => `  [${row.join(', ')}]`).join('\n')}`;
}
function renderMultiply(value) {
    const result = value;
    if (!result.valid)
        return `matrix multiply failed: ${result.error}`;
    return renderMatrix('product', result.product);
}
function renderCompute(value) {
    const result = value;
    if (!result.valid)
        return `${result.op} failed: ${result.error}`;
    if (result.result !== undefined)
        return `${result.op} = ${result.result}`;
    return renderMatrix(result.op, result.matrix);
}
function renderSolve(value) {
    const result = value;
    if (!result.valid)
        return `solve_linear failed: ${result.error}`;
    if (result.kind === 'none')
        return `no solution: ${result.message}`;
    if (result.kind === 'unique') {
        const vars = result.solution.map((v, i) => `x${i + 1} = ${v}`).join(', ');
        return `unique solution: ${vars}`;
    }
    const particular = `particular solution: [${result.particular.join(', ')}]`;
    const basis = result.nullspaceBasis.map((vec) => `[${vec.join(', ')}]`).join('; ');
    return `infinitely many solutions (${result.freeVariableCount} free variable(s))\n  ${particular}\n  nullspace basis: ${basis}\n  ${result.message}`;
}
function renderVector(value) {
    const result = value;
    if (!result.valid)
        return `vector op failed: ${result.error}`;
    if (result.result !== undefined)
        return `${result.op} = ${result.result}`;
    return `${result.op} = [${result.vector.join(', ')}]`;
}
/** Build all four linalg tool definitions. */
export function buildLinalgTools(config) {
    const maxDim = config.maxDimension;
    const roundPlaces = config.roundPlaces;
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
            render: (_args, value) => [{ type: 'text', text: renderMultiply(value) }],
        },
        async execute(args) {
            const pa = parseMatrix(args.a, maxDim, roundPlaces);
            if ('error' in pa)
                return { valid: false, error: `A: ${pa.error}` };
            const pb = parseMatrix(args.b, maxDim, roundPlaces);
            if ('error' in pb)
                return { valid: false, error: `B: ${pb.error}` };
            const product = multiply(pa.matrix, pb.matrix);
            if (product === null) {
                return { valid: false, error: `dimension mismatch: A is ${pa.matrix.length}x${pa.matrix[0]?.length ?? 0} but B is ${pb.matrix.length}x${pb.matrix[0]?.length ?? 0} (columns of A must equal rows of B)` };
            }
            const out = { valid: true, rows: product.length, cols: product[0]?.length ?? 0, product };
            return out;
        },
    });
    const matrix_compute = defineTool({
        name: 'matrix_compute',
        description: 'Compute a single matrix operation: transpose, determinant, inverse, trace, or reduced row '
            + 'echelon form (rref). Determinant and inverse require square matrices; singular matrices (determinant 0) '
            + 'report an error instead of producing garbage. Pure arithmetic, no external calls.',
        parameters: {
            matrix: { type: 'array', required: true, items: { type: 'array', items: { type: 'number' } }, description: `Matrix as rows of numbers, e.g. [[1,2],[3,4]]. Maximum dimension ${maxDim}x${maxDim}.` },
            op: { type: 'string', required: true, enum: [...MATRIX_OPS], description: 'Operation to perform: transpose | determinant | inverse | trace | rref.' },
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
            render: (_args, value) => [{ type: 'text', text: renderCompute(value) }],
        },
        async execute(args) {
            const op = args.op;
            const pm = parseMatrix(args.matrix, maxDim, roundPlaces);
            if ('error' in pm)
                return { valid: false, op, error: pm.error };
            const m = pm.matrix;
            switch (op) {
                case 'transpose': {
                    const out = { valid: true, op, matrix: transpose(m) };
                    return out;
                }
                case 'determinant': {
                    const d = determinant(m, roundPlaces);
                    if (d === null)
                        return { valid: false, op, error: `determinant requires a square matrix; got ${m.length}x${m[0]?.length ?? 0}` };
                    const out = { valid: true, op, result: d };
                    return out;
                }
                case 'inverse': {
                    const inv = inverse(m, roundPlaces);
                    if (inv === null) {
                        const d = determinant(m, roundPlaces);
                        if (d === null)
                            return { valid: false, op, error: `inverse requires a square matrix; got ${m.length}x${m[0]?.length ?? 0}` };
                        if (d === 0)
                            return { valid: false, op, error: 'matrix is singular (determinant 0); it has no inverse' };
                        return { valid: false, op, error: 'matrix is numerically singular (pivot below tolerance)' };
                    }
                    const out = { valid: true, op, matrix: inv };
                    return out;
                }
                case 'trace': {
                    const t = trace(m);
                    if (t === null)
                        return { valid: false, op, error: `trace requires a square matrix; got ${m.length}x${m[0]?.length ?? 0}` };
                    const out = { valid: true, op, result: t };
                    return out;
                }
                case 'rref': {
                    const out = { valid: true, op, matrix: rref(m, roundPlaces) };
                    return out;
                }
                default:
                    return { valid: false, op, error: `unknown operation: ${op}` };
            }
        },
    });
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
            render: (_args, value) => [{ type: 'text', text: renderSolve(value) }],
        },
        async execute(args) {
            const pa = parseMatrix(args.a, maxDim, roundPlaces);
            if ('error' in pa)
                return { valid: false, kind: 'none', error: `A: ${pa.error}` };
            const pb = parseVector(args.b, maxDim);
            if ('error' in pb)
                return { valid: false, kind: 'none', error: `b: ${pb.error}` };
            const outcome = solve(pa.matrix, pb.vector, roundPlaces);
            if ('error' in outcome)
                return { valid: false, kind: 'none', error: outcome.error };
            const out = { valid: true, kind: outcome.kind };
            if (outcome.solution !== undefined)
                out.solution = outcome.solution;
            if (outcome.particular !== undefined)
                out.particular = outcome.particular;
            if (outcome.nullspaceBasis !== undefined)
                out.nullspaceBasis = outcome.nullspaceBasis;
            if (outcome.freeVariableCount !== undefined)
                out.freeVariableCount = outcome.freeVariableCount;
            if (outcome.rref !== undefined)
                out.rref = outcome.rref;
            if (outcome.message !== undefined)
                out.message = outcome.message;
            return out;
        },
    });
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
            render: (_args, value) => [{ type: 'text', text: renderVector(value) }],
        },
        async execute(args) {
            const op = args.op;
            const pa = parseVector(args.a, maxDim);
            if ('error' in pa)
                return { valid: false, op, error: `a: ${pa.error}` };
            const a = pa.vector;
            switch (op) {
                case 'norm': {
                    const out = { valid: true, op, result: cleanValue(norm(a), roundPlaces) };
                    return out;
                }
                case 'dot':
                case 'cross':
                case 'projection':
                case 'angle': {
                    if (args.b === undefined)
                        return { valid: false, op, error: `${op} requires a second vector b` };
                    const pb = parseVector(args.b, maxDim);
                    if ('error' in pb)
                        return { valid: false, op, error: `b: ${pb.error}` };
                    const b = pb.vector;
                    if (op === 'dot') {
                        const d = dot(a, b);
                        if (d === null)
                            return { valid: false, op, error: `dot product requires equal-length vectors; got ${a.length} and ${b.length}` };
                        const out = { valid: true, op, result: cleanValue(d, roundPlaces) };
                        return out;
                    }
                    if (op === 'cross') {
                        const c = cross(a, b);
                        if (c === null)
                            return { valid: false, op, error: `cross product requires 3D vectors; got lengths ${a.length} and ${b.length}` };
                        const out = { valid: true, op, vector: cleanMatrix([c], roundPlaces)[0] };
                        return out;
                    }
                    if (op === 'projection') {
                        const p = projection(a, b);
                        if (p === null)
                            return { valid: false, op, error: `projection requires equal-length vectors and a non-zero b; got lengths ${a.length} and ${b.length}` };
                        const out = { valid: true, op, vector: cleanMatrix([p], roundPlaces)[0] };
                        return out;
                    }
                    const ang = angleDeg(a, b);
                    if (ang === null)
                        return { valid: false, op, error: `angle requires equal-length non-zero vectors; got lengths ${a.length} and ${b.length}` };
                    const out = { valid: true, op, result: cleanValue(ang, roundPlaces) };
                    return out;
                }
                default:
                    return { valid: false, op, error: `unknown operation: ${op}` };
            }
        },
    });
    return { matrix_multiply, matrix_compute, solve_linear, vector_ops };
}
//# sourceMappingURL=tools.js.map