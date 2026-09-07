/**
 * dsh-linalg — linear algebra toolbox for DeepSeek Harness.
 *
 * Four pure-math tools, zero runtime dependencies:
 *   matrix_multiply — matrix product A·B with dimension checking
 *   matrix_compute  — transpose / determinant / inverse / trace / rref
 *   solve_linear    — solve Ax = b, classified as unique / infinite / none
 *   vector_ops      — dot / cross / norm / projection / angle
 *
 * All computation is local arithmetic (Gaussian elimination with partial
 * pivoting); no network, no processes, no eval.
 *
 * @module dsh-linalg
 */
import type { Context } from '@deepseek-ai/cordis';
import z from '@deepseek-ai/schemastery';
/** Stable Cordis plugin name (also the config key under `plugins:`). */
export declare const name = "dsh-linalg";
/** Services required before tool registration can start. */
export declare const inject: string[];
/** Plugin configuration, resolved with defaults by the loader. */
export interface Config {
    /** Maximum matrix dimension (rows/columns) accepted by any tool (1–50). */
    maxDimension?: number;
    /** Decimal places every output number is rounded to (1–15). */
    roundPlaces?: number;
}
export declare const Config: z<Config>;
/** Config with every default resolved (all fields guaranteed). */
export interface ResolvedConfig {
    maxDimension: number;
    roundPlaces: number;
}
/** Resolve loader config into the effective runtime config. */
export declare function resolveConfig(config: Config): ResolvedConfig;
/** Mount the linalg tools on every live agent and every future one. */
export declare function apply(ctx: Context, config: Config): void;
//# sourceMappingURL=index.d.ts.map