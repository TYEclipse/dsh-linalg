/**
 * Tool definitions for dsh-linalg: four pure-math tools exposed via defineTool.
 *   matrix_multiply — matrix product A·B with dimension checking
 *   matrix_compute  — transpose / determinant / inverse / trace / rref
 *   solve_linear    — solve Ax = b with unique / infinite / none classification
 *   vector_ops      — dot / cross / norm / projection / angle
 *
 * @module dsh-linalg/tools
 */
import { type ToolDefinition } from '@deepseek-ai/dsh-tools';
import type { ResolvedConfig } from './index.ts';
export interface ToolSet {
    matrix_multiply: ToolDefinition;
    matrix_compute: ToolDefinition;
    solve_linear: ToolDefinition;
    vector_ops: ToolDefinition;
}
/** Build all four linalg tool definitions. */
export declare function buildLinalgTools(config: ResolvedConfig): ToolSet;
//# sourceMappingURL=tools.d.ts.map