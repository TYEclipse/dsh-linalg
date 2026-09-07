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
import z from '@deepseek-ai/schemastery';
import { buildLinalgTools } from "./tools.js";
/** Stable Cordis plugin name (also the config key under `plugins:`). */
export const name = 'dsh-linalg';
/** Services required before tool registration can start. */
export const inject = ['agents', 'tools'];
export const Config = z.object({
    maxDimension: z.number().step(1).min(1).max(50).default(20),
    roundPlaces: z.number().step(1).min(1).max(15).default(10),
});
/** Resolve loader config into the effective runtime config. */
export function resolveConfig(config) {
    return {
        maxDimension: config.maxDimension ?? 20,
        roundPlaces: config.roundPlaces ?? 10,
    };
}
/** Register every linalg tool on one agent; returns the disposer. */
function decorate(agent, tools) {
    const disposers = Object.values(tools).map((definition) => agent.ctx.tools.register(definition));
    return () => {
        for (const dispose of disposers) {
            try {
                dispose();
            }
            catch {
                // already disposed
            }
        }
    };
}
/** Mount the linalg tools on every live agent and every future one. */
export function apply(ctx, config) {
    const resolved = resolveConfig(config);
    const tools = buildLinalgTools(resolved);
    const disposers = new Set();
    const decorateAgent = (agent) => {
        try {
            disposers.add(decorate(agent, tools));
        }
        catch (error) {
            ctx.logger('linalg').warn(`tool registration for agent ${agent.id} failed: ${error instanceof Error ? error.message : String(error)}`);
        }
    };
    for (const agent of ctx.agents.list())
        decorateAgent(agent);
    const off = ctx.on('agent/created', ({ agent }) => decorateAgent(agent));
    ctx.effect(() => () => {
        off();
        for (const dispose of disposers) {
            try {
                dispose();
            }
            catch {
                // already disposed
            }
        }
        disposers.clear();
    });
}
//# sourceMappingURL=index.js.map