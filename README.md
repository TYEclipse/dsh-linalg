# dsh-linalg

Linear algebra toolbox for [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) (`dsh`): matrix multiply, determinant, inverse, trace, transpose, reduced row echelon form, a linear system solver that classifies unique / infinitely many / no solutions, and vector operations (dot / cross / norm / projection / angle).

Zero runtime dependencies, pure local arithmetic — no network, no processes, no eval.

## Why

Language models frequently make arithmetic errors on matrix multiplication, determinants, inverses and linear systems. This plugin hands those computations to exact local code with Gaussian elimination and partial pivoting, so results are deterministic and correct to 10 decimal places.

## Install

```sh
dsh plugin --profile web add github:TYEclipse/dsh-linalg
```

Replace `web` with your profile name. (Requires `pnpm` on your PATH.)

## Tools

| Tool | What it does |
|---|---|
| `matrix_multiply` | Multiply two matrices A·B with dimension checking |
| `matrix_compute` | One operation per call: `transpose`, `determinant`, `inverse`, `trace`, `rref` |
| `solve_linear` | Solve Ax = b; classifies unique / infinite (particular + nullspace basis + free-variable count) / no solution |
| `vector_ops` | `dot`, `cross` (3D), `norm`, `projection`, `angle` (degrees) |

Input matrices are plain JSON arrays of arrays, e.g. `[[1, 2], [3, 4]]`. Maximum dimension is 20×20 by default (configurable).

## Examples

Determinant:

```
matrix_compute(matrix=[[2,1,1],[1,3,2],[1,0,0]], op="determinant")  →  -1
```

Inverse:

```
matrix_compute(matrix=[[4,7],[2,6]], op="inverse")
→ [[0.6,-0.7],[-0.2,0.4]]
```

Solve 2x+3y=8, x−y=1:

```
solve_linear(a=[[2,3],[1,-1]], b=[8,1])
→ unique solution: [2.2, 1.2]
```

Under-determined system (1 equation, 3 variables) reports the full parametric form:

```
solve_linear(a=[[1,1,1],[1,2,3]], b=[6,14])
→ kind=infinite, particular=[-2,8,0], nullspaceBasis=[[1,-2,1]], freeVariableCount=1
```

Cross product:

```
vector_ops(op="cross", a=[1,2,3], b=[4,5,6])  →  [-3,6,-3]
```

Angle between [3,4] and the x-axis:

```
vector_ops(op="angle", a=[3,4], b=[4,0])  →  53.1301023542  (degrees)
```

## Numerical behaviour

- Gaussian elimination with partial pivoting (largest-magnitude pivot per column).
- All output numbers are rounded to 10 decimal places (configurable `roundPlaces`); values below 5e-12 are snapped to exactly 0, so `-0` never appears.
- Pivot tolerance is 1e-12: entries below that are treated as zero, so numerically singular matrices are reported as singular instead of producing garbage.
- Singular inverse requests fail cleanly with an error message; inconsistent systems return `kind: "none"` with the reducing row equation.

## Configuration

```yaml
- name: 'github:TYEclipse/dsh-linalg'
  config:
    maxDimension: 20   # max rows/columns (1–50)
    roundPlaces: 10    # decimals in every output number (1–15)
```

## Development

```sh
pnpm install
pnpm build
pnpm test
pnpm lint
```

## License

MIT
