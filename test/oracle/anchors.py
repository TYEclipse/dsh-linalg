#!/usr/bin/env python3
"""anchors.py — independent Python oracle for dsh-linalg (v0.2.0).

Every numeric expectation asserted in ``test/matrix.test.ts``,
``test/vector.test.ts`` and ``test/tools.test.ts`` is produced here, from
primitives that share no code with the TypeScript implementation:

  - matrix products / determinants / inverses / rref / rank / powers: exact
    ``fractions.Fraction`` Gaussian elimination (the TS side runs float
    doubles with partial pivoting — different arithmetic entirely)
  - matrix_eigen anchors: closed-form spectra (2x2 quadratic, the 4x4 path
    graph ``2 - 2·cos(k·pi/5)``, ``1 ± 2·sqrt(2)``, and an integer matrix
    built as ``H·diag(8,4,0,-8)·Hᵀ`` with an exact integer spectrum) plus a
    pure-Python Jacobi implementation used only to cross-check
  - vectors: ``math.sqrt`` / ``math.acos`` closed forms

Sections mirror the test files so a reviewer can walk assertion -> anchor:

  A. legacy (v0.1.x) anchors — parse, multiply, det, inverse, trace, rref, solve, vectors
  B. matrix_compute v0.2.0 anchors — rank, power
  C. matrix_eigen v0.2.0 anchors — spectra, eigenvectors, invariants

Usage:
  python3 test/oracle/anchors.py            # print every anchor
  python3 test/oracle/anchors.py --check    # + self-checks (exact vs. analytic)
"""

from fractions import Fraction as F
import math
import sys


def d10(x):
    """Round to 10 decimals the way the plugin reports numbers."""
    v = round(float(x), 10)
    return 0.0 if abs(v) < 5e-12 else v


def exact(lst):
    """Render exact rationals as floats (only used for reporting)."""
    return [float(x) for x in lst]


# --------------------------------------------------------------------------
# exact Fraction linear algebra (independent of the TypeScript implementation)
# --------------------------------------------------------------------------

def to_frac(m):
    return [[F(x) for x in row] for row in m]


def mul(a, b):
    n, k, p = len(a), len(b), len(b[0])
    return [[sum(a[i][t] * b[t][j] for t in range(k)) for j in range(p)] for i in range(n)]


def det_exact(m):
    a = [row[:] for row in m]
    n = len(a)
    if n != len(a[0]):
        return None
    out = F(1)
    for col in range(n):
        pivot = None
        for r in range(col, n):
            if a[r][col] != 0:
                pivot = r
                break
        if pivot is None:
            return F(0)
        if pivot != col:
            a[col], a[pivot] = a[pivot], a[col]
            out = -out
        out *= a[col][col]
        for r in range(col + 1, n):
            f = a[r][col] / a[col][col]
            if f:
                for c in range(col, n):
                    a[r][c] -= f * a[col][c]
    return out


def inverse_exact(m):
    n = len(m)
    if n != len(m[0]):
        return None
    if det_exact(m) == 0:
        return None
    aug = [row[:] + [F(1) if i == j else F(0) for j in range(n)] for i, row in enumerate(m)]
    for col in range(n):
        pivot = next((r for r in range(col, n) if aug[r][col] != 0), None)
        if pivot is None:
            return None
        aug[col], aug[pivot] = aug[pivot], aug[col]
        pv = aug[col][col]
        aug[col] = [x / pv for x in aug[col]]
        for r in range(n):
            if r == col or aug[r][col] == 0:
                continue
            f = aug[r][col]
            aug[r] = [x - f * y for x, y in zip(aug[r], aug[col])]
    return [row[n:] for row in aug]


def rref_exact(m):
    a = [row[:] for row in m]
    rows, cols = len(a), len(a[0])
    pivot_cols = []
    r = 0
    for c in range(cols):
        if r >= rows:
            break
        best = None
        for rr in range(r, rows):
            if a[rr][c] != 0:
                best = rr
                break
        if best is None:
            continue
        a[r], a[best] = a[best], a[r]
        pv = a[r][c]
        a[r] = [x / pv for x in a[r]]
        for rr in range(rows):
            if rr == r or a[rr][c] == 0:
                continue
            f = a[rr][c]
            a[rr] = [x - f * y for x, y in zip(a[rr], a[r])]
        pivot_cols.append(c)
        r += 1
    return a, pivot_cols


def rank_exact(m):
    _, cols = rref_exact(m)
    return len(cols)


def solve_exact(a, b):
    """Classify Ax=b exactly: unique / infinite / none (plus witness values)."""
    m, n = len(a), len(a[0])
    aug = [a[i][:] + [F(b[i])] for i in range(m)]
    red, pivots = rref_exact(aug)
    for row in red:
        if all(x == 0 for x in row[:n]) and row[n] != 0:
            return {"kind": "none"}
    if len(pivots) == n:
        solution = [F(0)] * n
        for i, c in enumerate(pivots):
            solution[c] = red[i][n]
        return {"kind": "unique", "solution": solution}
    particular = [F(0)] * n
    for i, c in enumerate(pivots):
        particular[c] = red[i][n]
    free = [c for c in range(n) if c not in pivots]
    basis = []
    for fc in free:
        vec = [F(0)] * n
        vec[fc] = F(1)
        for i, c in enumerate(pivots):
            vec[c] = -red[i][fc]
        basis.append(vec)
    return {"kind": "infinite", "particular": particular, "basis": basis, "free": len(free), "rref": red}


# --------------------------------------------------------------------------
# a second, independent eigen path: pure-Python Jacobi (cross-check only)
# --------------------------------------------------------------------------

def jacobi_eigenvalues(m, sweeps=100):
    n = len(m)
    a = [[float(x) for x in row] for row in m]
    for _ in range(sweeps):
        off = math.sqrt(sum(a[p][q] ** 2 for p in range(n) for q in range(p + 1, n)))
        if off <= 1e-14 * (1 + math.sqrt(sum(x * x for row in a for x in row))):
            break
        for p in range(n):
            for q in range(p + 1, n):
                if a[p][q] == 0:
                    continue
                theta = (a[q][q] - a[p][p]) / (2 * a[p][q])
                t = 1.0 if theta == 0 else math.copysign(1, theta) / (abs(theta) + math.sqrt(theta * theta + 1))
                c = 1 / math.sqrt(t * t + 1)
                s = t * c
                for k in range(n):
                    akp, akq = a[k][p], a[k][q]
                    a[k][p], a[k][q] = c * akp - s * akq, s * akp + c * akq
                for k in range(n):
                    apk, aqk = a[p][k], a[q][k]
                    a[p][k], a[q][k] = c * apk - s * aqk, s * apk + c * aqk
    return sorted((a[i][i] for i in range(n)), reverse=True)


def path_spectrum(size):
    """Eigenvalues of the size x size tridiagonal (2 on the diagonal, -1 off):
    2 - 2*cos(k*pi/(size+1)), descending."""
    return sorted((2 - 2 * math.cos(k * math.pi / (size + 1)) for k in range(1, size + 1)), reverse=True)


def path_eigenvector(size, k):
    """Unit eigenvector of that matrix for mode k: sin(j*k*pi/(size+1)), j = 1..size."""
    v = [math.sin(j * k * math.pi / (size + 1)) for j in range(1, size + 1)]
    norm = math.sqrt(sum(x * x for x in v))
    return signed([x / norm for x in v])


def signed(v):
    """Apply the plugin's deterministic sign convention: the leading non-negligible
    component is positive."""
    for x in v:
        if abs(x) > 1e-8:
            return [(-y if x < 0 else y) for y in v]
    return list(v)


# --------------------------------------------------------------------------
# A. legacy anchors (already asserted by the v0.1.x tests)
# --------------------------------------------------------------------------

def section_a():
    print("== A. legacy anchors ==")
    a = to_frac([[1, 2, 3], [4, 5, 6]])
    b = to_frac([[7, 8], [9, 10], [11, 12]])
    print("multiply 2x3 * 3x2 =", [[int(x) for x in row] for row in mul(a, b)])
    print("multiply 1x1 =", [[int(x) for x in mul(to_frac([[3]]), to_frac([[-7]]))[0]]])
    identity = to_frac([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    cube = to_frac([[2, 0, 1], [1, 3, 0], [0, 1, 4]])
    print("multiply by identity preserves:", mul(cube, identity) == cube)
    for label, m in [
        ("[[1,2],[3,4]]", [[1, 2], [3, 4]]),
        ("[[2,1,1],[1,3,2],[1,0,0]]", [[2, 1, 1], [1, 3, 2], [1, 0, 0]]),
        ("4x4 pivot [[0,1,0,3],[2,0,0,1],[0,0,1,0],[0,2,0,0]]", [[0, 1, 0, 3], [2, 0, 0, 1], [0, 0, 1, 0], [0, 2, 0, 0]]),
        ("singular [[1,2],[2,4]]", [[1, 2], [2, 4]]),
        ("[[1/2,1/3],[1/4,1/5]]", [["1/2", "1/3"], ["1/4", "1/5"]]),
    ]:
        m2 = [[F(x) if isinstance(x, str) else x for x in row] for row in m]
        print(f"det({label}) =", d10(det_exact(m2)), "| exact:", det_exact(m2))
    for label, m in [
        ("[[4,7],[2,6]]", [[4, 7], [2, 6]]),
        ("[[1,2,3],[0,1,4],[5,6,0]]", [[1, 2, 3], [0, 1, 4], [5, 6, 0]]),
        ("[[1/2,1/3],[1/4,1/5]]", [["1/2", "1/3"], ["1/4", "1/5"]]),
    ]:
        m2 = [[F(x) if isinstance(x, str) else x for x in row] for row in m]
        inv = inverse_exact(m2)
        print(f"inverse({label}) =", [exact(row) for row in inv] if inv else None,
              "| exact:", [[str(x) for x in row] for row in inv] if inv else None)
    print("trace([[1,2,3],[4,5,6],[7,8,9]]) =", 1 + 5 + 9)
    print("transpose([[1,2,3],[4,5,6]]) =", [[1, 4], [2, 5], [3, 6]])
    for label, m in [
        ("[[1,2,3],[4,5,6],[7,8,9]]", [[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        ("[[1,2,3,4],[2,4,6,8]]", [[1, 2, 3, 4], [2, 4, 6, 8]]),
        ("[[0,2,4],[1,2,3]]", [[0, 2, 4], [1, 2, 3]]),
    ]:
        red, _ = rref_exact(to_frac(m))
        print(f"rref({label}) =", [[int(x) for x in row] for row in red])
    for label, a_m, b_v in [
        ("2x+3y=8, x-y=1", [[2, 3], [1, -1]], [8, 1]),
        ("3x3", [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]], [8, -11, -3]),
        ("2 eq / 3 var", [[1, 1, 1], [1, 2, 3]], [6, 14]),
        ("1 eq / 3 var", [[1, 2, 3]], [6]),
        ("inconsistent 2x2", [[1, 1], [2, 2]], [3, 8]),
        ("overdetermined 3x2", [[1, 2], [3, 4], [5, 6]], [5, 11, 17]),
    ]:
        out = solve_exact(to_frac(a_m), b_v)
        if out["kind"] == "unique":
            print(f"solve({label}) = unique", exact(out["solution"]))
        elif out["kind"] == "infinite":
            print(f"solve({label}) = infinite particular", exact(out["particular"]),
                  "basis", [exact(v) for v in out["basis"]], "free", out["free"],
                  "rref", [[int(x) for x in row] for row in out["rref"]])
        else:
            print(f"solve({label}) = none")
    print("dot (1,2,3).(4,5,6) =", 4 + 10 + 18)
    print("dot (1,-2).(3,4) =", 3 - 8)
    print("cross (1,2,3)x(4,5,6) =", [2 * 6 - 3 * 5, 3 * 4 - 1 * 6, 1 * 5 - 2 * 4])
    print("cross x^y =", [0, 0, 1])
    print("norm (1,2,3) =", d10(math.sqrt(14)), "| sqrt14 =", math.sqrt(14))
    print("norm (3,4) =", math.sqrt(9 + 16))
    print("projection (3,4) onto (1,0) =", [3.0, 0.0])
    print("projection (1,1,1) onto (1,0,0) =", [1.0, 0.0, 0.0])
    print("angle x vs y =", d10(math.degrees(math.acos(0.0))))
    print("angle (1,1) vs (1,0) =", d10(math.degrees(math.acos(1 / math.sqrt(2)))))
    print("angle (1,0) vs (-1,0) =", d10(math.degrees(math.acos(-1.0))))
    print()


# --------------------------------------------------------------------------
# B. rank / power anchors (v0.2.0)
# --------------------------------------------------------------------------

def section_b():
    print("== B. matrix_compute v0.2.0 anchors: rank ==")
    for label, m in [
        ("[[1,2,3],[4,5,6],[7,8,9]]", [[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        ("[[1,2],[2,4]]", [[1, 2], [2, 4]]),
        ("[[0,0],[0,0]]", [[0, 0], [0, 0]]),
        ("[[1,2,3],[4,5,6]] 2x3", [[1, 2, 3], [4, 5, 6]]),
        ("identity 3x3", [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        ("[[1,2,3],[2,4,6],[1,1,1]]", [[1, 2, 3], [2, 4, 6], [1, 1, 1]]),
        ("rank-1 outer product 4x4", [[1, 2, 3, 4], [2, 4, 6, 8], [3, 6, 9, 12], [4, 8, 12, 16]]),
    ]:
        print(f"rank({label}) =", rank_exact(to_frac(m)))
    print()
    print("== B. matrix_compute v0.2.0 anchors: power ==")
    for label, m, k in [
        ("[[1,1],[0,1]]", [[1, 1], [0, 1]], 3),
        ("[[1,1],[0,1]]", [[1, 1], [0, 1]], 10),
        ("[[1,1],[0,1]]", [[1, 1], [0, 1]], 0),
        ("[[4,7],[2,6]]", [[4, 7], [2, 6]], 2),
        ("[[4,7],[2,6]]", [[4, 7], [2, 6]], -1),
        ("[[4,7],[2,6]]", [[4, 7], [2, 6]], -2),
        ("Fibonacci [[1,1],[1,0]]", [[1, 1], [1, 0]], 20),
        ("[[2,0],[0,3]]", [[2, 0], [0, 3]], 6),
    ]:
        m2 = to_frac(m)
        if k >= 0:
            out = to_frac([[1 if i == j else 0 for j in range(len(m))] for i in range(len(m))])
            for _ in range(k):
                out = mul(out, m2)
        else:
            inv = inverse_exact(m2)
            out = to_frac([[1 if i == j else 0 for j in range(len(m))] for i in range(len(m))])
            for _ in range(-k):
                out = mul(out, inv)
        print(f"power({label}, {k}) =", [exact(row) for row in out],
              "| exact:", [[str(x) for x in row] for row in out])
    print()
    print("power on a singular matrix (k<0): exact inverse is None ->",
          inverse_exact(to_frac([[1, 2], [2, 4]])) is None)
    print("power on non-square [[1,2,3]] (k=2): no product defined")
    print()


# --------------------------------------------------------------------------
# C. eigen anchors (v0.2.0)
# --------------------------------------------------------------------------

def eigen_table():
    """Anchors for matrix_eigen: (label, matrix, eigenvalues, eigenvectors or None)."""
    table = []
    # 1) 2x2 closed form: (trace ± sqrt(trace² - 4 det)) / 2
    m = [[2, 1], [1, 2]]
    tr, de = 4, 3
    root = math.sqrt(tr * tr - 4 * de)
    table.append(("[[2,1],[1,2]]", m, [(tr + root) / 2, (tr - root) / 2],
                  [[1 / math.sqrt(2), 1 / math.sqrt(2)], [1 / math.sqrt(2), -1 / math.sqrt(2)]]))
    # 2) diagonal matrix: spectrum is the diagonal, descending
    table.append(("diag(5,-2,3)", [[5, 0, 0], [0, -2, 0], [0, 0, 3]], [5, 3, -2],
                  [[1, 0, 0], [0, 0, 1], [0, 1, 0]]))
    # 3) a·I + b·(J - I): eigenvalues 6 (once) and 3 (twice) — eigenspace only
    table.append(("[[4,1,1],[1,4,1],[1,1,4]]", [[4, 1, 1], [1, 4, 1], [1, 1, 4]], [6, 3, 3], None))
    # 4) [[1,2,0],[2,1,2],[0,2,1]] -> 1, 1 ± 2*sqrt(2)
    s = math.sqrt(2)
    v1 = signed([1 / math.sqrt(2), 0.0, -1 / math.sqrt(2)])
    vp = signed([0.5, 1 / math.sqrt(2), 0.5])
    vm = signed([-0.5, 1 / math.sqrt(2), -0.5])
    table.append(("[[1,2,0],[2,1,2],[0,2,1]]", [[1, 2, 0], [2, 1, 2], [0, 2, 1]],
                  [1 + 2 * s, 1.0, 1 - 2 * s], [vp, v1, vm]))
    # 5) path graph P4: 2 - 2cos(k·pi/5), analytic eigenvectors sin(j·k·pi/5)
    table.append(("path 4x4 [[2,-1,0,0],[-1,2,-1,0],[0,-1,2,-1],[0,0,-1,2]]",
                  [[2, -1, 0, 0], [-1, 2, -1, 0], [0, -1, 2, -1], [0, 0, -1, 2]],
                  path_spectrum(4),
                  [path_eigenvector(4, 4), path_eigenvector(4, 3), path_eigenvector(4, 2), path_eigenvector(4, 1)]))
    # 6) H·diag(8,4,0,-8)·Hᵀ with H the 1/2·Hadamard matrix -> integer matrix,
    #    exact integer spectrum {8, 4, 0, -8} (verified: char poly λ⁴-4λ³-64λ²+256λ)
    had = [[F(s, 2) for s in row] for row in [[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]]]
    diag = [8, 4, 0, -8]
    a4 = [[sum(had[i][k] * diag[k] * had[j][k] for k in range(4)) for j in range(4)] for i in range(4)]
    a4int = [[int(x) for x in row] for row in a4]
    table.append(("H·diag(8,4,0,-8)·Hᵀ", a4int, [8.0, 4.0, 0.0, -8.0], None))
    # 7) 1x1
    table.append(("[[7]]", [[7]], [7.0], [[1.0]]))
    # 8) zero matrix
    table.append(("zeros 2x2", [[0, 0], [0, 0]], [0.0, 0.0], None))
    return table


def section_c():
    print("== C. matrix_eigen v0.2.0 anchors ==")
    for label, m, values, vectors in eigen_table():
        print(f"-- {label}")
        print("   eigenvalues (desc):", [d10(v) for v in values])
        if vectors is not None:
            print("   eigenvectors (unit, rows; leading non-zero component positive):",
                  [[d10(x) for x in v] for v in vectors])
        print("   trace sum:", d10(sum(values)), "| det:", d10(det_exact(to_frac(m))),
              "| product of eigenvalues:", d10(math.prod(values)) if values else None)
        if len(m) > 1:
            print("   independent Jacobi cross-check:", [d10(v) for v in jacobi_eigenvalues(m)])
    print("-- rejected inputs")
    print("   non-square [[1,2,3]] -> error (square required)")
    print("   non-symmetric [[1,2],[3,4]] -> error (asymmetry 1.0 > tolerance)")
    p5 = [[2, -1, 0, 0, 0], [-1, 2, -1, 0, 0], [0, -1, 2, -1, 0], [0, 0, -1, 2, -1], [0, 0, 0, -1, 2]]
    print("   path 5x5 spectrum:", [d10(v) for v in path_spectrum(5)], "| trace:", d10(sum(path_spectrum(5))))
    print("   path 5x5 eigenvector for the largest eigenvalue (analytic):",
          [d10(x) for x in path_eigenvector(5, 5)])
    print("   independent Jacobi cross-check:", [d10(v) for v in jacobi_eigenvalues(p5)])
    print()


def check():
    """Self-checks: the analytic anchors must agree with the independent paths."""
    ok = True
    for label, m, values, _ in eigen_table():
        jac = jacobi_eigenvalues(m)
        if len(m) == 1:
            continue
        worst = max(abs(a - b) for a, b in zip(values, jac))
        status = "OK" if worst < 1e-10 else "FAIL"
        ok = ok and worst < 1e-10
        print(f"[check] {label}: analytic vs Jacobi max diff {worst:.3e} {status}")
        if len(m) == 4 and label.startswith("H·"):
            cp = [1, -4, -64, 256, 0]
            sym = [sum(cp[i] * v ** (4 - i) for i in range(5)) for v in values]
            good = all(abs(x) < 1e-6 for x in sym)
            ok = ok and good
            print(f"[check] {label}: matches char poly λ⁴-4λ³-64λ²+256λ -> {good}")
    # legacy exactness: exact Fraction det/inverse agree with analytic values
    assert det_exact(to_frac([[1, 2], [3, 4]])) == -2
    assert det_exact(to_frac([[2, 1, 1], [1, 3, 2], [1, 0, 0]])) == -1
    assert det_exact(to_frac([[0, 1, 0, 3], [2, 0, 0, 1], [0, 0, 1, 0], [0, 2, 0, 0]])) == 12
    assert inverse_exact(to_frac([[4, 7], [2, 6]])) == [[F(3, 5), F(-7, 10)], [F(-1, 5), F(2, 5)]]
    assert rank_exact(to_frac([[1, 2, 3], [4, 5, 6], [7, 8, 9]])) == 2
    assert solve_exact(to_frac([[2, 3], [1, -1]]), [8, 1])["solution"] == [F(11, 5), F(6, 5)]
    print("[check] exact legacy anchors hold (det/inverse/rank/solve in Fractions) OK")
    print("ALL CHECKS PASSED" if ok else "CHECKS FAILED")
    return 0 if ok else 1


def main():
    section_a()
    section_b()
    section_c()
    if "--check" in sys.argv:
        return check()
    return 0


if __name__ == "__main__":
    sys.exit(main())
