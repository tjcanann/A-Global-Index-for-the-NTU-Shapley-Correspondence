#!/usr/bin/env python3
"""Exact arithmetic and interval certificates for NTU Shapley multiplicity.

Requires Python 3.10+ and mpmath 1.3.0. No NumPy, SymPy, or Pandas.
Proof comparisons use exact endpoint fractions; CSV/JSON intervals use
outward-rounded decimal endpoint strings rather than binary floats.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
from datetime import datetime, timezone
from decimal import Decimal, localcontext, ROUND_FLOOR, ROUND_CEILING
from fractions import Fraction as F
from pathlib import Path

import mpmath as mp

TITLE = "Robust and Exponential Multiplicity of the NTU Shapley Correspondence"
PAIRS = (((0, 1), (F(1, 9), F(77, 36))),
         ((0, 2), (F(7, 30), F(73, 30))),
         ((1, 2), (F(41, 20), F(1, 10))))
XA = (F(3, 4), F(43, 40), F(27, 20))
XB = (F(43, 52), F(1), F(41, 32))
LA = (F(2, 5), F(2, 5), F(1, 5))
LB = (F(13, 25), F(4, 25), F(8, 25))
EPS = F(1, 20)
MU = F(1, 20)
RADIUS = F(1, 100_000_000)
P = 120


def q(ctx, value):
    """Convert rationals independently in the chosen arithmetic context."""
    value = F(value)
    return ctx.mpf(value.numerator) / ctx.mpf(value.denominator)


def dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def det2(matrix):
    return matrix[0][0]*matrix[1][1]-matrix[0][1]*matrix[1][0]


def shapley3(h12, h13, h23, hn):
    return (h12/6+h13/6+(hn-h23)/3,
            h12/6+h23/6+(hn-h13)/3,
            h13/6+h23/6+(hn-h12)/3)


def exact_checks():
    embedded = []
    for pair, coeff in PAIRS:
        vertex = [F(0)]*3
        for i, c in zip(pair, coeff):
            vertex[i] = c
        embedded.append(tuple(vertex))
    vertices = [(F(0),)*3, *embedded, XA, XB]
    output = []
    for label, lam, payoff in (("A", LA, XA), ("B", LB, XB)):
        assert sum(lam) == 1 and min(lam) > 0
        pair_supports = [sum(lam[i]*c for i, c in zip(pair, coeff))
                         for pair, coeff in PAIRS]
        supports = [dot(lam, v) for v in vertices]
        hn = dot(lam, payoff)
        phi = shapley3(*pair_supports, hn)
        weighted = tuple(lam[i]*payoff[i] for i in range(3))
        assert phi == weighted
        gaps = [hn-s for v, s in zip(vertices, supports) if v != payoff]
        assert min(gaps) > 0
        # Derive DG from the three linear pair supports and active grand vertex.
        pair_grads = []
        for pair, coeff in PAIRS:
            row = [F(0)]*3
            for i, c in zip(pair, coeff):
                row[i] = c
            pair_grads.append(row)
        phi_grad = [list(v) for v in zip(*[
            shapley3(pair_grads[0][j], pair_grads[1][j],
                     pair_grads[2][j], payoff[j]) for j in range(3)])]
        ambient = [[phi_grad[i][j]-(payoff[i] if i == j else 0)
                    for j in range(3)] for i in range(2)]
        reduced = [[ambient[i][j]-ambient[i][2] for j in range(2)]
                   for i in range(2)]
        determinant = det2(reduced)
        assert determinant != 0
        expected = F(71, 6480) if label == "A" else -F(14279, 269568)
        assert determinant == expected
        output.append({
            "root": label, "lambda": [str(v) for v in lam],
            "payoff": [str(v) for v in payoff],
            "pair_supports": [str(v) for v in pair_supports],
            "grand_support": str(hn),
            "weighted_shapley": [str(v) for v in phi],
            "residual": [str(phi[i]-weighted[i]) for i in range(3)],
            "grand_support_gaps": [str(v) for v in gaps],
            "minimum_support_gap": str(min(gaps)),
            "reduced_jacobian": [[str(v) for v in row] for row in reduced],
            "reduced_jacobian_determinant": str(determinant),
            "passed": True,
        })
    assert XA != XB
    return output


def smooth_evaluate(lam, ctx):
    """G, reduced DG, TU Shapley vector, grand-support gradient.

    All rational model constants are rebuilt inside ctx. Thus interval
    evaluations never inherit rounded point coefficients or residuals.
    """
    zero, one, ep = ctx.mpf(0), ctx.mpf(1), q(ctx, EPS)
    fs, gs, hs = [], [], []
    for pair, coeff in PAIRS:
        i, j = pair
        cc = [q(ctx, c) for c in coeff]
        r = ctx.sqrt(lam[i]**2+lam[j]**2)
        f = cc[0]*lam[i]+cc[1]*lam[j]+ep*r
        g = [zero]*3
        g[i], g[j] = cc[0]+ep*lam[i]/r, cc[1]+ep*lam[j]/r
        h = [[zero for _ in range(3)] for _ in range(3)]
        for a in pair:
            for b in pair:
                h[a][b] = ep*((one if a == b else zero)/r-lam[a]*lam[b]/r**3)
        fs.append(f); gs.append(g); hs.append(h)
    for rational_vec in (XA, XB):
        vec = [q(ctx, v) for v in rational_vec]
        fs.append(dot(vec, lam)); gs.append(vec)
        hs.append([[zero for _ in range(3)] for _ in range(3)])
    hn = sum(f**P for f in fs)**q(ctx, F(1, P))
    b = [sum(fs[a]**(P-1)*gs[a][i] for a in range(5)) for i in range(3)]
    grad = [hn**(1-P)*b[i] for i in range(3)]
    hess = [[zero for _ in range(3)] for _ in range(3)]
    for i in range(3):
        for j in range(3):
            val = sum((P-1)*fs[a]**(P-2)*gs[a][i]*gs[a][j]
                      +fs[a]**(P-1)*hs[a][i][j] for a in range(5))
            hess[i][j] = hn**(1-P)*val-(P-1)*hn**(1-2*P)*b[i]*b[j]
    phi = shapley3(*fs[:3], hn)
    phi_grad = [list(v) for v in zip(*[
        shapley3(gs[0][j], gs[1][j], gs[2][j], grad[j])
        for j in range(3)])]
    residual = [phi[i]-lam[i]*grad[i] for i in range(2)]
    ambient = [[phi_grad[i][j]-(grad[i] if i == j else zero)
                -lam[i]*hess[i][j] for j in range(3)] for i in range(2)]
    reduced = [[ambient[i][j]-ambient[i][2] for j in range(2)] for i in range(2)]
    return residual, reduced, phi, grad


def binary_fraction(t):
    """Exact rational value of an mpmath finite binary endpoint."""
    sign, mantissa, exponent, _ = t
    if exponent >= 0:
        value = F(mantissa*(1 << exponent))
    else:
        value = F(mantissa, 1 << -exponent)
    return -value if sign else value


def endpoints(interval):
    return tuple(binary_fraction(t) for t in interval._mpi_)


def outward_decimal(value, lower):
    with localcontext() as ctx:
        ctx.prec = 100
        ctx.rounding = ROUND_FLOOR if lower else ROUND_CEILING
        return str(Decimal(value.numerator)/Decimal(value.denominator))


def bounds(interval):
    lo, hi = endpoints(interval)
    return [outward_decimal(lo, True), outward_decimal(hi, False)]


def excludes_zero(interval):
    lo, hi = endpoints(interval)
    return hi < 0 or lo > 0


def certify(label, guess):
    def point_system(a, b):
        return tuple(smooth_evaluate([a, b, 1-a-b], mp.mp)[0])
    root = mp.findroot(point_system, tuple(mp.mpf(v) for v in guess),
                       tol=mp.mpf("1e-90"), maxsteps=100)
    # Decimal centers are fixed rational inputs to the proof, independently
    # reconstructed in interval arithmetic (not singleton approximate mpfs).
    center_strings = [mp.nstr(v, 90) for v in root]
    center = [F(v) for v in center_strings]
    center.append(1-center[0]-center[1])
    # Each endpoint is a rational interval enclosure. Their union gives X.
    box = [(center[i]-RADIUS, center[i]+RADIUS) for i in range(2)]
    xiv = [mp.iv.mpf([q(mp.iv, lo).a, q(mp.iv, hi).b]) for lo, hi in box]
    xiv.append(mp.iv.mpf(1)-xiv[0]-xiv[1])
    _, jx, _, payoff_box = smooth_evaluate(xiv, mp.iv)
    point_lam = [q(mp.mp, v) for v in center]
    g_point, j_point, _, payoff_point = smooth_evaluate(point_lam, mp.mp)
    inverse = mp.inverse(mp.matrix(j_point))
    c_strings = [[mp.nstr(inverse[i, j], 90) for j in range(2)] for i in range(2)]
    c_rational = [[F(v) for v in row] for row in c_strings]
    c = [[q(mp.iv, v) for v in row] for row in c_rational]
    cdet = det2(c)
    assert excludes_zero(cdet), f"{label}: C interval determinant contains zero"
    # Crucial: G(center) is evaluated completely in interval arithmetic.
    # A small point residual is a diagnostic, never a proof input.
    center_iv = [q(mp.iv, v) for v in center]
    g0, _, _, _ = smooth_evaluate(center_iv, mp.iv)
    m = [[mp.iv.mpf(int(i == j))-sum(c[i][k]*jx[k][j] for k in range(2))
          for j in range(2)] for i in range(2)]
    contraction_rows = [sum(max(abs(lo), abs(hi)) for lo, hi in
                            (endpoints(entry) for entry in row)) for row in m]
    contraction_bound = max(contraction_rows)
    assert contraction_bound < 1, f"{label}: Krawczyk contraction bound >= 1"
    dx = [mp.iv.mpf([q(mp.iv, -RADIUS).a, q(mp.iv, RADIUS).b])]*2
    k = [center_iv[i]-sum(c[i][j]*g0[j] for j in range(2))
         +sum(m[i][j]*dx[j] for j in range(2)) for i in range(2)]
    strict = all(box[i][0] < endpoints(k[i])[0]
                 and endpoints(k[i])[1] < box[i][1] for i in range(2))
    jdet = det2(jx)
    positive = all(endpoints(v)[0] > MU for v in xiv)
    assert strict, f"{label}: strict Krawczyk inclusion failed"
    assert excludes_zero(jdet), f"{label}: interval DG determinant contains zero"
    assert positive, f"{label}: box outside positive cone interior"
    slacks = [min(endpoints(k[i])[0]-box[i][0], box[i][1]-endpoints(k[i])[1])
              for i in range(2)]
    result = {
        "root": label,
        "lambda_centers": center_strings+[mp.nstr(q(mp.mp, center[2]), 90)],
        "lambda3_center_exact": str(center[2]),
        "radius_exact": str(RADIUS),
        "lambda_box": [bounds(v) for v in xiv],
        "G_center_interval": [bounds(v) for v in g0],
        "C_decimal_entries": c_strings,
        "C_determinant_interval": bounds(cdet),
        "DG_box": [[bounds(v) for v in row] for row in jx],
        "DG_determinant_interval": bounds(jdet),
        "krawczyk_image": [bounds(v) for v in k],
        "krawczyk_contraction_bound_upper": outward_decimal(contraction_bound, False),
        "strict_inclusion_slack_lower": [outward_decimal(v, True) for v in slacks],
        "payoff_enclosure": [bounds(v) for v in payoff_box],
        "payoff_point_diagnostic": [mp.nstr(v, 75) for v in payoff_point],
        "point_residual_diagnostic": mp.nstr(max(abs(v) for v in g_point), 75),
        "strict_krawczyk_inclusion": strict,
        "krawczyk_contraction": contraction_bound < 1,
        "interval_jacobian_nonsingular": excludes_zero(jdet),
        "interval_C_nonsingular": excludes_zero(cdet),
        "box_in_Lambda_mu_interior": positive,
    }
    return result, payoff_box


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[1]/"results")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    mp.mp.dps, mp.iv.dps = 110, 80
    started = datetime.now(timezone.utc).isoformat()
    log = [TITLE, "UTC run start: "+started,
           "Python: "+platform.python_version(), "mpmath: "+mp.__version__,
           "Point dps: 110; interval dps: 80",
           "Exact rational radius: 1/100000000; epsilon=mu=1/20; p=120"]
    exact = exact_checks()
    write_json(args.output/"exact_rational_verification.json", exact)
    exact_csv = [{"root": row["root"],
                  **{"lambda"+str(i+1): row["lambda"][i] for i in range(3)},
                  **{"payoff"+str(i+1): row["payoff"][i] for i in range(3)},
                  "minimum_support_gap": row["minimum_support_gap"],
                  "det_reduced_jacobian": row["reduced_jacobian_determinant"],
                  "all_three_residuals_exactly_zero": True,
                  "passed": True} for row in exact]
    write_csv(args.output/"exact_rational_verification.csv", exact_csv)
    log.append("PASS exact rational construction: 2 distinct allocations, 6 zero residuals,")
    log.append("strict grand support gaps, and independently derived nonzero determinants.")
    certs, payoff_boxes = [], []
    for label, guess in (("R1", ("0.3873277033", "0.4171080596")),
                         ("R2", ("0.5384740734", "0.1469485836")),
                         ("R3", ("0.5548420673", "0.1196628780"))):
        cert, payoff_box = certify(label, guess)
        certs.append(cert); payoff_boxes.append(payoff_box)
        log.append("PASS "+label+": strict interval Krawczyk inclusion and contraction; nonsingular DG and C;")
        log.append("all comparison weights > 1/20. Center: "+", ".join(cert["lambda_centers"]))
    separations = []
    for i in range(3):
        for j in range(i+1, 3):
            separated = []
            for a in range(3):
                li, ui = endpoints(payoff_boxes[i][a])
                lj, uj = endpoints(payoff_boxes[j][a])
                if ui < lj or uj < li:
                    gap = lj-ui if ui < lj else li-uj
                    separated.append({"coordinate": a+1,
                                      "gap_lower": outward_decimal(gap, True)})
            assert separated, f"payoff enclosures for R{i+1},R{j+1} overlap in every coordinate"
            separations.append({"roots": [f"R{i+1}", f"R{j+1}"],
                                "separated_coordinates": separated, "passed": True})
    log.append("PASS all 3 pairwise allocation distinctions by disjoint payoff enclosures.")
    write_json(args.output/"smooth_interval_certificates.json",
               {"title": TITLE, "parameters_exact": {"epsilon": str(EPS), "mu": str(MU),
                                                     "p": P, "radius": str(RADIUS)},
                "point_dps": mp.mp.dps, "interval_dps": mp.iv.dps,
                "certificates": certs, "payoff_distinctions": separations})
    rows = []
    for cert in certs:
        row = {"root": cert["root"]}
        for i in range(3):
            row[f"lambda{i+1}_center"] = cert["lambda_centers"][i]
            row[f"lambda{i+1}_box_lower"], row[f"lambda{i+1}_box_upper"] = cert["lambda_box"][i]
            row[f"payoff{i+1}_lower"], row[f"payoff{i+1}_upper"] = cert["payoff_enclosure"][i]
        for i in range(2):
            row[f"K{i+1}_lower"], row[f"K{i+1}_upper"] = cert["krawczyk_image"][i]
        row["DG_determinant_lower"], row["DG_determinant_upper"] = cert["DG_determinant_interval"]
        row["krawczyk_contraction_bound_upper"] = cert["krawczyk_contraction_bound_upper"]
        for key in ("strict_krawczyk_inclusion", "krawczyk_contraction", "interval_jacobian_nonsingular",
                    "interval_C_nonsingular", "box_in_Lambda_mu_interior"):
            row[key] = cert[key]
        rows.append(row)
    write_csv(args.output/"smooth_interval_certificates.csv", rows)
    lower_bounds = [{"players": n, "three_player_blocks": n//3,
                     "analytic_allocation_lower_bound": 2**(n//3),
                     "certified_smooth_allocation_lower_bound": 3**(n//3),
                     "analytic_provenance": "exact rational two-allocation block; independent-block theorem",
                     "smooth_provenance": "three interval-certified distinct allocations at epsilon=1/20,p=120; independent-block theorem",
                     "interpretation": "allocation-count lower bounds, not all-chart enumeration or an input-size complexity theorem"}
                    for n in range(3, 31)]
    write_csv(args.output/"independent_block_multiplicity_bounds.csv", lower_bounds)
    log.append("PASS exact integer lower-bound table for n=3,...,30 using independent blocks.")
    status = {"title": TITLE, "run_started_utc": started,
              "run_finished_utc": datetime.now(timezone.utc).isoformat(),
              "python": platform.python_version(), "mpmath": mp.__version__,
              "checks_passed": {"exact_construction": True,
                                "three_smooth_interval_certificates": True,
                                "all_pairwise_payoff_separations": True,
                                "independent_block_bound_table": True},
              "status": "PASS", "scope": "focused replication of the letter's multiplicity results"}
    write_json(args.output/"replication_status.json", status)
    log.append("REPLICATION STATUS: PASS")
    text = "\n".join(log)+"\n"
    (args.output/"replication_run.log").write_text(text, encoding="utf-8")
    print(text, end="")
    print("Results directory:", args.output.resolve())


if __name__ == "__main__":
    main()
