#!/usr/bin/env python3
"""Interval-certified internal roots and direct coalition checks for Paper 2.

Python 3.10+; sole third-party requirement: mpmath==1.3.0.
The imported ORL verifier is an unchanged copy of Paper 1's fresh verifier.
Interval proof inputs are reconstructed from exact rationals. Point residuals
and coalition formula discrepancies are diagnostics, never proof inputs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction as F
from math import factorial
from pathlib import Path
import platform
import mpmath as mp
import orl_base_verifier as base

ETA = F(1, 10000)
GUESSES = (("R1", ("0.3873277033", "0.4171080596")),
           ("R2", ("0.5384740734", "0.1469485836")),
           ("R3", ("0.5548420673", "0.1196628780")))


def delta(n, ctx):
    return base.q(ctx, ETA)*ctx.sqrt(base.q(ctx, F(3, n)))


def coupled_evaluate(lam, n, ctx):
    residual, jacobian, phi, grad = base.smooth_evaluate(lam, ctx)
    d = delta(n, ctx)
    residual = [residual[i]+d*(base.q(ctx, F(1, 3))-lam[i])
                for i in range(2)]
    jacobian = [[jacobian[i][j]-(d if i == j else ctx.mpf(0))
                 for j in range(2)] for i in range(2)]
    return residual, jacobian, phi, [v+d for v in grad]


def certify(n, label, guess):
    """Prove a unique regular internal root in a rational two-coordinate box."""
    def system(a, b):
        return tuple(coupled_evaluate([a, b, 1-a-b], n, mp.mp)[0])
    root = mp.findroot(system, tuple(mp.mpf(v) for v in guess),
                       tol=mp.mpf("1e-90"), maxsteps=100)
    strings = [mp.nstr(v, 90) for v in root]
    center = [F(v) for v in strings]
    center.append(1-center[0]-center[1])
    box = [(center[i]-base.RADIUS, center[i]+base.RADIUS) for i in range(2)]
    xiv = [mp.iv.mpf([base.q(mp.iv, lo).a, base.q(mp.iv, hi).b])
           for lo, hi in box]
    xiv.append(mp.iv.mpf(1)-xiv[0]-xiv[1])
    _, jx, _, payoff_box = coupled_evaluate(xiv, n, mp.iv)
    point = [base.q(mp.mp, v) for v in center]
    gp, jp, _, xp = coupled_evaluate(point, n, mp.mp)
    cinverse = mp.inverse(mp.matrix(jp))
    cstrings = [[mp.nstr(cinverse[i, j], 90) for j in range(2)] for i in range(2)]
    c = [[base.q(mp.iv, F(v)) for v in row] for row in cstrings]
    cdet = base.det2(c)
    assert base.excludes_zero(cdet)
    civ = [base.q(mp.iv, v) for v in center]
    g0, _, _, _ = coupled_evaluate(civ, n, mp.iv)
    m = [[mp.iv.mpf(int(i == j))-sum(c[i][k]*jx[k][j] for k in range(2))
          for j in range(2)] for i in range(2)]
    row_bounds = [sum(max(abs(lo), abs(hi)) for lo, hi in
                     (base.endpoints(v) for v in row)) for row in m]
    contraction = max(row_bounds)
    dx = [mp.iv.mpf([base.q(mp.iv, -base.RADIUS).a,
                     base.q(mp.iv, base.RADIUS).b])]*2
    image = [civ[i]-sum(c[i][j]*g0[j] for j in range(2))
             +sum(m[i][j]*dx[j] for j in range(2)) for i in range(2)]
    strict = all(box[i][0] < base.endpoints(image[i])[0]
                 and base.endpoints(image[i])[1] < box[i][1] for i in range(2))
    jdet = base.det2(jx)
    interior = all(base.endpoints(v)[0] > base.MU for v in xiv)
    assert contraction < 1, (n, label, "Krawczyk contraction")
    assert strict, (n, label, "Krawczyk inclusion")
    assert base.excludes_zero(jdet), (n, label, "regularity")
    assert interior, (n, label, "comparison-cone interior")
    lo, hi = base.endpoints(jdet)
    slacks = [min(base.endpoints(image[i])[0]-box[i][0],
                  box[i][1]-base.endpoints(image[i])[1]) for i in range(2)]
    result = {
        "players": n, "root": label,
        "eta_exact": str(ETA), "delta_expression": f"(1/10000)*sqrt(3/{n})",
        "delta_interval": base.bounds(delta(n, mp.iv)),
        "rho_centers": strings+[mp.nstr(base.q(mp.mp, center[2]), 90)],
        "rho3_center_exact": str(center[2]), "radius_exact": str(base.RADIUS),
        "rho_box": [base.bounds(v) for v in xiv],
        "G_center_interval": [base.bounds(v) for v in g0],
        "C_decimal_entries": cstrings, "C_determinant_interval": base.bounds(cdet),
        "DG_box": [[base.bounds(v) for v in row] for row in jx],
        "DG_determinant_interval": base.bounds(jdet),
        "determinant_sign": 1 if lo > 0 else -1,
        "krawczyk_image": [base.bounds(v) for v in image],
        "krawczyk_contraction_bound_upper": base.outward_decimal(contraction, False),
        "strict_inclusion_slack_lower": [base.outward_decimal(v, True) for v in slacks],
        "coupled_payoff_enclosure": [base.bounds(v) for v in payoff_box],
        "coupled_payoff_point_diagnostic": [mp.nstr(v, 75) for v in xp],
        "point_residual_diagnostic": mp.nstr(max(abs(v) for v in gp), 75),
        "strict_krawczyk_inclusion": strict, "krawczyk_contraction": True,
        "interval_jacobian_nonsingular": True, "interval_C_nonsingular": True,
        "box_in_Lambda_mu_interior": interior,
    }
    return result, payoff_box


def pairwise_separation(n, certificates, boxes):
    result = []
    for i in range(3):
        for j in range(i+1, 3):
            separated = []
            for coordinate in range(3):
                li, ui = base.endpoints(boxes[i][coordinate])
                lj, uj = base.endpoints(boxes[j][coordinate])
                if ui < lj or uj < li:
                    gap = lj-ui if ui < lj else li-uj
                    separated.append({"coordinate": coordinate+1,
                                      "gap_lower": base.outward_decimal(gap, True)})
            assert separated, (n, i, j, "payoff separation")
            result.append({"players": n,
                           "roots": [certificates[i]["root"], certificates[j]["root"]],
                           "separated_coordinates": separated, "passed": True})
    return result


def block_sizes(n):
    return [3]*(n//3)+([n % 3] if n % 3 else [])


def scales(sizes):
    den = sum(mp.sqrt(m) for m in sizes)
    return [mp.sqrt(m)/den for m in sizes]


def scale_derivative_checks(n):
    """Numerical finite-precision checks of the manuscript's analytic formula."""
    sizes = block_sizes(n)
    ell = len(sizes)
    r = scales(sizes)
    ep = base.q(mp.mp, ETA)
    qnorm = mp.sqrt(sum(v*v for v in r))
    a = [mp.mpf(m)/n for m in sizes]
    lvals = [ep*(a[b]*qnorm-r[b]*r[b]/qnorm) for b in range(ell)]
    # Ambient derivative formula from the manuscript. Restrict by dr_last=-sum dr.
    ambient = [[2*ep*mp.sqrt(a[b])*(mp.sqrt(a[b])*mp.sqrt(a[c])-int(b == c))
                for c in range(ell)] for b in range(ell)]
    reduced = [[ambient[b][c]-ambient[b][-1] for c in range(ell-1)]
               for b in range(ell-1)]
    def system(*coords):
        rr = [*coords, 1-sum(coords)]
        qq = mp.sqrt(sum(v*v for v in rr))
        return [ep*(a[b]*qq-rr[b]*rr[b]/qq) for b in range(ell-1)]
    if ell > 1:
        numerical = mp.matrix([[mp.diff(lambda t: system(*[
            t if j == c else r[j] for j in range(ell-1)])[b], r[c])
            for c in range(ell-1)] for b in range(ell-1)])
        discrepancy = max(abs(numerical[b, c]-reduced[b][c])
                          for b in range(ell-1) for c in range(ell-1))
        determinant = mp.det(mp.matrix(reduced))
        assert abs(determinant) > mp.mpf("1e-30")
    else:
        discrepancy, determinant = mp.mpf(0), mp.mpf(1)
    assert discrepancy < mp.mpf("1e-90")
    return {"players": n, "block_sizes": sizes,
            "block_masses": [mp.nstr(v, 75) for v in r],
            "scale_residual_max_diagnostic": mp.nstr(max(abs(v) for v in lvals), 75),
            "reduced_scale_determinant_diagnostic": mp.nstr(determinant, 75),
            "derivative_formula_error_diagnostic": mp.nstr(discrepancy, 75),
            "zero_dimensional_scale_convention": ell == 1, "passed": True}


def block_support(lam, mask):
    """Evaluate the uncoupled base support of a block coalition."""
    m = len(lam)
    selected = [i for i in range(m) if mask & (1 << i)]
    if not selected:
        return mp.mpf(0)
    if m == 1:
        return lam[0]
    if m == 2:
        return mp.sqrt(sum(v*v for v in lam)) if len(selected) == 2 else mp.mpf(0)
    if len(selected) == 1:
        return mp.mpf(0)
    fs = []
    for pair, coeff in base.PAIRS:
        i, j = pair
        fs.append(base.q(mp.mp, coeff[0])*lam[i]
                  +base.q(mp.mp, coeff[1])*lam[j]
                  +base.q(mp.mp, base.EPS)*mp.sqrt(lam[i]**2+lam[j]**2))
    if len(selected) == 2:
        return fs[[pair for pair, _ in base.PAIRS].index(tuple(selected))]
    for vertex in (base.XA, base.XB):
        fs.append(sum(base.q(mp.mp, vertex[i])*lam[i] for i in range(3)))
    return sum(f**base.P for f in fs)**base.q(mp.mp, F(1, base.P))


def direct_coalition_check(n, certificate_by_n):
    """Independent finite-coalition Shapley sum compared to scale-shape formula."""
    sizes = block_sizes(n)
    r = scales(sizes)
    rho, gradients, residuals = [], [], []
    for b, m in enumerate(sizes):
        if m == 3:
            cert = certificate_by_n[n][b % 3]
            rr = [mp.mpf(v) for v in cert["rho_centers"]]
            gg, _, _, xx = base.smooth_evaluate(rr, mp.mp)
            gg.append(-sum(gg))
        elif m == 2:
            rr = [mp.mpf("0.5")]*2
            xx = [1/mp.sqrt(2)]*2
            gg = [mp.mpf(0)]*2
        else:
            rr, xx, gg = [mp.mpf(1)], [mp.mpf(1)], [mp.mpf(0)]
        rho.append(rr); gradients.append(xx); residuals.append(gg)
    lambdas = [[r[b]*v for v in rho[b]] for b in range(len(sizes))]
    ep = base.q(mp.mp, ETA)
    qnorm = mp.sqrt(sum(v*v for v in r))
    fullmask = (1 << n)-1
    values = []
    for mask in range(1 << n):
        value, offset = mp.mpf(0), 0
        for b, m in enumerate(sizes):
            value += block_support(lambdas[b], (mask >> offset) & ((1 << m)-1))
            offset += m
        if mask == fullmask:
            value += ep*qnorm
        values.append(value)
    phi = []
    for i in range(n):
        total = mp.mpf(0)
        for mask in range(1 << n):
            if mask & (1 << i):
                continue
            count = mask.bit_count()
            weight = base.q(mp.mp, F(factorial(count)*factorial(n-count-1), factorial(n)))
            total += weight*(values[mask | (1 << i)]-values[mask])
        phi.append(total)
    direct, formula, offsets = [], [], []
    offset = 0
    for b, m in enumerate(sizes):
        for i in range(m):
            direct.append(phi[offset+i]-lambdas[b][i]*(gradients[b][i]+ep*r[b]/qnorm))
            formula.append(r[b]*residuals[b][i]+ep*(qnorm/n-r[b]*rho[b][i]*r[b]/qnorm))
        offsets.append(offset)
        offset += m
    error = max(abs(x-y) for x, y in zip(direct, formula))
    root_error = max(abs(v) for v in direct)
    assert error < mp.mpf("1e-80"), (n, "direct coalition formula discrepancy")
    assert root_error < mp.mpf("1e-80"), (n, "direct root residual")
    return {"players": n, "coalitions_evaluated": 1 << n,
            "block_sizes": sizes, "selected_three_block_branches":
                [f"R{b % 3+1}" for b, m in enumerate(sizes) if m == 3],
            "direct_residual_vector_diagnostic": [mp.nstr(v, 75) for v in direct],
            "direct_residual_max_diagnostic": mp.nstr(root_error, 75),
            "direct_vs_scale_shape_error_diagnostic": mp.nstr(error, 75),
            "passed": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[1]/"results")
    args = parser.parse_args()
    if mp.__version__ != "1.3.0":
        raise RuntimeError("This replication pins mpmath==1.3.0")
    args.output.mkdir(parents=True, exist_ok=True)
    mp.mp.dps, mp.iv.dps = 110, 80
    started = datetime.now(timezone.utc).isoformat()
    certificates, separation, by_n = [], [], {}
    log = ["Paper 2 Phase 3C weak-coupling replication", "UTC run start: "+started,
           "Python: "+platform.python_version()+"; mpmath: "+mp.__version__,
           "Point dps: 110; interval dps: 80; eta=1/10000; epsilon=1/20; p=120"]
    for n in range(3, 13):
        certs, boxes = [], []
        for label, guess in GUESSES:
            cert, box = certify(n, label, guess)
            certs.append(cert); boxes.append(box)
        certificates.extend(certs)
        by_n[n] = certs
        separation.extend(pairwise_separation(n, certs, boxes))
        line = f"PASS n={n}: 3 internal regular roots certified; all 3 pairwise payoff separations."
        print(line, flush=True); log.append(line)
    base.write_json(args.output/"coupled_internal_interval_certificates.json", {
        "parameters_exact": {"eta": str(ETA), "epsilon": str(base.EPS),
                             "p": base.P, "radius": str(base.RADIUS)},
        "point_dps": mp.mp.dps, "interval_dps": mp.iv.dps,
        "certificates": certificates, "pairwise_payoff_distinctions": separation,
        "scope": "Internal three-player shape systems only; global regularity additionally uses analytic scale theorem."})
    flat = []
    for cert in certificates:
        row = {"players": cert["players"], "root": cert["root"],
               "rho1_center": cert["rho_centers"][0], "rho2_center": cert["rho_centers"][1],
               "rho3_center": cert["rho_centers"][2],
               "DG_determinant_lower": cert["DG_determinant_interval"][0],
               "DG_determinant_upper": cert["DG_determinant_interval"][1],
               "determinant_sign": cert["determinant_sign"],
               "krawczyk_contraction_bound_upper": cert["krawczyk_contraction_bound_upper"],
               "strict_krawczyk_inclusion": True, "interval_jacobian_nonsingular": True}
        flat.append(row)
    base.write_csv(args.output/"coupled_internal_interval_certificates.csv", flat)
    scale_checks = [scale_derivative_checks(n) for n in range(3, 13)]
    base.write_json(args.output/"scale_derivative_diagnostics.json", scale_checks)
    coalition_checks = [direct_coalition_check(n, by_n) for n in range(3, 9)]
    base.write_json(args.output/"direct_coalition_diagnostics.json", coalition_checks)
    bounds = [{"players": n, "three_player_blocks": n//3,
               "certified_regular_value_lower_bound": 3**(n//3),
               "fixed_eta": str(ETA),
               "provenance": "interval-certified coupled internal roots + analytic nonsingular scale/anchor and payoff-injectivity proofs",
               "all_roots_enumerated": False} for n in range(3, 13)]
    base.write_csv(args.output/"finite_n_coupled_lower_bounds.csv", bounds)
    log.extend(["PASS direct coalition-sum residual checks for n=3,...,8.",
                "PASS scale derivative formula diagnostics for n=3,...,12.",
                "Finite-n certificates do not prove this fixed eta works for every n.",
                "No global enumeration, degree computation, boundary exclusion, or input-size complexity claim.",
                "REPLICATION STATUS: PASS"])
    status = {"run_started_utc": started, "run_finished_utc": datetime.now(timezone.utc).isoformat(),
              "python": platform.python_version(), "mpmath": mp.__version__,
              "status": "PASS", "checks_passed": {
                  "30_coupled_internal_interval_certificates": True,
                  "30_pairwise_payoff_separations": True,
                  "10_scale_derivative_diagnostics": True,
                  "6_direct_coalition_diagnostics": True},
              "limitations": ["global regularity uses analytic scale and recombination arguments",
                              "no exhaustive global root search", "fixed eta certificate is finite-n only"]}
    base.write_json(args.output/"coupling_replication_status.json", status)
    (args.output/"coupling_replication_run.log").write_text("\n".join(log)+"\n", encoding="utf-8")
    print("REPLICATION STATUS: PASS")
    print("Results directory:", args.output.resolve())


if __name__ == "__main__":
    main()
