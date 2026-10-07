"""Exact symbolic checks for example_anchor_collision.tex.

No numerical root search, interval certification, or global enumeration
is used by this verifier. The global arguments remain in the LaTeX proof.
"""

import sympy as sp


def simplify_exact(value):
    return sp.simplify(sp.sqrtdenest(sp.simplify(value)))


def check_zero(label, value):
    values = list(value) if isinstance(value, sp.MatrixBase) else [value]
    simplified = [simplify_exact(item) for item in values]
    assert all(item == 0 for item in simplified), (label, simplified)
    print(f"PASS {label}: {simplified}")


print("Exact anchor and payoff-collision verifier")
print(f"SymPy {sp.__version__}")

# Four-player Euclidean block plus positive one-player anchor.
u, v, w = sp.symbols("u v w", positive=True)
eta = sp.symbols("eta", positive=True)
block_weights = [u, v, w]
r = u + v + w
anchor = 1 - r
q = sp.sqrt(u*u + v*v + w*w)
Q = sp.sqrt(r*r + anchor*anchor)
g = sp.Matrix([q/3 - li*li/q + eta*(Q/4-li*r/Q)
               for li in block_weights])
g4 = eta*(Q/4-anchor*anchor/Q)
root_entry = (3-sp.sqrt(3))/6
root = {u: root_entry, v: root_entry, w: root_entry}
check_zero("n4 first-three residuals", g.subs(root))
check_zero("n4 anchor residual", g4.subs(root))
check_zero("n4 efficiency identity", sum(g)+g4)
check_zero("n4 block mass", r.subs(root)-(3-sp.sqrt(3))/2)
check_zero("n4 anchor mass", anchor.subs(root)-(sp.sqrt(3)-1)/2)
check_zero("n4 Euclidean block-mass norm", Q.subs(root)-(sp.sqrt(3)-1))

H = q+anchor+eta*Q
# Differentiate the original homogeneous grand support, before the chart.
x1,x2,x3,x4 = sp.symbols("x1 x2 x3 x4", positive=True)
rr=x1+x2+x3
HH=sp.sqrt(x1*x1+x2*x2+x3*x3)+x4+eta*sp.sqrt(rr*rr+x4*x4)
lambda_root={x1:root_entry,x2:root_entry,x3:root_entry,x4:(sp.sqrt(3)-1)/2}
payoff=sp.Matrix([sp.diff(HH,li).subs(lambda_root) for li in [x1,x2,x3,x4]])
expected_payoff=sp.Matrix([1/sp.sqrt(3)+eta*sp.sqrt(3)/2]*3+[1+eta/2])
check_zero("n4 payoff formula",payoff-expected_payoff)

jac = g.jacobian(block_weights).subs(root).applyfunc(simplify_exact)
shape_eigen = -2/sp.sqrt(3)-eta*sp.sqrt(3)/2
scale_eigen = -eta*(3+sp.sqrt(3))/4
check_zero("n4 first shape eigenvector", jac*sp.Matrix([1,-1,0])-shape_eigen*sp.Matrix([1,-1,0]))
check_zero("n4 second shape eigenvector", jac*sp.Matrix([1,0,-1])-shape_eigen*sp.Matrix([1,0,-1]))
check_zero("n4 scale eigenvector", jac*sp.ones(3,1)-scale_eigen*sp.ones(3,1))
check_zero("n4 determinant formula", jac.det()-scale_eigen*shape_eigen**2)
print(f"n4 exact determinant: {sp.factor(scale_eigen*shape_eigen**2)}")

# The strict scalar grand-entry margin used by the collision example.
y=sp.symbols("y", real=True)
f=1-y+2000*(y-sp.Rational(19,20))**4
check_zero("quartic derivative vanishes at y=1",sp.diff(f,y).subs(y,1))
check_zero("quartic minimum value 1/80",f.subs(y,1)-sp.Rational(1,80))
check_zero("quartic derivative factorization",sp.diff(f,y)-8000*((y-sp.Rational(19,20))**3-sp.Rational(1,20)**3))

# Exact roots and grand gradients for the payoff-collision construction.
aa,bb,cc=sp.symbols("aa bb cc",positive=True)
s=aa+bb+cc
L=[sp.Rational(3,10)*aa+sp.Rational(21,10)*bb,
   sp.Rational(6,5)*aa+sp.Rational(2,5)*bb,
   4*cc]
grand=s+2000/s**3*sum(sp.Max(0,lj-sp.Rational(19,20)*s)**4 for lj in L)
grand_grad=sp.Matrix([sp.diff(grand,li) for li in [aa,bb,cc]])

cases=[("A",sp.Rational(3,10),sp.Rational(21,10),
        [sp.Rational(11,20),sp.Rational(7,20),sp.Rational(1,10)],
        sp.Matrix([[-sp.Rational(97,60),-sp.Rational(19,60)],
                   [sp.Rational(83,60),sp.Rational(41,60)]]),-sp.Rational(2,3)),
       ("B",sp.Rational(6,5),sp.Rational(2,5),
        [sp.Rational(3,5),sp.Rational(1,5),sp.Rational(1,5)],
        sp.Matrix([[-sp.Rational(22,15),-sp.Rational(3,5)],
                   [sp.Rational(23,15),sp.Rational(2,5)]]),sp.Rational(1,3))]

for name,a,b,lam,expected_jac,expected_det in cases:
    lam_subs={aa:lam[0],bb:lam[1],cc:lam[2]}
    L_values=[lj.subs(lam_subs) for lj in L]
    assert all(value<sp.Rational(19,20) for value in L_values)
    assert (L_values[0]>L_values[1]) if name=="A" else (L_values[1]>L_values[0])
    print(f"PASS collision {name} strict flat region and strict active pair branch: {L_values}")
    check_zero(f"collision {name} grand support",grand.subs(lam_subs)-1)
    check_zero(f"collision {name} grand gradient",grand_grad.subs(lam_subs)-sp.ones(3,1))
    # In the open flat grand region, local weighted payoffs equal lambda.
    chart_pair=a*u+b*v
    chart_z=1-u-v
    local_g=sp.Matrix([sp.Rational(1,3)-u+(chart_pair+4*chart_z)/6,
                       sp.Rational(1,3)-v+(chart_pair-8*chart_z)/6])
    chart_root={u:lam[0],v:lam[1]}
    check_zero(f"collision {name} local root",local_g.subs(chart_root))
    local_jac=local_g.jacobian([u,v])
    check_zero(f"collision {name} exact Jacobian",local_jac-expected_jac)
    check_zero(f"collision {name} exact determinant",local_jac.det()-expected_det)
    print(f"collision {name} Jacobian: {local_jac.tolist()}; determinant: {local_jac.det()}")

print("All exact symbolic checks passed.")
print("Scope: identity verification for the exact examples; the LaTeX proofs establish admissibility, superadditivity, n4 uniqueness, and local regularity. No global collision-root enumeration or payoff-parity claim.")
