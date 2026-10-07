"""Exact checks and high precision local checks for Phase 3E examples.

The coupled n=4 computations locate three branches and check point Jacobians;
they do not certify boxes or prove a complete root list.
"""

import argparse
import json
from pathlib import Path

import mpmath as mp
import sympy as sp

mp.mp.dps = 80

t = sp.symbols("t", real=True)
q = sp.sqrt(t**2 + (1-t)**2)
G = (sp.Rational(1,5)*t-sp.Rational(1,2)*(1-t))/2+(1-2*t)/(2*q)
n2 = {
    "root_residual": str(sp.simplify(G.subs(t,sp.Rational(3,7)))),
    "root_derivative": str(sp.simplify(sp.diff(G,t).subs(t,sp.Rational(3,7)))),
    "derivative_identity_check": str(sp.simplify(sp.diff(G,t)-(sp.Rational(7,20)-1/(2*q**3)))),
}

u,v = sp.symbols("u v", real=True)
z = 1-u-v
collision = []
for name,a,b,root in [
    ("A",sp.Rational(3,10),sp.Rational(21,10),(sp.Rational(11,20),sp.Rational(7,20))),
    ("B",sp.Rational(6,5),sp.Rational(2,5),(sp.Rational(3,5),sp.Rational(1,5))),
]:
    pair = a*u+b*v
    g = sp.Matrix([sp.Rational(1,3)-u+(pair+4*z)/6,
                   sp.Rational(1,3)-v+(pair-8*z)/6])
    jac = g.jacobian([u,v])
    subs = {u:root[0],v:root[1]}
    collision.append({"name":name,"lambda":[str(root[0]),str(root[1]),str(1-sum(root))],
                      "residual":[str(sp.simplify(x.subs(subs))) for x in g],
                      "jacobian":[[str(x) for x in row] for row in jac.tolist()],
                      "determinant":str(jac.det())})

# A separate exact check of the Euclidean-block four-player anchor example.
w,eta_sym = sp.symbols("w eta", real=True)
lam3=[u,v,w]
mass=sum(lam3)
qb=sp.sqrt(sum(x*x for x in lam3))
qmass=sp.sqrt(mass**2+(1-mass)**2)
gs=sp.Matrix([qb/3-x*x/qb+eta_sym*(qmass/4-x*mass/qmass) for x in lam3])
root_component=(3-sp.sqrt(3))/6
subs={u:root_component,v:root_component,w:root_component}
exact_simplify=lambda x:sp.simplify(sp.sqrtdenest(sp.simplify(x)))
js=gs.jacobian(lam3).subs(subs).applyfunc(exact_simplify)
shape_eigen=-2/sp.sqrt(3)-eta_sym*sp.sqrt(3)/2
scale_eigen=-eta_sym*(3+sp.sqrt(3))/4
det_expected=scale_eigen*shape_eigen**2
n4_exact={
    "root_residuals":[str(exact_simplify(x.subs(subs))) for x in gs],
    "shape_eigenvector_check":[str(exact_simplify(x)) for x in js*sp.Matrix([1,-1,0])-shape_eigen*sp.Matrix([1,-1,0])],
    "scale_eigenvector_check":[str(exact_simplify(x)) for x in js*sp.ones(3,1)-scale_eigen*sp.ones(3,1)],
    "determinant_identity_check":str(exact_simplify(js.det()-det_expected)),
    "exact_determinant":str(det_expected),
}

M = mp.mpf
epsilon = M("0.05")
p = 120
eta = M("0.001")
sqrt3 = mp.sqrt(3)
r = sqrt3/(sqrt3+1)
r_anchor = 1/(sqrt3+1)
Q = mp.sqrt(r*r+r_anchor*r_anchor)
delta = eta*sqrt3/2
c = [((0,1),(M(1)/9,M(77)/36)),
     ((0,2),(M(7)/30,M(73)/30)),
     ((1,2),(M(41)/20,M(1)/10))]
xA = [M(3)/4,M(43)/40,M(27)/20]
xB = [M(43)/52,M(1),M(41)/32]

def base_support_and_gradient(lam):
    components=[]
    gradients=[]
    for (i,j),(a,b) in c:
        norm=mp.sqrt(lam[i]**2+lam[j]**2)
        components.append(a*lam[i]+b*lam[j]+epsilon*norm)
        grad=[M(0),M(0),M(0)]
        grad[i]=a+epsilon*lam[i]/norm
        grad[j]=b+epsilon*lam[j]/norm
        gradients.append(grad)
    for x in [xA,xB]:
        components.append(sum(lam[i]*x[i] for i in range(3)))
        gradients.append(x)
    h=sum(a**p for a in components)**(M(1)/p)
    grad=[sum((components[k]/h)**(p-1)*gradients[k][i] for k in range(5)) for i in range(3)]
    return h,grad,components[:3]

def base_g(lam):
    h,x,pairs=base_support_and_gradient(lam)
    p12,p13,p23=pairs
    phi=[h/3+(p12+p13-2*p23)/6,
         h/3+(p12+p23-2*p13)/6,
         h/3+(p13+p23-2*p12)/6]
    return [phi[i]-lam[i]*x[i] for i in range(3)]

def internal(a,b):
    rho=[a,b,1-a-b]
    g=base_g(rho)
    return [g[i]+delta*(M(1)/3-rho[i]) for i in range(2)]

def full_g(a,b,cval):
    lam=[a,b,cval,1-a-b-cval]
    rb=sum(lam[:3])
    ra=lam[3]
    qglobal=mp.sqrt(rb*rb+ra*ra)
    rho=[li/rb for li in lam[:3]]
    gb=base_g(rho)
    return [rb*gb[i]+eta*(qglobal/4-lam[i]*rb/qglobal) for i in range(3)]

seeds=[("1","0.3873277033","0.4171080596"),
       ("2","0.5384740734","0.1469485836"),
       ("3","0.5548420673","0.1196628780")]
coupled=[]
for name,a,b in seeds:
    a,b=mp.findroot(internal,(M(a),M(b)),tol=M("1e-70"),maxsteps=100)
    rho=[a,b,1-a-b]
    lam=[r*x for x in rho]+[r_anchor]
    h,x,_=base_support_and_gradient(rho)
    payoff=[x[i]+eta*r/Q for i in range(3)]+[1+eta*r_anchor/Q]
    ij=mp.matrix([[mp.diff(lambda aa:internal(aa,b)[i],a),
                   mp.diff(lambda bb:internal(a,bb)[i],b)] for i in range(2)])
    gj=mp.matrix([[mp.diff(lambda aa:full_g(aa,lam[1],lam[2])[i],lam[0]),
                   mp.diff(lambda bb:full_g(lam[0],bb,lam[2])[i],lam[1]),
                   mp.diff(lambda cc:full_g(lam[0],lam[1],cc)[i],lam[2])]
                 for i in range(3)])
    coupled.append({"name":name,"rho":[mp.nstr(x,65) for x in rho],
                    "lambda":[mp.nstr(x,65) for x in lam],
                    "payoff":[mp.nstr(x,65) for x in payoff],
                    "internal_point_determinant":mp.nstr(mp.det(ij),65),
                    "full_point_determinant":mp.nstr(mp.det(gj),65),
                    "max_full_point_residual":mp.nstr(max(abs(x) for x in full_g(*lam[:3])),10)})

out={"precision_decimal_digits":mp.mp.dps,"n2_exact":n2,"collision_exact":collision,
     "euclidean_n4_exact":n4_exact,
     "coupled_n4":{"epsilon":str(epsilon),"p":p,"eta":str(eta),
                   "block_mass":mp.nstr(r,65),"anchor_mass":mp.nstr(r_anchor,65),
                   "Q":mp.nstr(Q,65),"delta":mp.nstr(delta,65),"branches":coupled,
                   "scope":"High precision local point checks; no interval certification or complete enumeration."}}
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path,
                    default=Path(__file__).resolve().parents[1]/"results"/"examples"/"verification_results.json")
args = parser.parse_args()
output=args.output
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out,indent=2))
