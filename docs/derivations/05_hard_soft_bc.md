# Hard vs. Soft Boundary Enforcement, and Why the Hard Ansatz Removes an NTK Block

`project.md` §3 item 5; D6 (`00_MASTER_PLAN.md`); builds on `03_ntk_pinn.md`'s block-NTK
result.

## Soft BC

The direct approach adds the boundary condition as an extra *loss term*:
$$
J(\theta) = \underbrace{\frac1{N_r}\sum_i \big(\mathcal{L}[u_\theta](x_i)-f(x_i)\big)^2}_{\text{residual}}
\;+\; \lambda_b\underbrace{\frac1{N_b}\sum_j \big(u_\theta(x_j)-g(x_j)\big)^2}_{\text{boundary}} ,
$$
with $u_\theta$ an *unconstrained* network output. As derived in `03_ntk_pinn.md`,
linearizing gradient flow on this two-term objective gives a **block** NTK $\Theta =
\begin{pmatrix}\Theta_{rr}&\Theta_{rb}\\\Theta_{br}&\Theta_{bb}\end{pmatrix}$ acting on the
stacked error $(e_r; e_b)$ — both blocks are present in the loss, both blocks' parameter
gradients interact through $\Theta$, and neither the residual nor the boundary condition
is satisfied exactly at any finite training time (both are merely *penalized*).

## Hard BC: the ansatz $u_\theta = B(x) + D(x)\,N_\theta(x)$

Instead, define
$$
u_\theta(x) := B(x) + D(x)\, N_\theta(x),
$$
where $N_\theta$ is the raw network output and $B, D$ are **fixed** (not trained)
functions chosen so that:
$$
B(x_b) = g(x_b) \quad\text{and}\quad D(x_b) = 0 \qquad \text{for every } x_b\in\partial\Omega.
$$
($B$ is the "lift" — any smooth extension of the boundary data into the interior; $D$ is
the "mask" — vanishes exactly on $\partial\Omega$. This codebase: `PDE.bc_lift`,
`PDE.bc_mask`, `PDE.apply_hard_bc`, `01_CONVENTIONS.md` §5; e.g. P1's mask $D(x)=x(1-x)$
vanishes at $x=0,1$.)

**Claim: $u_\theta$ satisfies the boundary condition exactly, for every $\theta$, with no
boundary loss term at all.** Evaluate at $x_b\in\partial\Omega$:
$$
u_\theta(x_b) = B(x_b) + D(x_b)\, N_\theta(x_b) = g(x_b) + 0\cdot N_\theta(x_b) = g(x_b),
$$
using $D(x_b)=0$ to annihilate the (arbitrary, untrained) $N_\theta(x_b)$ term entirely —
this holds identically in $\theta$, before any training. There is therefore **nothing to
penalize**: $u_\theta - g \equiv 0$ on $\partial\Omega$ by construction, for every
$\theta$ the optimizer will ever visit, so the loss reduces to the residual term alone,
$$
J(\theta) = \frac1{N_r}\sum_i \big(\mathcal{L}[u_\theta](x_i)-f(x_i)\big)^2 \qquad \text{(no boundary term)}.
$$

## This removes the boundary NTK block entirely

Re-run `03_ntk_pinn.md`'s linearization on this loss: there is only one residual vector
$e_r$ now (no $e_b$ exists — $u_\theta-g\equiv 0$ has no gradient information to
contribute, since it's identically zero regardless of $\theta$), so the block NTK
collapses to the single block $\Theta_{rr}$:
$$
\dot e_r = -\Theta_{rr}\, e_r \qquad\text{(no } \Theta_{rb}, \Theta_{br}, \Theta_{bb} \text{ terms at all).}
$$
The boundary condition is not merely *given a large weight* or *learned quickly* — it is
**absent from the optimization problem**, satisfied identically by construction. This is
exactly "removes one NTK block entirely," derived rather than asserted: with soft BC there
genuinely are two coupled blocks (`xai/ntk.py`'s `block_mass`, T1.2, only reports
non-trivial $\Theta_{bb}$, $\Theta_{rb}$ under `bc_mode='soft'`, per that function's own
docstring — "only meaningful under soft BC" — precisely because under hard BC those
blocks do not exist).

## Trade-off

The hard ansatz is not free: constructing $B,D$ satisfying the required vanishing
properties is straightforward for simple domains (P1's interval, P4's square) but becomes
genuinely hard to construct for complex geometries or nonlinear/non-Dirichlet boundary
operators — which is exactly why this project keeps the soft-BC path available as a
variant (D6: "hard default, soft variant") rather than removing it outright. For the
domains in this project's PDE ladder (`project.md` §4), the hard ansatz is always
constructible in closed form, so it is the default.
