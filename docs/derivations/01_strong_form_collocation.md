# Strong-Form Residual Minimization and Why a PINN Is a Collocation Method

`project.md` §3 item 1.

## Setup

Consider a linear (or nonlinear) PDE on domain $\Omega\subset\mathbb{R}^d$:
$$
\mathcal{L}[u](x) = f(x),\; x\in\Omega, \qquad \mathcal{B}[u](x) = g(x),\; x\in\partial\Omega .
$$
A PINN parameterizes $u_\theta:\Omega\to\mathbb{R}$ by a neural network and minimizes
$$
J(\theta) = \frac{1}{N_r}\sum_{i=1}^{N_r} \big(\mathcal{L}[u_\theta](x_i) - f(x_i)\big)^2
\;+\; \lambda\,\frac{1}{N_b}\sum_{j=1}^{N_b} \big(\mathcal{B}[u_\theta](x_j) - g(x_j)\big)^2 ,
$$
where $\{x_i\}$ are *collocation points* sampled in $\Omega$ and $\{x_j\}$ on
$\partial\Omega$ (this codebase's `pinn_loss`, `01_CONVENTIONS.md` §8). $\mathcal{L}[u_\theta]$
is evaluated by literally substituting $u_\theta$ into the PDE operator — pointwise, at
each $x_i$, using automatic differentiation to compute $\partial_x u_\theta$,
$\partial_x^2 u_\theta$, etc. — and driving the residual $\mathcal{L}[u_\theta]-f$ to zero
in a least-squares sense.

## This is a strong-form method

$\mathcal{L}[u_\theta](x_i) - f(x_i)$ requires $u_\theta$ to be differentiable to the
order of $\mathcal{L}$ *pointwise* at $x_i$, and the loss penalizes the **strong-form
residual directly at a finite set of points** — no averaging against a test function, no
integration by parts. This is exactly the definition of a **collocation method**: solve
$\mathcal{L}[u]=f$ by forcing the residual to vanish (or, here, to be minimized) at a
discrete set of *collocation points*, as opposed to forcing a *weak* (integrated,
test-function-averaged) form to vanish. That $u_\theta$ happens to be a neural network
rather than a global polynomial (classical spectral collocation, e.g. Chebyshev) or a
piecewise polynomial (finite differences) is a detail of the ansatz; the *numerical
method* — pointwise strong-form residual minimization at a point set — is unchanged.

## Contrast with Galerkin / FEM

The Galerkin method starts from the **weak form**: multiply $\mathcal{L}[u]=f$ by an
arbitrary test function $v$ in a test space $V$ and integrate by parts once (for a
second-order operator) to move one derivative onto $v$:
$$
a(u,v) := \int_\Omega \nabla u\cdot\nabla v \,dx = \int_\Omega f v\,dx =: \ell(v) \quad \forall v\in V .
$$
FEM then (i) builds a **mesh** of $\Omega$, (ii) chooses a **finite-dimensional subspace**
$V_h\subset V$ spanned by piecewise-polynomial basis functions supported on the mesh
elements, and (iii) requires $a(u_h,v_h)=\ell(v_h)$ for every $v_h\in V_h$ — the Galerkin
projection. Writing $u_h = \sum_k c_k \phi_k$ in the basis $\{\phi_k\}$ of $V_h$ turns
this into a **linear system** $Ac=b$, $A_{kl}=a(\phi_k,\phi_l)$, solved exactly (up to
floating point) by direct or iterative linear-algebra methods.

The structural differences that matter:

| | PINN (strong-form collocation) | Galerkin / FEM (weak form) |
|---|---|---|
| Form of the equation used | strong (pointwise) | weak (integrated against test functions) |
| Requires a mesh? | **No** — $u_\theta$ is a single global (nonlinear) function, collocation points are just samples | **Yes** — $V_h$ is built from mesh elements |
| Resulting problem in the unknowns | Non-convex least-squares in $\theta$ (network is nonlinear in $\theta$) | **Linear** system in $c$ (if $\mathcal{L}$ is linear and $V_h$ is a linear span) |
| Differentiation order needed on the trial function | Full order of $\mathcal{L}$ (e.g. $C^2$ pointwise for a 2nd-order PDE) | Only *half* the order (integration by parts moves derivatives onto $v$; e.g. $H^1$, not $H^2$, suffices for a 2nd-order elliptic problem) |
| Conservation / stability guarantees | None automatic — must be checked empirically | Often automatic from the variational structure (e.g. energy norms, coercivity give a priori error bounds via Céa's lemma) |

Two consequences worth deriving explicitly, since they explain *why* PINNs behave
differently from FEM in practice (not just definitionally):

1. **Non-convexity.** In Galerkin/FEM, $u_h$ is *linear* in the unknowns $c$ (a fixed
   basis, linear coefficients), so for linear $\mathcal{L}$ the discretized problem
   $Ac=b$ is a linear system — convex, uniquely solvable (given well-posedness), no local
   minima. In a PINN, $u_\theta(x)$ is a *nonlinear* function of $\theta$ (composition of
   affine maps and nonlinear activations), so even for linear $\mathcal{L}$, $J(\theta)$
   is a **non-convex** function of $\theta$ (nonlinear reparameterization of a convex
   loss in $u$-space does not preserve convexity in $\theta$-space) — trained by
   first-order stochastic optimization (Adam, L-BFGS; `01_CONVENTIONS.md` §8) rather than
   solved exactly. This is the direct cost of trading a fixed linear basis for a flexible,
   trainable nonlinear ansatz.
2. **Higher differentiability requirement, exactly computable.** Because there is no
   integration by parts, the strong-form residual needs $\partial_x^{\mathrm{ord}(\mathcal{L})}
   u_\theta$ pointwise — for `-u''=f` (P1) this means $u_\theta$ must be twice
   differentiable *as a function*, which a smooth-activation network (`tanh`, this
   project's `c_mlp`, T0.15) satisfies everywhere, and reverse-mode automatic
   differentiation computes these derivatives **exactly** (see
   `02_reverse_mode_ad.md`) rather than via a fixed stencil (finite differences) or a
   weak-form relaxation (FEM) — the PINN needs *more* smoothness from its trial function
   than FEM does, but in exchange gets exact pointwise derivatives essentially for free.
