# Reverse-Mode AD Gives Exact Derivatives; Cost of the $k$-th Derivative

`project.md` §3 item 2.

## Forward evaluation as a computational graph

Any function computed by composing elementary operations — $u_\theta(x) = g_L\circ
g_{L-1}\circ\cdots\circ g_1(x,\theta)$ (a neural network's layers, or any differentiable
program) — defines a directed acyclic graph of intermediate values $v_0=x$, $v_l =
g_l(v_{l-1})$, $v_L = u_\theta(x)$. Every $g_l$ is a *known, differentiable* primitive
(matrix multiply, `tanh`, addition, ...), so its local Jacobian $J_l = \partial v_l /
\partial v_{l-1}$ is computable in closed form at the point where it is evaluated.

## Reverse-mode: the chain rule applied back-to-front

The full Jacobian of the composition is, by the chain rule,
$$
\frac{\partial v_L}{\partial x} = J_L J_{L-1} \cdots J_1 .
$$
**Forward-mode** AD computes this product left-to-right, propagating a *tangent* vector
$\dot v_0 = \dot x$ forward through $\dot v_l = J_l \dot v_{l-1}$ — one forward pass per
INPUT direction needed, cost $O(d_{\text{in}})$ passes for a $d_{\text{in}}$-dimensional
input.

**Reverse-mode** AD instead computes the product right-to-left: it first runs the forward
pass once (storing every $v_l$), then propagates a *cotangent* (adjoint) vector backward,
$\bar v_{l-1} = J_l^\top \bar v_l$, starting from $\bar v_L = $ (the seed, e.g. $1$ for a
scalar output). This is exactly a sequence of **vector-Jacobian products** (VJPs), each
costing about the same as one forward evaluation of $g_l$ — so the **entire gradient**
$\partial v_L/\partial x = \bar v_0$ is obtained in **one backward pass**, cost $O(1)$
forward-pass-equivalents *regardless of $d_{\text{in}}$*. This is why reverse-mode
(`torch.autograd.grad`, used everywhere in this codebase, `01_CONVENTIONS.md` §6) is the
right choice whenever the output is scalar (or low-dimensional) and the input is
high-dimensional — exactly the PINN setting ($u_\theta:\mathbb{R}^d\to\mathbb{R}$, but
$\theta$ has $O(10^3\text{--}10^5)$ components).

## Exactness (versus finite differences)

Every $J_l$ used above is the *exact, closed-form* derivative of a known primitive (e.g.
$\partial\tanh(z)/\partial z = 1-\tanh^2(z)$, evaluated at the exact stored $z$). The
chain rule is then applied with **exact arithmetic** (up to floating-point rounding, the
same source of error present in the forward evaluation itself — no *additional*
approximation is introduced). Contrast with a finite-difference estimate
$$
\frac{\partial f}{\partial x} \approx \frac{f(x+h)-f(x-h)}{2h} = \frac{\partial f}{\partial x} + \frac{h^2}{6}f'''(\xi) + O(h^4),
$$
which has an **irreducible truncation error** of order $h^2$ (central difference) —
choosing $h$ smaller trades truncation error for floating-point cancellation error
(subtracting two nearly-equal numbers), and there is no value of $h$ that eliminates
both. Reverse-mode AD has **no truncation error term at all**: it computes the same
derivative the calculus rules would give on paper. This is exactly why
`01_CONVENTIONS.md` §6 mandates `torch.autograd.grad` everywhere in this project and
explicitly forbids finite differences.

## Cost of the $k$-th derivative, and why $u_{xx}$ in 2-D is the real bottleneck

Computing a **second** derivative means differentiating the *first-derivative graph*
again. Concretely, `diffops.d1(u, x, i)` (this codebase) calls
`torch.autograd.grad(u, x, create_graph=True)` — the `create_graph=True` flag means the
backward pass **itself is recorded as a new computational graph**, roughly as large as
the forward pass plus the first backward pass combined. `diffops.d2` then calls
`torch.autograd.grad` *again* on that first-derivative graph to get the second
derivative. Each additional order of differentiation:

- **Adds one more backward pass** through a graph that is itself already the record of a
  previous backward pass — the graph size grows roughly linearly with the differentiation
  order (each `create_graph=True` call appends the new backward computation's own graph
  on top of what's already there), so the $k$-th derivative costs roughly $O(k)$ forward-pass
  equivalents, not $O(1)$.
- **Cannot be shared across independent second partials for free.** The Laplacian in $d$
  spatial dimensions, $\Delta u = \sum_{i=1}^d \partial^2 u/\partial x_i^2$, requires $d$
  *separate* second-derivative computations (`diffops.laplacian` loops over `dims` and
  calls `d2(u,x,i,i)` for each $i$) — each is its own `torch.autograd.grad` call on the
  first-derivative graph, since $\partial^2 u/\partial x_i^2$ and $\partial^2
  u/\partial x_j^2$ are extracted from the *same* first-derivative vector $\nabla u$ but
  require differentiating *different* scalar components of it, which PyTorch's
  `grad_outputs` mechanism handles as independent backward passes.

For a 1-D problem ($d=1$, P1), the Laplacian is a single second derivative — cheap. For a
2-D problem (P4, Helmholtz), $\Delta u = u_{xx}+u_{yy}$ needs **two** independent
second-derivative backward passes per residual evaluation, each roughly $O(1)$ forward-pass
cost on top of an already-doubled (first-derivative) graph — this compounding is exactly
why this project's own performance spike (T0.21, `docs/plan/BENCH.md`) measured P4
(Helmholtz, second derivatives in 2-D) at roughly **2x** the per-step cost of P1
(Poisson, second derivative in 1-D) even before accounting for any quantum-layer cost —
purely from the extra second-derivative pass. This is the "real bottleneck" the phase doc
flags: the cost of $k$-th-order autodiff is genuinely governed by the *number of distinct
second partials the PDE operator needs*, not merely by network size.
