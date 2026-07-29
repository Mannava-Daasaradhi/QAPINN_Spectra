# The Neural Tangent Kernel for a PINN, Lazy Training, and Loss-Term Imbalance

`project.md` §3 item 3.

## Definition

For a network $u_\theta$ (or, for a PINN, its *residual* $N[u_\theta](x) :=
\mathcal{L}[u_\theta](x)$), the empirical Neural Tangent Kernel is
$$
\Theta(x,x') = \nabla_\theta N[u_\theta](x) \cdot \nabla_\theta N[u_\theta](x')^\top
= \sum_{k=1}^{P} \frac{\partial N[u_\theta](x)}{\partial\theta_k}\,\frac{\partial N[u_\theta](x')}{\partial\theta_k},
$$
i.e. the Gram matrix of the *per-parameter gradients* of the model output, evaluated at a
(finite) set of collocation points $\{x_i\}_{i=1}^B$: $\Theta_{ij} = \Theta(x_i,x_j)$, a
$B\times B$ positive semi-definite matrix (a Gram matrix is always PSD:
$v^\top\Theta v = \|\sum_i v_i \nabla_\theta N(x_i)\|^2 \ge 0$ for any $v$ — this is
exactly what `xai/ntk.py`'s `spectrum_stats` asserts numerically, T1.2).

## Linearized (lazy) training dynamics

Consider gradient flow on a least-squares loss $J(\theta) = \tfrac12\sum_i
(N[u_\theta](x_i) - 0)^2$ (residual-loss form; the same argument applies to any
least-squares objective, including the boundary term):
$$
\dot\theta = -\nabla_\theta J(\theta) = -\sum_i N[u_\theta](x_i)\,\nabla_\theta N[u_\theta](x_i).
$$
Let $e_i(t) := N[u_{\theta(t)}](x_i)$ be the residual vector. By the chain rule,
$$
\dot e_i = \nabla_\theta N[u_\theta](x_i)\cdot\dot\theta
= -\sum_j \nabla_\theta N[u_\theta](x_i)\cdot\nabla_\theta N[u_\theta](x_j)\; e_j
= -\sum_j \Theta_{ij}(\theta)\, e_j .
$$
In vector form, $\dot e = -\Theta(\theta)\, e$. In the **lazy / NTK regime** (wide
network, small learning rate, or more precisely the infinite-width limit where $\Theta$
provably stays close to its value at initialization throughout training — Jacot et al.
2018), $\Theta(\theta(t))\approx\Theta(\theta(0)) =: \Theta_0$ is *approximately constant*,
turning the nonlinear training dynamics into a **linear** ODE:
$$
\boxed{\dot e = -\Theta_0\, e.}
$$

## Mode-wise exponential decay

$\Theta_0$ is symmetric PSD, so it diagonalizes: $\Theta_0 = \sum_i \lambda_i v_i
v_i^\top$, $\lambda_i \ge 0$, $\{v_i\}$ orthonormal. Writing $e(t) = \sum_i \alpha_i(t)
v_i$ and substituting into $\dot e = -\Theta_0 e$ decouples the ODE mode-by-mode:
$\dot\alpha_i = -\lambda_i \alpha_i \;\Rightarrow\; \alpha_i(t) = \alpha_i(0)\,
e^{-\lambda_i t}.$
$$
\boxed{e(t) = \sum_i \alpha_i(0)\, e^{-\lambda_i t}\, v_i \quad\Longrightarrow\quad
\text{mode } i \text{ decays as } \exp(-\lambda_i t).}
$$
Large-$\lambda_i$ eigendirections of $\Theta_0$ collapse fast; small-$\lambda_i$
directions decay slowly and dominate the *late-time* residual. This single formula is the
mechanism behind spectral bias once the eigenbasis is identified with Fourier modes
(`04_spectral_bias.md`) and is exactly what `xai/ntk.py`'s `spectrum_stats` and this
project's per-frequency error trajectory (`xai/spectral_error.py`, T1.4/T1.5) are built to
measure directly: T1.5's spectral-bias staircase is a literal picture of $e(t)$ collapsing
mode-by-mode, ordered by $\lambda_i$.

## Loss-term imbalance $=$ eigenvalue imbalance

When the loss has multiple blocks — residual and boundary (soft-BC mode, D6) — write
$J(\theta) = \tfrac12\|e_r\|^2 + \tfrac{\lambda_b}{2}\|e_b\|^2$ over the stacked
collocation/boundary points. The *same* linearization gives a **block** NTK,
$$
\Theta = \begin{pmatrix}\Theta_{rr} & \Theta_{rb}\\ \Theta_{br} & \Theta_{bb}\end{pmatrix},
\qquad
\begin{pmatrix}\dot e_r\\ \dot e_b\end{pmatrix} = -\Theta\begin{pmatrix}e_r\\ e_b\end{pmatrix}
$$
(this is exactly `xai/ntk.py`'s `block_mass`, T1.2, which reports $\operatorname{tr}(\Theta_{rr})$,
$\operatorname{tr}(\Theta_{bb})$, $\operatorname{tr}(\Theta_{rb})$). If $\Theta_{rr}$'s
eigenvalues are systematically much larger (or smaller) than $\Theta_{bb}$'s, the two
blocks' errors decay at *characteristically different rates* regardless of the scalar
weight $\lambda_b$ chosen in the loss — reweighting $\lambda_b$ rescales $\Theta_{bb}$
and $\Theta_{rb}$ but cannot, by itself, fix a *shape* mismatch between the two blocks'
eigenvalue spectra (only their overall scale). This is precisely why "loss-term
imbalance" is diagnosed as an eigenvalue-imbalance phenomenon, not merely a
weight-tuning problem, and why `block_mass` reports the trace (total eigenvalue mass) per
block rather than trying to fold everything into a single scalar weight.

## Scope of the approximation

The lazy-training linearization is exact only in the idealized limit; for a finite-width
network trained for many steps, $\Theta(\theta(t))$ genuinely drifts (this project tracks
that drift directly: `xai/ntk.py`'s `ntk_drift`, T1.2, $\|\Theta_t-\Theta_0\|_F/\|\Theta_0\|_F$).
The formula above is the *first-order* picture — accurate early in training and as a
qualitative guide throughout, and it is the picture this project's entire NTK
instrumentation (T1.2–T1.3) is built to test against the actual, non-linearized training
trajectory.
