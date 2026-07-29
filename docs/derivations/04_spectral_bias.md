# Spectral Bias / F-Principle, and Why Fourier Features Fix It

`project.md` §3 item 4. Builds directly on `03_ntk_pinn.md`'s mode-decay formula
$e(t)=\sum_i\alpha_i(0)e^{-\lambda_i t}v_i$.

## From NTK eigenvalues to Fourier-mode decay rate

`03_ntk_pinn.md` shows each NTK eigendirection $v_i$ (eigenvalue $\lambda_i$) of the
*error* decays as $e^{-\lambda_i t}$. To connect this to *frequency*, identify the
eigenbasis of $\Theta_0$ with Fourier modes. For a wide network with i.i.d. initialization,
$\Theta_0(x,x')$ is (approximately, over the random initialization) a function of $x-x'$
alone for translation-insensitive input statistics — a stationary kernel. A stationary
kernel diagonalizes exactly in the Fourier basis (Bochner's theorem: $\Theta_0(x,x') =
\int \hat\Theta_0(k)\, e^{ik(x-x')}\,dk$), so **the Fourier modes ARE (approximately) the
NTK eigenbasis**, with eigenvalue $\lambda(k) = \hat\Theta_0(k)$ — the kernel's own
Fourier transform.

## Why $\hat\Theta_0(k)$ decays as a power law, not exponentially

$\Theta_0$ is built from the network's own activation function ($\tanh$ for `c_mlp`,
T0.15). The key fact connecting a kernel's *smoothness* to its Fourier-coefficient decay
rate is a standard one: **a function with $m$ continuous derivatives (and an
$(m{+}1)$-th derivative that is merely bounded, or has a jump) has a Fourier transform
that decays as $|k|^{-(m+1)}$** — integrate the Fourier integral by parts $m{+}1$ times;
each integration by parts trades one derivative of the function for one factor of
$1/(ik)$, and the process terminates (rather than being pushed further, giving faster
decay) exactly at the point where the function stops being differentiable smoothly. A
$\tanh$/ReLU-activated infinite-width network's NTK is built from *finitely many*
derivatives of the activation (ReLU: a single kink, i.e. bounded but discontinuous first
derivative; $\tanh$: analytic, but the induced NTK kernel — via the arccos-kernel-type
identities for random-feature/infinite-width limits — still has *finite* smoothness at
the relevant order in the standard NTK parameterization) — hence $\hat\Theta_0(k)$ decays
**polynomially**, $\hat\Theta_0(k)\sim |k|^{-(d+1)}$ in $d$ input dimensions (the exact
exponent depends on the activation and input dimension; Basri et al. 2020 and
Bietti & Bach 2021 derive the precise rate for ReLU networks on the sphere — this project
cites their asymptotic result rather than re-deriving the full spherical-harmonic
machinery, which is beyond whiteboard scope, but the *mechanism* — smoothness bounds
Fourier decay rate, by the integration-by-parts argument above — is exactly reproducible
by hand).

**Crucially: polynomial decay, not exponential.** This is the qualitative fact that
matters for what follows.

## Assembling spectral bias

Combine the two facts:
$$
\lambda(k) = \hat\Theta_0(k) \sim |k|^{-(d+1)} \quad\text{(polynomial decay in frequency)},
\qquad
e_k(t) \sim e^{-\lambda(k)\, t} \quad\text{(from `03_ntk_pinn.md`).}
$$
For *any fixed training time* $t$, as $|k|\to\infty$, $\lambda(k)\to 0$ polynomially, so
$e^{-\lambda(k)t}\to 1$ — **high-frequency error barely decays at all within a fixed
training budget**, while low-frequency modes ($\lambda(k)$ large) collapse quickly. This
is spectral bias (equivalently, the "F-Principle": networks fit low frequencies
first, in order of decreasing $\lambda(k)$, i.e. increasing $|k|$). It is *not* that high
frequencies are unrepresentable — a sufficiently wide/deep network can represent them
(universal approximation) — it is that **gradient descent reaches them last**, and only
after training far longer than the low-frequency modes require. This exact,
empirically-measured phenomenon is what T1.5's spectral-bias staircase figure visualizes
directly on P1, and what `xai/ntk.py`'s `decay_exponent` (T1.3, `c_mlp` baseline
$-2.037$, `docs/plan/BENCH.md`) quantifies as a single number: the (negative) slope of
$\log\lambda_i$ vs. $\log i$, i.e. an empirical estimate of the power-law exponent above.

## Fourier-feature / random-feature fixes

If, instead of feeding $x$ directly to the network, one first maps $x \mapsto
[\sin(Bx),\cos(Bx)]$ for a fixed (or random) matrix $B$ of frequencies (`c_ff`, T0.16),
the *effective* kernel becomes
$$
\Theta_0^{\mathrm{ff}}(x,x') = \phi(x)^\top W W^\top \phi(x'), \qquad \phi(x) = [\sin(Bx),\cos(Bx)],
$$
i.e. a network built on TOP of an explicit, fixed harmonic basis at the frequencies in
$B$. Its Fourier transform is no longer a smooth, slowly-decaying function of $k$ but is
instead **concentrated exactly at the rows of $B$** (each feature $\sin(b_j\cdot x)$
contributes a delta-like spike at $k=\pm b_j$ to $\hat\phi$) — so the induced NTK gets an
eigenvalue boost precisely at those frequencies, *independent of $|k|$ growing*, as long
as $B$ contains a row near $k$. This is the classical (non-quantum) way to defeat the
power-law decay above: **choose $B$ to cover the frequencies you need**, exactly the
principle this project's own `c_rff_matched` family embodies (T0.16, T2.14) and the one
`Prop. 1`/SMCD (`docs/derivations/07_circuit_fourier_spectrum.md`) replaces with a
*quantum*, *constructively-designed* discrete frequency set $\Omega$ instead of a
hand-picked or randomly-drawn $B$. `c_ff` (Tancik-style, $B\sim\mathcal{N}(0,\sigma^2)$)
is the fair classical baseline this project must beat (`project.md` §6): it fixes
spectral bias *broadly* (spreads coverage randomly over a band set by $\sigma$) but
without SMCD's guarantee that the *specific* target frequencies are covered.
