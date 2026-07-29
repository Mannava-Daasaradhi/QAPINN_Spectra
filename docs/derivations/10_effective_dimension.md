# Effective Dimension and Fisher Information as a Capacity Measure

`project.md` §3 item 10 (Abbas et al. 2021). Formalizes the derivation already worked out
ad hoc while implementing `xai/fisher.py` (T1.7) — see `docs/plan/BENCH.md`'s T1.7
section for the numerical confirmation this document's formula is the correct one.

## Empirical Fisher information

For a model with parameters $\theta$ producing output $N_\theta(x)$, and a least-squares
objective (this project's residual loss is exactly this form), the classical Fisher
information matrix reduces to the **Gauss–Newton** form:
$$
F = \mathbb{E}_x\big[\nabla_\theta N_\theta(x)\, \nabla_\theta N_\theta(x)^\top\big]
\;\approx\; \frac1B\sum_{b=1}^B g_b g_b^\top, \qquad g_b := \nabla_\theta N_\theta(x_b),
$$
— the same object as the NTK (`03_ntk_pinn.md`), but with the **outer product taken over
the parameter axis** ($F$ is $P\times P$, $P=\dim\theta$) rather than the data axis ($\Theta$
is $B\times B$): $F = J^\top J / B$ and $\Theta = J J^\top$ for the same Jacobian $J\in
\mathbb{R}^{B\times P}$, $J_{b,k}=\partial N_\theta(x_b)/\partial\theta_k$. $F$ is PSD by
the same Gram-matrix argument as $\Theta$, and its nonzero eigenvalues are shared with
$\Theta$'s (both equal the squared singular values of $J$) — this is exactly why
`xai/fisher.py`'s `empirical_fisher` can compute large-$P$ eigenvalue spectra via a
(randomised) SVD of $J$ directly, never materializing the $P\times P$ matrix (T1.7).

## Effective dimension (Abbas et al. 2021)

The *effective dimension* asks: of the $P$ nominal parameters, how many are actually
"active" — contributing distinguishable directions of variation in the model's output, as
measured by $F$'s eigenvalue spectrum $\{\lambda_i\}$ (normalized so $\overline\lambda=1$
by convention). Abbas et al. define, for $n$ data points and confidence parameter
$\gamma\in(0,1]$,
$$
d_{\gamma,n}(\mathcal{M}) = 2\,\frac{\log\!\left(\frac1{V_\Theta}\int_\Theta
\sqrt{\det\!\big(I + \tfrac{\gamma n}{2\pi\log n} \hat F(\theta)\big)}\, d\theta\right)}
{\log\!\left(\frac{\gamma n}{2\pi\log n}\right)} .
$$
The determinant term is a volume-growth measure of the local Fisher-information ellipsoid
(a Laplace/Bayesian-information-criterion-style capacity measure — parameters along which
$F$ has large eigenvalues contribute more to the *effective* count than parameters along
near-null directions of $F$, which barely change the model's predictions and so should
not count as "really" adding capacity).

## The single-point (MLE) approximation, and its correct normalisation

Evaluating the *integral* over parameter space is generally intractable; the practical
approximation replaces it with the value at a single point $\theta$ (e.g. the trained /
current parameters):
$$
d_{\gamma,n} \approx \frac{2\log\sqrt{\det(I+c\hat F)}}{\log c}, \qquad c := \frac{\gamma n}{2\pi\log n}.
$$
**Simplify the numerator.** $\det(I+c\hat F) = \prod_i (1+c\lambda_i)$ (determinant of a
matrix diagonal in $\hat F$'s eigenbasis), so
$$
\log\sqrt{\det(I+c\hat F)} = \tfrac12\log\prod_i(1+c\lambda_i) = \tfrac12\sum_i \log(1+c\lambda_i).
$$
Substituting back, **the outer factor of $2$ and this inner factor of $\tfrac12$ cancel
exactly**:
$$
\boxed{d_{\gamma,n} \approx \frac{\sum_i \log(1+c\lambda_i)}{\log c}, \qquad c = \frac{\gamma n}{2\pi\log n}, \quad \lambda_i \text{ normalised so } \overline\lambda=1.}
$$
This is a genuine algebraic step, not a restatement — a naive transcription that keeps
the outer "$2\cdot$" *without* first simplifying $\log\sqrt{\det(\cdot)}=\tfrac12\sum\log(\cdot)$
produces a formula with a spurious leading factor of 2, exactly the error found (and
fixed) while implementing this in `xai/fisher.py` (T1.7): for $k$ equal eigenvalues
(after normalisation, $\lambda_i=1$ for all $i$), the uncorrected "$2\sum_i\log(1+c)/\log
c$" formula provably converges to $2k$ as $n\to\infty$ (verified numerically to $10^{-12}$
relative error before the fix — see `docs/plan/BENCH.md`'s T1.7 section), not $k$; the
formula boxed above converges to exactly $k$, matching the intended interpretation
("$k$ independent, fully-informative parameters $\Rightarrow$ effective dimension $k$").

*Proof of the limit, for completeness.* As $n\to\infty$, $c=\gamma n/(2\pi\log n)\to\infty$.
For $k$ eigenvalues all equal to $1$: $\sum_i\log(1+c\lambda_i) = k\log(1+c)$, and
$\log(1+c)/\log c = 1 + \log(1+1/c)/\log c \to 1$ as $c\to\infty$ (since $\log(1+1/c)\to
0$ while $\log c\to\infty$), so the boxed formula $\to k\cdot 1 = k$.

## Effective dimension as a capacity measure, and why it belongs in this project's XAI suite

Unlike raw parameter count $P$, $d_{\gamma,n}$ accounts for *how much the parameters
actually matter* to the model's output, exactly the sense in which a quantum circuit with
few genuinely-informative gate angles should be credited with a small effective
dimension even if $P$ is nominally large (or vice versa) — this is the "quantum-native
XAI metric" `project.md` flags it as, because effective dimension (unlike, say, raw qubit
or gate count) is defined identically for classical and quantum models via the same
Fisher-information object, making direct classical-vs-quantum comparisons meaningful.
`xai/fisher.py`'s DoD (T1.7) tests exactly the $k$-independent-features limit derived
above: a linear model with $k$ independent (orthonormal sine-basis) features recovers
$d_{\mathrm{eff}}=k$ to $10^{-10}$ relative error as $n\to 10^{12}$.
