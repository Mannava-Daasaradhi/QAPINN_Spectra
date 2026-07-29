# The Parameter-Shift Rule (Exact Gradients on Hardware)

`project.md` §3 item 8 (T2.2). Extended in T2.7 with the second-order rule for $\partial_x^2$
(needed for the PDE residual) — see the bottom section, added at that task.

## Setup

Consider $f(\phi) = \langle\psi\rvert U^\dagger(\phi)\, M\, U(\phi)\lvert\psi\rangle$ where
a single gate $U(\phi) = \exp(-i\phi G/2)$ depends on a scalar parameter $\phi$ (either a
trainable angle $\theta_k$, or — see below — the *input* $x$ itself, since encoding gates
are also parameterized by an angle) and $G$ is Hermitian with **exactly two distinct
eigenvalues** $\{+1,-1\}$ (a Pauli generator; the general two-eigenvalue case reduces to
this by an affine rescaling of $\phi$).

## $f(\phi)$ is a single-frequency sinusoid — reusing `07_circuit_fourier_spectrum.md`

This is *exactly* the object `07_circuit_fourier_spectrum.md` §1–§2 already derived, with
the "layer count" set to $L=1$ and the "scaling" set to $\omega=1$ (varying $\phi$ plays
the role that $x$ played there): expanding $U(\phi)$ in the eigenbasis of $G$ gives, by
the identical bra/ket eigenvalue-difference argument,
$$
f(\phi) = c_0 + c_{+1}\, e^{i\phi} + c_{-1}\, e^{-i\phi} = A + B\cos\phi + C\sin\phi
$$
for real constants $A,B,C$ depending on $\lvert\psi\rangle,\theta,M$ (rewriting the
complex-exponential form as $A+B\cos\phi+C\sin\phi$ using $c_{-1}=c_{+1}^*$, as in
`07_circuit_fourier_spectrum.md` §1). **$f$ is an exact single-frequency (period
$2\pi$) trigonometric polynomial in $\phi$** — not merely smooth, but of this exact,
finite form. Its derivative is therefore also a *single-frequency* function:
$$
f'(\phi) = -B\sin\phi + C\cos\phi .
$$

## The shift rule, derived (not merely stated)

Evaluate $f$ at $\phi\pm\pi/2$:
$$
f(\phi+\tfrac\pi2) = A + B\cos(\phi+\tfrac\pi2) + C\sin(\phi+\tfrac\pi2) = A - B\sin\phi + C\cos\phi,
$$
$$
f(\phi-\tfrac\pi2) = A + B\cos(\phi-\tfrac\pi2) + C\sin(\phi-\tfrac\pi2) = A + B\sin\phi - C\cos\phi,
$$
using $\cos(\phi\pm\tfrac\pi2)=\mp\sin\phi$, $\sin(\phi\pm\tfrac\pi2)=\pm\cos\phi$.
Subtracting:
$$
f(\phi+\tfrac\pi2) - f(\phi-\tfrac\pi2) = -2B\sin\phi + 2C\cos\phi = 2\big(-B\sin\phi+C\cos\phi\big) = 2f'(\phi).
$$
$$
\boxed{\partial_\phi f = \tfrac12\big[f(\phi+\tfrac\pi2) - f(\phi-\tfrac\pi2)\big].}
$$
This is **exact** — no truncation error, no $h\to0$ limit — because it exploits $f$'s
*exact* finite trigonometric-polynomial structure (`02_reverse_mode_ad.md` contrasts this
with finite differences, which approximate a *generic* smooth function and always carry
$O(h^2)$ truncation error; here there is no such term because $f$ genuinely has only one
frequency, so two evaluations suffice to solve for $B,C$ — equivalently, to exactly
determine $f'$ — with zero residual).

## Extending to $\partial_\theta$ over the full circuit

The argument above isolates one gate $U(\phi)$ inside a larger circuit $U(x,\theta) =
V_2\, U(\phi)\, V_1$ (everything before/after the shifted gate held fixed). $f(\phi) =
\langle 0\rvert V_1^\dagger U(\phi)^\dagger V_2^\dagger M V_2 U(\phi) V_1\lvert
0\rangle$ has exactly the same two-eigenvalue-generator structure (only $U(\phi)$
depends on $\phi$; $V_1,V_2$ are fixed unitaries, absorbed into $\lvert\psi\rangle$ and
$M$ respectively in the setup above), so the shift rule applies verbatim to *any single
trainable angle* $\theta_k$ inside a circuit of arbitrary size and depth: $\partial_{\theta_k}
f = \tfrac12[f(\theta_k+\tfrac\pi2) - f(\theta_k-\tfrac\pi2)]$, holding every other
parameter fixed. This licenses `pshift.py`'s `psr_grad_theta` (T2.7).

## $\partial_x$ is *also* a parameter-shift rule — a genuinely underused fact

An **encoding** gate $S(\omega x) = \exp(-i\omega x\, Z/2)$ has $\phi = \omega x$ — the
*input itself* enters the circuit as a gate angle, exactly like a trainable $\theta_k$.
By the chain rule, $\partial_x f = \omega\,\partial_\phi f\big|_{\phi=\omega x}$, so:
$$
\partial_x f = \omega\cdot\tfrac12\Big[f\big(x+\tfrac{\pi}{2\omega}\big) - f\big(x-\tfrac{\pi}{2\omega}\big)\Big]
$$
— the *same* shift rule, with the shift rescaled by $1/\omega$ to compensate the chain
rule. **This means a PINN's residual — which needs $\partial_x u$ (and, via the
second-order extension, T2.7, $\partial_x^2 u$) — can be evaluated *exactly on real
quantum hardware*, using only the same kind of circuit evaluations used to compute the
observable itself**, no classical automatic differentiation of the quantum part required.
This is the fact `project.md` §3 item 8 calls "a genuinely underused fact worth a
slide": most VQC literature applies parameter-shift only to $\theta$ (for training) and
falls back to finite differences or simulator-only autodiff for $\partial_x$ (needed by a
PINN's *residual*, not just its loss gradient) — but the input angle is not
mathematically different from a trainable angle in this respect, so the exact rule
applies to both, and `pshift.py`'s `psr_grad_input` (T2.7) implements exactly this.
