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

## T2.7 extension: the second-order rule for a single gate

We want $f''(\phi)$ where $f(\phi) = A + B\cos\phi + C\sin\phi$ (exact, from §1). Evaluate
$f$ at $\phi \pm s$ for a shift $s$ to be chosen, and use $\cos(\phi\pm s) + \cos(\phi\mp
s)$-type sum identities — concretely, $\cos(\phi+s)+\cos(\phi-s)=2\cos\phi\cos s$ and
$\sin(\phi+s)+\sin(\phi-s)=2\sin\phi\cos s$ — to get
$$
f(\phi+s) + f(\phi-s) = 2A + 2\cos(s)\big(B\cos\phi + C\sin\phi\big) = 2A(1-\cos s) + 2\cos(s)\, f(\phi).
$$
We also have $f''(\phi) = -B\cos\phi - C\sin\phi = A - f(\phi)$ directly (since $f$ solves
the SHM-type identity $f''+f=A$ for a pure single-frequency-plus-constant sinusoid). We
want to write $f''(\phi)$ as a fixed linear combination $a\big[f(\phi+s)+f(\phi-s)\big] +
b\, f(\phi)$ that holds for **every** $A,B,C$ (i.e. for any state/observable, not just
this one) — matching coefficients of the (fixed, $\phi$-independent) constant $A$ and of
the $\phi$-varying part $f(\phi)$ separately:
$$
2a(1-\cos s) = 1, \qquad 2a\cos s + b = -1.
$$
Choosing $s=\pi/2$ (the **same** shift as the first-order rule) gives $\cos s = 0$, so
$a = 1/2$ and $b=-1$:
$$
\boxed{f''(\phi) = \tfrac12\big[f(\phi+\tfrac\pi2) - 2f(\phi) + f(\phi-\tfrac\pi2)\big].}
$$
This is the standard exact "diagonal" second-derivative shift rule for a two-eigenvalue
generator (it reuses the *same* $\pi/2$ shift already needed for $\partial_\phi f$, not a
different one).

**Correcting `04_PHASE2_theory_smcd.md`'s literal T2.7 formula.** The phase doc states
$\partial_x^2 f = (\omega^2/2)[f(x+\pi/\omega) - 2f(x) + f(x-\pi/\omega)]$ — shift
$\pi/\omega$ (i.e. $s=\pi$ in $\phi$-space), same coefficient $\omega^2/2$. Substituting
$s=\pi$ into the two matching-coefficient equations above gives $a = 1/(2(1-\cos\pi)) =
1/4$, **not** $1/2$ — the phase doc's formula is off by an exact factor of 2, confirmed
numerically (`A,B,C,\omega` drawn at random: literal formula returns exactly `2x` the true
$\partial_x^2 f$ every time). This is the same class of bug as T1.7's
`effective_dimension` formula (`10_effective_dimension.md`): a plausible-looking closed
form in the phase doc that doesn't survive being checked against the thing it claims to
compute. The correct rule, with $\phi=\omega x$ and $d^2\phi/dx^2=0$ so
$\partial_x^2 f = \omega^2 \partial_\phi^2 f$:
$$
\partial_x^2 f = \frac{\omega^2}{2}\Big[f\big(x+\tfrac{\pi}{2\omega}\big) - 2f(x) + f\big(x-\tfrac{\pi}{2\omega}\big)\Big]
$$
— shift $\pi/(2\omega)$, the *same magnitude* as the first-order rule's shift, not
$\pi/\omega$. `pshift.py`'s `psr_grad2_input` implements this corrected form.

## The multi-gate case: re-uploading introduces cross terms at second order

Section "Extending to $\partial_\theta$" showed the *first*-order rule composes additively
across gates with no cross terms — that's just linearity of the derivative. Second order
is different. Under re-uploading, $x$ (via a fixed dimension) typically feeds **several**
encoding gates $\phi_1,\dots,\phi_k$ (one per layer that re-encodes it, times one per wire
assigned to that dimension), each $\phi_i = \omega_i x$ linear in $x$. By Prop. 1
(`07_circuit_fourier_spectrum.md`), $f$ as a function of $(\phi_1,\dots,\phi_k)$ jointly is
an exact multivariate trigonometric polynomial with frequency components in
$\{-1,0,1\}^k$ — which means **mixed** partials $\partial^2 f/\partial\phi_i\partial\phi_j$
($i\neq j$) need not vanish. Since each $\phi_i$ is linear in $x$ ($d^2\phi_i/dx^2=0$),
the multivariable chain rule gives exactly
$$
\frac{d^2f}{dx^2} = \sum_i \omega_i^2\,\frac{\partial^2 f}{\partial\phi_i^2} + \sum_{i\neq j}\omega_i\omega_j\,\frac{\partial^2 f}{\partial\phi_i\,\partial\phi_j}.
$$
The diagonal terms are the single-gate rule above. The mixed terms are exact too, by the
same composability argument used for $\partial_\theta$: since $f(\phi_i,\cdot)$ (for fixed
$\phi_j$) is of the single-frequency form in $\phi_i$, the first-order shift rule applies
to it for *any* fixed $\phi_j$; applying it again in $\phi_j$ to the resulting function
(itself single-frequency in $\phi_j$, since it's a linear combination of two such
functions) gives the exact four-point rule
$$
\frac{\partial^2 f}{\partial\phi_i\,\partial\phi_j} = \tfrac14\Big[f_{++} - f_{+-} - f_{-+} + f_{--}\Big],
$$
where $f_{\pm\pm}$ shifts $\phi_i,\phi_j$ independently by $\pm\pi/2$, all other gates
held fixed. **Dropping the mixed terms is wrong whenever more than one encoding-gate
instance reads the same input dimension** — i.e. essentially always for $L>1$, since
re-uploading re-encodes $x$ every layer by construction. `pshift.py`'s `psr_grad2_input`
sums both the diagonal and the mixed terms over all pairs of encoding gates reading the
requested dimension; `tests/test_pshift.py` exercises `n_layers>1` specifically so this
isn't silently untested.
