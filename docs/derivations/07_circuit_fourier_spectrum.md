# Proposition 1 — The Fourier Spectrum of a Data-Re-Uploading Circuit

`project.md` §3 item 7 / §5.2 Prop. 1. Written by hand, first, before any quantum code
(T2.1) — the point of the exercise is to *derive* this, not summarize it.

Notation fixed throughout: qubit computational basis $\{\lvert 0\rangle, \lvert
1\rangle\}$, Pauli $Z\lvert 0\rangle = +\lvert 0\rangle$, $Z\lvert 1\rangle = -\lvert
1\rangle$. An encoding gate with scaling $\omega$ on the Pauli-$Z$ axis is
$$
S(\omega x) \;=\; \exp\!\left(-\,i\,\omega x\, \frac{Z}{2}\right) \;=\;
\operatorname{diag}\!\left(e^{-i\omega x/2},\, e^{+i\omega x/2}\right)
\quad\text{(diagonal in the } Z \text{ eigenbasis).}
$$
This is `RZ(ωx)` in the codebase's convention (`01_CONVENTIONS.md` §2). All trainable
gates are collected into single-qubit unitaries $W_l \in SU(2)$ (a *general* single-qubit
unitary needs three real parameters, e.g. the Euler decomposition $RZ(\alpha)RY(\beta)RZ(\gamma)$
— two gates, `RY` then `RZ` alone, span only a 2-parameter subset of the 3-parameter
group $SU(2)$, not all of it; §6 below shows this restriction is not actually what causes
the practical issue found in T2.6, but it is worth stating correctly regardless). Treating
$W_l$ below as an arbitrary $2\times 2$ unitary matrix is the general derivation; §6 notes
where the *specific* circuit implementation (T2.4) departs from full genericity in
practice.

---

## 1. One qubit, one layer

Circuit: $U(x,\theta) = W_1(\theta)\, S(\omega x)\, W_0(\theta)$, applied to $\lvert
0\rangle$, then measure $Z$: $f(x) = \langle 0 \rvert U^\dagger(x,\theta)\, Z\,
U(x,\theta) \lvert 0\rangle$.

Write $W_0\lvert 0\rangle = a\lvert 0\rangle + b\lvert 1\rangle$ (an arbitrary state,
since $W_0$ is an arbitrary single-qubit unitary applied to $\lvert 0\rangle$). Applying
the diagonal encoding gate:
$$
S(\omega x)\big(a\lvert 0\rangle + b\lvert 1\rangle\big)
= a\,e^{-i\omega x/2}\lvert 0\rangle + b\,e^{+i\omega x/2}\lvert 1\rangle .
$$
Write $W_1 = \begin{pmatrix} w_{00} & w_{01} \\ w_{10} & w_{11}\end{pmatrix}$. The final
state's amplitudes are
$$
c_0(x) = w_{00}\, a\, e^{-i\omega x/2} + w_{01}\, b\, e^{+i\omega x/2}, \qquad
c_1(x) = w_{10}\, a\, e^{-i\omega x/2} + w_{11}\, b\, e^{+i\omega x/2}.
$$
Expand $\lvert c_0(x)\rvert^2$:
$$
\lvert c_0(x)\rvert^2 = \lvert w_{00}a\rvert^2 + \lvert w_{01}b\rvert^2
+ w_{00}a\,(w_{01}b)^*\, e^{-i\omega x} + (w_{00}a)^*\, w_{01}b\, e^{+i\omega x}
= \lvert w_{00}a\rvert^2 + \lvert w_{01}b\rvert^2 + 2\,\mathrm{Re}\!\left[w_{00}a\,(w_{01}b)^* e^{-i\omega x}\right],
$$
using $e^{-i\omega x/2}\cdot \overline{e^{+i\omega x/2}} = e^{-i\omega x}$ and its
conjugate. The same expansion holds for $\lvert c_1(x)\rvert^2$ with its own constant and
coefficient. Since $f(x) = \lvert c_0(x)\rvert^2 - \lvert c_1(x)\rvert^2$, every term is
either constant in $x$ or proportional to $e^{\pm i\omega x}$:
$$
\boxed{f(x) = c_0(\theta) + c_{-\omega}(\theta)\, e^{-i\omega x} + c_{+\omega}(\theta)\, e^{+i\omega x}}
\qquad \text{i.e. frequencies } \{-\omega, 0, +\omega\}.
$$
($c_0$ is real since $f$ is real-valued and $c_{-\omega} = c_{+\omega}^*$ enforces that;
this is exactly $c_0 + 2\,\mathrm{Re}[c_{+\omega}e^{i\omega x}]$, a real trigonometric
polynomial as required.) No other frequency can appear: the encoding gate is the *only*
source of $x$-dependence, and it is diagonal with two eigenvalues, so every matrix element
of $U^\dagger Z U$ picks up a phase that is a difference of at most two possible half-turns
$\pm\omega/2$.

---

## 2. One qubit, $L$ layers (data re-uploading)

Now the encoding gate is re-applied at every layer, generally with a *different* scaling
$\omega_l$ (data re-uploading):
$$
U(x,\theta) = W_L\, S(\omega_L x)\, W_{L-1}\, S(\omega_{L-1} x)\, \cdots\, W_1\, S(\omega_1 x)\, W_0 .
$$
$S(\omega_l x)$ is diagonal in the $Z$-eigenbasis $\{\lvert +\rangle, \lvert -\rangle\}$
(shorthand for $\lvert 0\rangle,\lvert 1\rangle$ with eigenvalues $\lambda\in\{+1,-1\}$):
$S(\omega_l x) = \sum_{\lambda\in\{+1,-1\}} e^{-i\omega_l x \lambda/2}\lvert
\lambda\rangle\langle\lambda\rvert$. Insert this resolution at **every** occurrence of
$S(\omega_l x)$ in $U$, and independently at every occurrence of $S(\omega_l
x)^\dagger$ in $U^\dagger$ (the bra side). Writing $\lambda_l$ for the eigenvalue chosen
in the ket expansion at layer $l$ and $\lambda_l'$ for the (independent) eigenvalue chosen
in the bra expansion at layer $l$:
$$
f(x) = \langle 0\rvert U^\dagger M U \lvert 0\rangle
= \sum_{\{\lambda_l\}} \sum_{\{\lambda_l'\}} A\big(\{\lambda_l\},\{\lambda_l'\};\theta\big)
\; \exp\!\left(i x \sum_{l=1}^{L} \frac{\omega_l}{2}\big(\lambda_l' - \lambda_l\big)\right),
$$
where $A(\cdot)$ collects the (now $x$-independent) matrix elements of every $W_l$ along
that particular bra/ket path, plus the matrix elements of $M$ at the end — a pure function
of $\theta$. Define, per layer, $m_l := \tfrac{1}{2}(\lambda_l' - \lambda_l)$. Since
$\lambda_l,\lambda_l'\in\{+1,-1\}$, their difference is in $\{-2,0,+2\}$, so
$$
m_l \in \{-1, 0, +1\} \qquad \text{for every } l.
$$
The total exponent is $ix\sum_l \omega_l m_l$, i.e. every (bra-path, ket-path) pair
contributes to frequency $\omega = \sum_{l=1}^{L} m_l\,\omega_l$. Collecting all paths
that land on the same total frequency and summing their amplitudes gives the coefficient
$c_\omega(\theta)$:
$$
\boxed{f(x) = \sum_{\omega \in \Omega} c_\omega(\theta)\, e^{i\omega x}, \qquad
\Omega = \left\{ \sum_{l=1}^{L} m_l\,\omega_l \;:\; m_l \in \{-1,0,+1\} \right\}.}
$$
This is exactly Prop. 1's statement specialised to a single Pauli-$Z$-generated qubit: the
frequency set is the set of all *signed subset sums* of the per-layer scalings — because
the bra and ket each independently traverse the encoding gates, and their per-layer
eigenvalue difference is confined to a signed unit (times $\omega_l$), never a fraction of
it (§5, the factor-of-2 note, makes this last point precise).

---

## 3. General statement (Schuld–Sweke–Meyer 2021), multi-qubit / multi-dimensional

The one-qubit derivation above generalizes with no new idea, only more bookkeeping:

**General generator.** If layer $l$'s encoding gate is $\exp(-ix\,H^{(l)})$ for a
Hermitian $H^{(l)}$ with eigendecomposition $H^{(l)} = \sum_a \lambda_a^{(l)} \lvert
a\rangle\langle a\rvert$ (not necessarily a single qubit — $H^{(l)}$ may act on the full
$n$-qubit Hilbert space), the same bra/ket eigenbasis expansion applies with $\lambda_l
\to \lambda_{a_l}^{(l)}$ (ket path) and $\lambda_l' \to \lambda_{b_l}^{(l)}$ (bra path),
ranging over the *full* eigenvalue spectrum of $H^{(l)}$ rather than just $\{\pm 1\}$:
$$
f(x) = \sum_{\omega\in\Omega} c_\omega(\theta)\, e^{i\omega x}, \qquad
\Omega = \left\{ \sum_{l=1}^{L} \big(\lambda_{a_l}^{(l)} - \lambda_{b_l}^{(l)}\big) \right\}
$$
— exactly `project.md`'s statement of Prop. 1, $\Omega = \{\sum_l (\lambda_a^{(l)} -
\lambda_b^{(l)})\}$. Nothing in the derivation used $H^{(l)}$ having only two eigenvalues;
that restriction only mattered for pinning $m_l\in\{-1,0,1\}$ in the Pauli-$Z$ special
case above.

**Multi-qubit.** For $n$ wires, if wire $q$'s encoding generator at layer $l$ is a *local*
Pauli $Z_q$ scaled by $\omega_{l,q}$ (the codebase's convention — one `RZ` per wire per
layer, T2.4), the total encoding generator at layer $l$ is $H^{(l)} = \sum_q \omega_{l,q}
Z_q/2$, whose eigenvalues are sums $\sum_q \omega_{l,q}\, s_q/2$ over sign choices
$s_q\in\{+1,-1\}$ (one per wire — the generator is a sum of commuting local terms, so its
eigenbasis is the product computational basis and its eigenvalues are sums of the local
eigenvalues). Repeating the bra/ket argument per wire, the realised frequency set becomes
a sum over per-wire, per-layer signed contributions:
$$
\Omega = \left\{ \sum_{l=1}^L \sum_{q=1}^n m_{l,q}\,\omega_{l,q} \;:\; m_{l,q}\in\{-1,0,+1\} \right\}.
$$
*Without entanglement* (no gate correlates different wires' amplitudes before the
observable is measured), the observable factorises across wires and the cross terms
between *different* wires' $m_{l,q}$ choices cancel in the reduced state of the measured
wire — only that wire's own $\sum_l m_l \omega_l$ survives. *With* an entangler (e.g.
`ring_cz`, T2.4), the wires' phases become genuinely correlated before measurement, and
sums that mix two different wires' scalings — $\omega_{x}\!\pm\!\omega_{y}$ for wires
encoding different input coordinates — appear in the single scalar observable. This is
precisely what `tests/test_circuit_spectrum.py` (T2.6) checks directly: `entangler="none"`
on wire 0's observable shows *only* wire-0 frequencies; `entangler="ring_cz"` shows cross
terms.

**Multi-dimensional input.** When wire $q$ encodes coordinate $x_{\mathrm{wire\_to\_dim}(q)}$
of a vector input $x\in\mathbb{R}^d$ rather than a shared scalar $x$, each wire's phase
$\exp(-i\,\omega_{l,q}\, x_{\mathrm{wire\_to\_dim}(q)}/2)$ contributes independently to
*its own coordinate axis* of a frequency vector $n\in\mathbb{Z}^d$ (or $\mathbb{R}^d$ for
non-integer scalings), and the theorem becomes a genuinely multivariate trigonometric
polynomial:
$$
f(x) = \sum_{n\in\Omega} c_n(\theta)\, e^{i\langle n, x\rangle}, \qquad x\in\mathbb{R}^d,\; n\in\mathbb{R}^d,
$$
with $\Omega\subset\mathbb{R}^d$ built the same way, componentwise, from the per-wire
scalings that encode each coordinate. This is the form actually used throughout this
project (P1 is the $d=1$ special case worked above; P4 is $d=2$).

---

## 4. The ternary corollary and the corrected depth rule (D5)

Set $\omega_l = \Delta \cdot 3^{l-1}$ for $l=1,\dots,L$ (single wire, so $\Omega\subset
\mathbb{R}$ as in §2). Then
$$
\Omega = \left\{ \Delta \sum_{l=1}^{L} m_l\, 3^{l-1} \;:\; m_l\in\{-1,0,+1\} \right\}.
$$

**Claim: the map $(m_1,\dots,m_L)\mapsto N:=\sum_{l=1}^L m_l\,3^{l-1}$ is a bijection onto
the integers in $\left[-\tfrac{3^L-1}{2},\,\tfrac{3^L-1}{2}\right]$** — the *balanced
ternary* representation.

*Proof.* Substitute $m_l = d_l - 1$ with $d_l \in \{0,1,2\}$ (a relabelling, bijective
per digit). Then
$$
N = \sum_{l=1}^L (d_l - 1)\,3^{l-1} = \underbrace{\sum_{l=1}^L d_l\, 3^{l-1}}_{=:D} - \sum_{l=1}^L 3^{l-1}
= D - \frac{3^L - 1}{2},
$$
using $\sum_{l=1}^L 3^{l-1} = \frac{3^L-1}{2}$ (geometric series, base 3). $D$ is an
*ordinary* base-3 number with $L$ digits $d_l\in\{0,1,2\}$: the map
$(d_1,\dots,d_L)\mapsto D$ is the standard positional-base-3 representation, a bijection
onto the integers $D\in[0, 3^L-1]$ (every integer in that range has exactly one base-3
digit expansion of length $L$, allowing leading zeros). Composing with the affine shift
$N = D - \frac{3^L-1}{2}$ (a bijection on integers) gives that $N$ ranges bijectively over
$\left[-\tfrac{3^L-1}{2},\, 3^L-1-\tfrac{3^L-1}{2}\right] = \left[-\tfrac{3^L-1}{2},\,
\tfrac{3^L-1}{2}\right]$. $\blacksquare$

Hence $\Omega = \Delta\cdot\{-\tfrac{3^L-1}{2},\dots,-1,0,1,\dots,\tfrac{3^L-1}{2}\}$ with
**exactly** $3^L$ distinct values (no degeneracy — every $m$-tuple gives a distinct
frequency, unlike the linear-scaling case $\omega_l=\omega_0$ below) and
$\max\lvert\Omega\rvert = \Delta\,\tfrac{3^L-1}{2}$.

**Worked arithmetic, $L=1,2,3,4$** (verified numerically by brute-force enumeration over
all $3^L$ sign tuples before writing this down):

| $L$ | $3^L$ | $\max\lvert\Omega\rvert/\Delta = \tfrac{3^L-1}{2}$ | $\Omega/\Delta$ (explicit) |
|---|---|---|---|
| 1 | 3  | 1  | $\{-1, 0, 1\}$ |
| 2 | 9  | 4  | $\{-4,-3,-2,-1, 0, 1,2,3,4\}$ |
| 3 | 27 | 13 | $\{-13,-12,\dots,-1,0,1,\dots,12,13\}$ (all 27 integers in $[-13,13]$) |
| 4 | 81 | 40 | $\{-40,-39,\dots,-1,0,1,\dots,39,40\}$ (all 81 integers in $[-40,40]$) |

(The $L=3$ row, $\Delta=\pi$, gives exactly the 27 values $\pi\cdot\{-13,\dots,13\}$ that
`ReuploadCircuit.frequencies()` must reproduce numerically — T2.4's DoD.)

**Contrast: linear scalings.** If instead $\omega_l = \omega_0$ for every $l$ (no
per-layer growth), $\Omega = \omega_0\cdot\{\sum_l m_l : m_l\in\{-1,0,1\}\}=
\omega_0\cdot\{-L,\dots,L\}$ — only $2L+1$ distinct frequencies (heavily degenerate: many
$m$-tuples collide on the same sum), growing *linearly* in depth, versus the ternary
scaling's $3^L$ growing *exponentially* in depth. This is the design knob `project.md` §5.2
flags as the one almost every prior QPINN paper leaves at 1 (i.e. uses linear/constant
scalings by default, without noticing the exponential alternative).

**The corrected depth rule (D5).** To guarantee the circuit's reachable band covers a
target maximum frequency $K$ at ternary spacing $\Delta$, require
$$
\max\lvert\Omega\rvert \ge K
\iff \Delta\,\frac{3^L-1}{2} \ge K
\iff 3^L \ge \frac{2K}{\Delta}+1
\iff L \ge \log_3\!\left(\frac{2K}{\Delta}+1\right)
\;\Longrightarrow\;
\boxed{L = \left\lceil \log_3\!\left(\frac{2K}{\Delta}+1\right) \right\rceil.}
$$
`project.md` Algorithm 1 step 4 instead states $L\leftarrow\lceil\log_3(K/\Delta)\rceil$
— missing the factor of 2 inside the log and the $+1$ term entirely. **Worked check, P1**
($K=15\pi$, $\Delta=\pi$, matching P1's target spectrum $\{\pi, 15\pi\}$, T2.8):
$$
\frac{2K}{\Delta}+1 = 2(15)+1 = 31, \qquad \log_3(31) = 3.048\ldots, \qquad L = \lceil 3.048\rceil = 4.
$$
The spec's uncorrected formula gives $\lceil\log_3(15)\rceil = \lceil 2.46\rceil = 3$,
whose reachable band is only $\max\lvert\Omega\rvert = \Delta\cdot\tfrac{3^3-1}{2} =
13\pi < 15\pi = K$ — it **silently misses the target mode** (the $L=3$ row of the table
above confirms $13\pi$ is indeed the reachable maximum, one short of $15\pi$). This is
exactly the failure the phase doc's D5 note warns about, now derived rather than merely
asserted.

---

## 5. The factor-of-2 note

Two different "factors of 2" appear in this derivation and must not be confused:

1. **Generator eigenvalues are half the scaling.** `RZ(ωx) = exp(-iωxZ/2)`: the
   *generator* is $\omega Z/2$, whose eigenvalues are $\pm\omega/2$ — not $\pm\omega$.
   Someone reading only "the generator's eigenvalues" off the gate and treating that
   number as *the* realised frequency would report $\omega/2$, half of the truth.
2. **Realised frequencies are eigenvalue *differences*, which restores the full
   scaling.** As §1–§2 derive explicitly, the $x$-dependence in $\langle Z\rangle$ comes
   from *products of two* encoding-gate matrix elements (one from the bra, one from the
   ket), whose phases are $e^{\mp i\omega x/2}$ each — so their **difference** appears in
   the combined exponent, $e^{-i\omega x/2}\cdot\overline{e^{+i\omega x/2}} = e^{-i\omega
   x}$, a *full* $\omega$, not $\omega/2$. Concretely: $\lambda,\lambda'\in\{+1,-1\}$ give
   differences $\lambda-\lambda'\in\{-2,0,+2\}$, and $m_l = \frac12(\lambda-\lambda')$
   multiplies the *scaling* $\omega_l$ (not $\omega_l/2$) in the final exponent — the two
   factor-of-2's (half from the generator, double from the eigenvalue-difference pairing)
   cancel exactly, and the realised frequency set is $\{-\omega,0,+\omega\}$, at the gate's
   own scaling, not half of it.

**Getting this wrong halves every frequency in the project.** This is precisely the
quantity `xai/spectral_error.py`'s `pauli_freq(omega_scaling) -> float` pins down in code
— it returns `omega_scaling` *unchanged*, not `omega_scaling / 2`, exactly because the
realised frequency equals the encoding gate's own scaling once the eigenvalue-difference
structure above is accounted for. `tests/test_fft_convention.py` (T0.9) and
`tests/test_circuit_spectrum.py` (T2.6) both test this numerically, on the actual FFT of
a circuit's output, so this note is not merely a warning — it is a claim with a
machine-checked witness.

---

## 6. A practical corollary found while verifying this proposition (T2.6)

Prop. 1 states which frequencies $\omega\in\Omega$ *can* appear — it does not, by itself,
guarantee every $\omega\in\Omega$ appears with *nonzero* amplitude $c_\omega(\theta)$ for
every circuit implementation and every $\theta$. `T2.4`'s circuit prepends a fixed
$RY(\phi_{\text{prep}})$ to move the qubit off $\lvert 0\rangle$ before the first encoding
gate (§1's derivation needs $a\ne 0\ne b$; $RZ$ on $\lvert 0\rangle$ is only a global
phase). Verified directly against `tests/test_circuit_spectrum.py`'s FFT check (and cross-
checked bit-for-bit against the PennyLane oracle, T2.5, so this is not a simulator
artifact): choosing $\phi_{\text{prep}}=\pi/2$ **exactly** ($a=b=1/\sqrt2$, an equal
superposition) makes every ternary-scaled frequency whose balanced-ternary digit $m_1=0$
— exactly one third of $\Omega$ — have **identically zero** amplitude, for *every*
$\theta$, independent of how general the trainable block is (checked with the 2-parameter
`RY`-then-`RZ` block T2.4 specifies, a full 3-parameter Euler block, and a 4-parameter
`RX`-`RY`-`RZ` block — same result each time). A perturbation of $\phi_{\text{prep}}$ by
as little as $0.1$ rad away from $\pi/2$ (or, more simply, any other value such as
$\pi/3$) removes the degeneracy entirely.

This matters beyond a test threshold: Prop. 3 (`project.md` §5.2, "if $S_\varepsilon
\subseteq \Omega$ the hybrid can represent the solution with a linear head") would be
**false** for any target spectrum touching one of these dead frequencies under the
literal $\phi_{\text{prep}}=\pi/2$ prescription — a genuine gap in the constructive
guarantee, not merely a numerical inconvenience. `src/qapinn/models/circuits.py` now uses
$\phi_{\text{prep}}=\pi/3$ (`PREP_ANGLE`), which was verified to have zero such dead
frequencies. The exact value $\pi/3$ is not special beyond avoiding the two known
degenerate points ($\phi_{\text{prep}}\in\{0,\pi/2,\pi\} \pmod \pi$); any other choice
away from those would work equally well.
