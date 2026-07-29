# Barren Plateaus: Exponential Gradient-Variance Decay, and Why This Project Avoids It

`project.md` §3 item 9.

## The phenomenon

For a parameterized circuit $U(\theta)$ drawn from a sufficiently random/expressive
ensemble (approximating the Haar measure on the unitary group, e.g. deep enough with
enough distinct gates), and a **global** cost function $C(\theta) = \langle 0\rvert
U^\dagger(\theta)\, M\, U(\theta)\lvert 0\rangle$ where $M$ acts non-trivially on *every*
qubit (e.g. $M = \lvert\phi\rangle\langle\phi\rvert$ for some target state, or a Pauli
string $Z_1\otimes Z_2\otimes\cdots\otimes Z_n$), McClean et al. (2018) show
$$
\mathbb{E}[\partial_{\theta_k} C] = 0, \qquad
\operatorname{Var}[\partial_{\theta_k} C] \;=\; O\!\left(\frac{1}{4^n}\right)
$$
— the gradient's typical magnitude vanishes **exponentially** in the number of qubits
$n$. A landscape with exponentially small gradient almost everywhere is a *barren
plateau*: gradient-based optimization needs exponentially many measurement shots (to
resolve a signal below the shot-noise floor) or exponentially many optimization steps to
make any progress, for even moderate $n$.

## The mechanism (why global cost + randomness $\Rightarrow$ exponential decay)

The rigorous derivation uses Weingarten calculus (integrating polynomials of unitary
matrix elements over the Haar measure) — a research-level computation not reproduced
in full here, but its *mechanism* is a standard, whiteboard-level concentration-of-measure
argument:

1. For $U$ Haar-random on the unitary group $U(2^n)$, the induced state $U\lvert
   0\rangle$ is (statistically) a uniformly random point on the $2^n$-dimensional complex
   unit sphere. A classical fact about high-dimensional spheres: a smooth function of a
   uniformly random point on a $D$-dimensional sphere concentrates around its mean with
   fluctuations $O(1/\sqrt D)$ (Lévy's lemma / measure concentration), and for the
   quadratic-form-type functions relevant here (expectation values $\langle\psi\rvert
   M\lvert\psi\rangle$ are quadratic in the state), the *variance* of such a fluctuation
   scales as $O(1/D)$.
2. Here $D = 2^n$ (the Hilbert space dimension), so $\operatorname{Var}[\ldots] \sim
   1/2^n$ for the *value* of a global observable; a more careful accounting of the
   *gradient* (which involves an additional derivative of a similarly-structured
   quadratic form, effectively squaring the relevant concentration exponent because the
   gradient itself is expressed via commutators/products of Haar-random blocks) gives the
   sharper $1/4^n = 1/(2^n)^2$ scaling McClean et al. derive exactly via Weingarten
   calculus. The qualitative conclusion — **variance decays exponentially in $n$, with
   the base of the exponential set by the Hilbert space dimension $2^n$** — is the
   reproducible part of the argument; the precise power ($4^n$ vs. $2^n$) is where the
   full calculation is needed.
3. **Why "global" matters.** If $M$ instead acts non-trivially on only $k\ll n$ qubits
   (a *local* observable, e.g. $M=Z_1$ alone), the argument above applies to the
   *reduced* state on those $k$ qubits, which lives in a $2^k$-dimensional space —
   *independent of $n$* for shallow enough circuits (where the reduced state on a small
   subsystem hasn't yet "seen" the full Haar-random structure of all $n$ qubits, i.e. the
   circuit's light-cone from the observable back to the inputs is still $O(1)$ in depth).
   The variance then does **not** decay exponentially in $n$ — Cerezo et al. (2021) prove
   this directly: for local cost functions and $O(\log n)$-depth circuits, the variance
   decays at worst polynomially.

## Why this project avoids it: local observable + shallow circuit + small-angle init

Three independent, standard mitigations, all present in this project's circuit
convention (T2.4):

1. **Local observable only.** `observable ∈ {"z0", "z_mean"}`, never a global Pauli
   string spanning all $n$ qubits — `ReuploadCircuit` raises on anything else (Alg. 1
   step 8). By point 3 above, this alone removes the *exponential-in-$n$* mechanism.
2. **Shallow, prescribed-depth circuits.** SMCD (T2.9) fixes $L$ from the *target
   spectrum's* resource bound (`07_circuit_fourier_spectrum.md` §4's depth rule), not
   from an unbounded search — $L$ stays small (single digits) by construction for every
   problem in this project's ladder, keeping the circuit within the shallow-circuit
   regime where locality's protection applies.
3. **Small-angle initialization.** `theta ~ N(0, 0.1^2)` (Alg. 1 step 10) rather than
   uniform-random angles: at $\theta\approx 0$ the circuit is close to the identity
   (or close to a fixed, structured state, since the encoding gates and the $R_Y(\pi/2)$
   preparation are *not* randomized), far from the Haar-random regime the barren-plateau
   argument assumes — gradients near a near-identity circuit are $O(1)$, not vanishing,
   and only shrink towards the barren-plateau estimate if training pushes $\theta$ into a
   genuinely random-looking regime (which small-angle init, combined with shallow depth,
   is specifically chosen to avoid).

`xai/gradvar.py`'s `gradient_variance`/`barren_plateau_fit` (T1.9) measure this directly:
re-initialize the circuit's parameters many times, measure the empirical
$\operatorname{Var}[\partial_\theta L]$, and fit $\log(\mathrm{var}) = a - b\cdot n$ — the
theoretical *worst-case* reference is $b=\log 2$ (i.e. $\mathrm{Var}\sim 2^{-n}$, matching
$4^{-n}$ up to the base-2-vs-base-4 convention used for the fit); this project's own
classical negative control (T1.9, `docs/plan/BENCH.md`) confirms the instrument itself
does not manufacture a plateau where none exists ($b=0.052$, far below $\log 2\approx
0.693$, for classical MLPs of increasing width) — the same instrument will be applied to
the real quantum circuits once T2.4 exists, to confirm the three mitigations above
actually keep $b$ well below the barren-plateau reference in practice, not just in
principle.
