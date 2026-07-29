# Qubits, Single- and Two-Qubit Gates, and the Expectation-Value Formula

`project.md` §3 item 6. Sets up the notation `07_circuit_fourier_spectrum.md` (T2.1) uses.

## Qubit state space

A single qubit's state is a unit vector in $\mathbb{C}^2$, written in the computational
basis $\{\lvert 0\rangle, \lvert 1\rangle\}$ (eigenbasis of Pauli $Z$: $Z\lvert
0\rangle=+\lvert 0\rangle$, $Z\lvert 1\rangle=-\lvert 1\rangle$) as $\lvert\psi\rangle =
\alpha\lvert 0\rangle+\beta\lvert 1\rangle$, $\lvert\alpha\rvert^2+\lvert\beta\rvert^2=1$.
An $n$-qubit state lives in $(\mathbb{C}^2)^{\otimes n}$, dimension $2^n$ (this codebase's
`qsim.py`, T2.3, represents this as a tensor of shape $[B,2,2,\dots,2]$ — batch dimension
plus $n$ trailing binary axes, one per qubit, rather than a flattened $2^n$-vector, so
single-qubit gates act as local tensor contractions without ever materializing a
$2^n\times 2^n$ matrix).

## Single-qubit rotation gates

The Pauli matrices $X=\begin{pmatrix}0&1\\1&0\end{pmatrix}$,
$Y=\begin{pmatrix}0&-i\\i&0\end{pmatrix}$, $Z=\begin{pmatrix}1&0\\0&-1\end{pmatrix}$
generate rotations $R_P(\phi) = \exp(-i\phi P/2)$. Since $P^2=I$ for any Pauli $P$, the
matrix exponential truncates exactly (as for any involution):
$$
\exp(-i\phi P/2) = \cos(\phi/2)\, I - i\sin(\phi/2)\, P
$$
(Taylor-expand $\exp(-i\phi P/2)=\sum_k \frac{(-i\phi/2)^k}{k!}P^k$, split into even/odd
$k$; $P^{2m}=I$, $P^{2m+1}=P$ collapse the two sums into $\cos$ and $-i\sin$ series
exactly). Explicitly:
$$
R_Z(\phi) = \begin{pmatrix} e^{-i\phi/2} & 0 \\ 0 & e^{i\phi/2}\end{pmatrix}, \qquad
R_Y(\phi) = \begin{pmatrix}\cos(\phi/2) & -\sin(\phi/2) \\ \sin(\phi/2) & \cos(\phi/2)\end{pmatrix}, \qquad
R_X(\phi) = \begin{pmatrix}\cos(\phi/2) & -i\sin(\phi/2)\\ -i\sin(\phi/2) & \cos(\phi/2)\end{pmatrix}.
$$
$R_Z$ is diagonal in the computational basis (used throughout
`07_circuit_fourier_spectrum.md`'s derivation); $R_X,R_Y$ are not, and are exactly the
gates that move amplitude between $\lvert 0\rangle$ and $\lvert 1\rangle$ — this is why
`07_circuit_fourier_spectrum.md`'s $W_l$ (built from $R_Y,R_Z$) can turn a diagonal
encoding gate into a genuine $x$-dependent interference pattern: a state that is purely
$\lvert 0\rangle$ before an $R_Z$ picks up only a global phase (unobservable), which is
exactly why T2.4's circuit design prepends $R_Y(\pi/2)$ on every wire before the first
encoding gate — to put the qubit into a genuine superposition first.

## Two-qubit entangling gates

Acting on wires $(q_0,q_1)$ within an $n$-qubit register:

- **CNOT** (control $q_0$, target $q_1$): flips $q_1$ iff $q_0=\lvert 1\rangle$. On the
  ordered basis $\{\lvert00\rangle,\lvert01\rangle,\lvert10\rangle,\lvert11\rangle\}$:
  $\mathrm{CNOT} = \begin{pmatrix}1&0&0&0\\0&1&0&0\\0&0&0&1\\0&0&1&0\end{pmatrix}$ — identity
  on the $q_0=0$ subspace, Pauli-$X$ on $q_1$ within the $q_0=1$ subspace.
- **CZ**: applies a $-1$ phase iff *both* qubits are $\lvert 1\rangle$, identity
  otherwise: $\mathrm{CZ}=\operatorname{diag}(1,1,1,-1)$ — purely diagonal, so applying it
  is just multiplying the $\lvert 11\rangle$ amplitude sub-block by $-1$, no matrix
  multiply needed (`qsim.py`'s `apply_cz`, T2.3).

Both are used to correlate the phases picked up by different wires' encoding gates before
measurement — without an entangler, each wire's contribution to the measured observable
factorizes and stays independent (per-wire frequencies only); with `ring_cz` (a CZ
between wire $q$ and $q{+}1\bmod n$ for every $q$), different wires' encoding phases
become genuinely correlated and *cross-frequency* terms ($\omega_x\pm\omega_y$) appear in
a single measured observable — this is precisely
`07_circuit_fourier_spectrum.md`'s §3 point, tested directly by
`tests/test_circuit_spectrum.py` (T2.6).

## The expectation-value formula

A circuit is a unitary $U(x,\theta)$ built from a fixed sequence of gates (some
$x$-dependent — the encoding gates; some $\theta$-dependent — the trainable gates),
applied to a fixed initial state $\lvert 0\rangle^{\otimes n}$. Measuring an observable
$M$ (Hermitian; a Pauli string or sum of Pauli strings) reports its **Born-rule
expectation value**:
$$
f(x) = \langle 0\rvert U^\dagger(x,\theta)\, M\, U(x,\theta) \lvert 0\rangle
= \langle\psi(x,\theta)\rvert M \lvert\psi(x,\theta)\rangle, \qquad \lvert\psi(x,\theta)\rangle := U(x,\theta)\lvert 0\rangle .
$$
This is real-valued for Hermitian $M$ (standard fact: $\langle\psi\rvert M\lvert\psi\rangle^*
= \langle\psi\rvert M^\dagger\lvert\psi\rangle = \langle\psi\rvert M\lvert\psi\rangle$ using
$M=M^\dagger$), so $f(x)$ is a genuine real scalar output usable as a PINN's predicted
value $u_\theta(x)$ (or a component feeding into a classical head, this project's hybrid
models, T2.12). This is exactly the object
`07_circuit_fourier_spectrum.md` expands in $x$ to derive Prop. 1, and exactly what
`qsim.py`'s `expval_z`/`expval_z_mean` compute (T2.3), and what `pshift.py`'s
parameter-shift rule (T2.7, `08_parameter_shift.md`) differentiates.
