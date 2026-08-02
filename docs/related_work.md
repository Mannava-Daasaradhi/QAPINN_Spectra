# Related work positioning (T2.13)

**Note on reference [4].** The original project brief cites a `[4]` behind a Scribd link
that is login-walled and not fetchable by anyone working on this repo, including the
author. Per `project.md` §1's own documented fallback ("if it has not arrived by the
start of Phase 3, position against the inferred candidates... and add a visible note that
the reference was inaccessible"), and Phase 3 having started (T3.1, T3.2 committed)
without the real citation arriving, this document positions against the inferred
candidate below instead. If the real reference [4] is supplied later, this paragraph
should be revised to cite it directly.

## Differentiation

The closest identified prior work is *Hybrid quantum-classical PINNs for nonlinear PDEs:
when and where is hybridization effective?* (arXiv:2606.04679), with a likely earlier
lineage in *Hybrid quantum physics-informed neural networks* (Mach. Learn.: Sci. Technol.,
`10.1088/2632-2153/ad43b2`). That line of work characterizes **when** hybridization helps,
empirically: it runs classical-vs-hybrid PINN comparisons across problem settings and
reports where the hybrid model wins or loses, without a principled account of *why* or a
procedure for designing the quantum circuit itself. This project instead gives a
**constructive** design rule (SMCD: map a PDE's operator symbol and forcing/BC spectra
directly to an encoding-gate schedule, re-uploading depth, and qubit count, with no
search) plus an **XAI mechanism** (NTK block decomposition and per-frequency error
tracking) that explains the *why* — the hybrid model wins specifically when its encoded
frequency set covers the PDE's own high-frequency content, and the encoded band is
directly visible in the NTK spectrum and the per-frequency error trajectory. Where prior
work reports an empirical win/loss table, this project reports a design rule that predicts
the table's outcome in advance (T3.1's pre-registered predictions) and an instrument set
that shows the mechanism, not just the result.
