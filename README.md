# The Birth of Aeon

The raw working files behind Aeon — Colab notebooks, training-run artifacts, and research
documents — published as-is.

This is an **archive, not a product**. Nothing here has been cleaned up, refactored, or
rewritten for an audience. It is the actual trail of work: dead ends, duplicated notebooks,
`Copy of Copy of` filenames, empty scratch files and all. That mess is the point. It is
what the process actually looked like.

## What this archive is

> **Scope.** The bulk of this repository is the DPPU/VRU lineage — v2 through v23 — that
> *precedes* Aeon. Aeon itself is the proof-of-concept package in [`poc/`](./poc); the
> notebooks, training runs and documents are its prehistory. Where the two disagree, `poc/`
> is current and the documents are historical record.
>
> **`VRU` = Vitruvian Recurrent Unit** throughout. The architecture diagram's "Variable
> Recurrent Unit" is a one-off variant; see [`ERRATA.md`](./ERRATA.md) E5.


Two architectures live here, and the second one **refutes the first**. Reading the
documents without knowing that will mislead you.

**The VRU (Vitruvian Recurrent Unit)**, formerly the **DPPU (Dynamic Pi-Phi Processing
Unit)**, is built on a fixed geometric constant:

```
phi = 4/pi = 1.2732...      alpha = pi/4 = 0.7854...      phi x alpha = 1
```

`phi` is the circle-square ratio derived from Da Vinci's Vitruvian Man — hence the name.
It scales the recurrent weight path, with its reciprocal dual `alpha` on the input
projection, and the claim was that this proportion alone holds gradients stable across
arbitrarily long sequences. The geometry is set out in
[`docs/foundations/`](./docs/foundations), written a year before any of the code.
Documents 1–9 in [`docs/vru-architecture/`](./docs/vru-architecture) build the case and
then take it apart: document 8 fails to prove the mechanism, document 9 finds where it
breaks.

**Aeon** is what came next, and it drops `phi` entirely. From the canonical definition in
[`poc/aeon/recursion.py`](./poc/aeon/recursion.py):

```
BANNED
------
* No phi = 4/pi anywhere. Refuted by Delta-1 ablation.
* No "purge" / circuit breaker doing load-bearing stability work.
  Stability is the parameterization, not a safety net.
```

In its place is something provable. The recurrent map is parameterized so its largest
singular value is **below 1 by construction** — `W = sigmoid(s) · MARGIN · Cayley(A) ·
diag(tanh(d))` — rather than clipped, gated, or hoped for. As the module puts it: "There is
no setting of the parameters that violates it." Contraction then follows from Banach's
fixed point theorem: perturbations decay exponentially and numerical error cannot compound.

That is the arc this repository records — a geometric intuition, pursued for fourteen
months, falsified by its author's own ablation, and replaced by a mechanism that carries a
proof. The founding idea did not survive. The discipline that killed it is the result.

## What's in here

| Path | Contents |
|---|---|
| [`notebooks/`](./notebooks) | 96 Google Colab notebooks. See [`notebooks/INDEX.md`](./notebooks/INDEX.md) for a grouped index. |
| [`runs/dppu_vru/`](./runs/dppu_vru) | Training logs, checkpoints and metadata for runs **v17–v23**. |
| [`docs/foundations/`](./docs/foundations) | The January 2025 origin papers — where `phi` and `pi_dyn` come from. |
| [`docs/vru-architecture/`](./docs/vru-architecture) | The core research series — what the architecture is and why it works. |
| [`docs/vru-trading-bot/`](./docs/vru-trading-bot) | A complete six-part series applying it to markets. |
| [`poc/`](./poc) | **Aeon itself** — the working proof-of-concept package. |
| [`raw/`](./raw) | Verbatim source dumps that are not notebooks. |

## Documents

Two **separate** numbered series live here. Both have a document 6, which is why they sit
in different folders.

### `docs/foundations/` — the origin papers (January 2025)

Five documents from **January 2025**, fourteen months before the architecture work. This
is where the constants come from, and they answer the question the later documents leave
open.

| Document | Date |
|---|---|
| Dynamic Pi Transformation | 29 Jan 2025 |
| Dynamic Phi Transformation | 30 Jan 2025 |
| Unified Master Equation for pi-dynamic and phi-dynamic | 30 Jan 2025 |
| LaTeX Formulas for Major Equations | 30 Jan 2025 |
| Dimensional Math — Symbol Glossary | undated |

*Dynamic Pi Transformation* is the origin. Taking Da Vinci's Vitruvian Man, it posits two
centres rather than one — a physical centre at the stomach and a perceptual centre in the
mind — and a folding factor `Δ` shifting between them. From that dual-centre model it
argues that `pi` should be treated as a function rather than a fixed constant, approaching
4 under folding, which is the sense in which the circle gets squared. *Dynamic Phi*
(which carries a second co-author byline) applies the same move to the golden ratio, and
the *Master Equation* combines them as `Ω(Δ) = pi_dynamic · phi_dynamic`.

**These are speculative theoretical papers, not peer-reviewed results.** They propose
modifications to the Schwarzschild radius, Bekenstein-Hawking entropy, the Einstein
lensing radius and the Schrödinger equation. The archive presents them as the recorded
origin of the idea; the empirical claims elsewhere in this repository stand on their own
experiments, not on these.

**Where they connect to the code.** The operators in `raw/HZ.txt` are these equations,
literally:

| Paper | Code (`raw/HZ.txt`) |
|---|---|
| `pi_dynamic = 4 - (4 - pi)e^(-Δ)` | `def pi_dyn(delta): return 4.0 - (4.0 - PI_CL) * torch.exp(-delta)` |
| `Ω(Δ) = pi_dynamic · phi_dynamic` | `def omega(delta): return (pi_dyn(delta) * phi_dyn(delta)) / (1.0 + delta + EPS)` |

**One discrepancy worth naming.** The `phi` of these papers is the *classical golden
ratio*, 1.618. The `phi` of the VRU architecture is `4/pi`, 1.273. Same letter, different
number — and reading the two bodies of work together, this is the easiest thing to trip
over.

Document 1 resolves part of this: within the field system it *defines* `phi_classical` as
`4/pi`, so the architecture is internally consistent on its own terms.

The two bodies of work are not reconciled, however. The `phi` of the January 2025 papers is
the **golden ratio, 1.618**. The `phi` of the VRU architecture is **4/pi, 1.273**. They are
different numbers under the same letter, and **this archive does not contain a derivation
linking them.** A constant in `raw/HZ.txt` (`GOLDEN_DECAY_TIME`, line 17) converts between
them, but it is a decay time restating its own definition rather than a derivation — see
[`ERRATA.md`](./ERRATA.md) E4.

### `docs/vru-architecture/` — the core research

Documents **1–9, complete**. Document 6 records the rename from DPPU to VRU; documents
1–5 used the DPPU designation. Document 9 names a Document 10 as planned but not yet
written.

| # | Document | What it does |
|---|---|---|
| 01 | Origin | Derives `phi = 4/pi` from Vitruvian dual-centre geometry |
| 02 | Mathematical Formalization | The cell definition, parameter counts, four falsifiable conjectures |
| 03 | Progress v2 | Experiments v2–v5, sine wave, seq_len 200 → 3,000 |
| 04 | LSTM Comparison | v6 three-way comparison |
| 05 | Extreme Sequence Stress Test | seq_len 5,000 full BPTT |
| 06 | Comprehensive GPU Experiment | Four tasks, RTX 5090, to 15,000 timesteps |
| 07 | Why It Works | Spectral probing of `W_h` — the geometric attractor explanation |
| 08 | The Probe Journey | Five probes attempting to prove document 7's claim |
| 09 | The Arithmetic Benchmark | Carry propagation, and where the mechanism stops working |
| — | Architecture Diagram | One-page cell diagram: fields, gate, anchor, carry state |

**The derivation** (document 1): normalize the Vitruvian circumscribed circle to radius
`r = 1`. The inscribed square's side is `s = pi/2`, so its half-side is `pi/4`. Then

```
phi = r / (pi/4) = 4/pi = 1.273239544735163
```

Document 1 is explicit that this "is not the golden ratio (1.618...) nor any standard
mathematical constant" — it is the circle-to-square proportion of the figure, offered as
the boundary between oscillatory and rectilinear dynamics.

**The cell** (document 2) is one scalar away from a standard Elman RNN:

```
Elman:  h_t = tanh( W_x·x_t +       W_h·h_{t-1} + b )
DPPU:   h_t = tanh( W_x·x_t + phi · W_h·h_{t-1} + b )
```

`phi` is fixed, not learned — no gate, no normalization layer, **zero added parameters**.
At hidden=32 both DPPU and vanilla have 1,121 parameters against LSTM's 4,513. Document 2
then states four falsifiable conjectures, and documents 3–9 test them in order.

Known defects in document 01 are recorded in [`ERRATA.md`](./ERRATA.md) — the limits are
stated backwards, Table 1 was generated from the mirrored formula, and the critical point
`D* ~ 0.816` does not exist (the true maximum is `D* = ln 2`).

Read in order, these get **less** conclusiveKnown defects in document 01 are recorded in [`ERRATA.md`](./ERRATA.md) — the limits are
stated backwards, Table 1 was generated from the mirrored formula, and the critical point
`D* ~ 0.816` does not exist (the true maximum is `D* = ln 2`).

Read in order, these get **less** conclusive, deliberately, and that is the most
interesting thing about them:

- **03** finds no advantage at all at seq_len 200, then a 1.57x gradient advantage at
  1,000 that compounds toward 3,000 — the null result at short length is reported as
  readily as the positive one.
- **04** reports DPPU beating LSTM by 3.7x on final loss at 3.7x fewer parameters, and
  scopes it immediately: the task is a smooth sine wave, and LSTM keeps its advantage
  where "selective gating is essential."
- **05** reports DPPU-RNN reaching the lowest final loss of the three architectures
  (0.000029 against LSTM's 0.000080) at 4x fewer parameters — then notes the sine-wave
  task is "too smooth to fully stress-test" the claim, and proposes harder ones.
- **06** delivers those harder tasks and a 2x loss advantage at extreme sequence length,
  then immediately undercuts it: a vanilla RNN *also* stayed stable at 15,000 timesteps,
  "unexpected based on conventional gradient theory," raising whether gradient clipping
  alone explains it.
- **07** answers with a mechanism: phi drives `W_h` toward a spectral radius of `2/pi`,
  putting effective recurrent scaling at `phi²/2 = 8/pi² ≈ 0.8106` — a contractive fixed
  point below 1.0.
- **08** tries to prove that identity across five probes and **fails**, saying so in its
  opening line: "A clean closed-form proof was not found." What survives is weaker and
  broader — phi insulates the spectral radius from activation drift, VRU drifting 7x less
  than vanilla. The document argues this is the stronger position for being more modest.
- **09** then finds the mechanism's boundary. On multi-digit arithmetic the model plateaus
  at 50–66% accuracy, and a carry probe reads **0% throughout training** — it never learns
  carry as a general algorithm. The conclusion is a limit, stated plainly: phi's spectral
  insulation "provides continuous memory stability but cannot substitute for discrete
  symbolic working memory."

### `docs/vru-trading-bot/` — the application (complete, 1–6)

| # | Document |
|---|---|
| 01 | What Is Aladdin (BlackRock) |
| 02 | VRU vs Aladdin Competitive Analysis |
| 03 | Third-Party Data Pipeline Strategy |
| 04 | Order Flow Parasitism Trading Strategy |
| 05 | VRU Trading Bot Architecture Specification |
| 06 | Development Roadmap |

It runs as an argument: 01 establishes that BlackRock's Aladdin is risk infrastructure
with a human approving every trade, not an autonomous trader. 02 concedes the honest
comparison is against quant strategies rather than Aladdin itself, and concedes where VRU
loses — data depth and execution latency. 03 addresses the data gap through third-party
feeds. 04 states the strategy, 05 specifies the architecture, 06 schedules the build.

A note on 04, since the name invites misreading: "order flow parasitism" describes
inferring institutional intent from *publicly available* market data — consolidated-tape
dark pool prints, exchange options tape, order book imbalance, 13F filings. The document
is explicit that this is not front-running and involves no material non-public
information.

### `docs/` — standalone

**Recursion as a Load-Bearing Principle** is a literature synthesis situating the work
against the current recurrent-model frontier — Mamba/Mamba-2, xLSTM, Griffin/Hawk, RWKV,
Mixture-of-Recursions. It is the sharpest critique in this archive: it presses on whether
`phi = 4/pi` is derivable rather than cosmetic, and warns that beating an LSTM on
parameter count "is a low bar" against a 1997 architecture. It is included because the
criticism is part of the record.

## Aeon — the proof of concept (`poc/`)

An installable Python package: the recursion cell grafted onto a frozen
**R1-Distill-Qwen-1.5B** backbone as `AeonR1ForCausalLM`, a Qwen2 subclass.

```
poc/aeon/recursion.py    the canonical cell — "single source of truth" (Gen 6)
poc/aeon/block.py        per-block integration with a learned gate
poc/aeon/model.py        AeonModel / AeonR1ForCausalLM
poc/aeon/audit.py        contraction-certificate auditing
poc/scripts/             from_r1, prepare_alpaca, train_stage1/2, verify, probe, chat
poc/tests/               recursion, block shapes, stage-0 gate
```

**The certificate.** `sigma_max(W_h) < MARGIN_H` and `sigma_max(W_c) < MARGIN_C`, both
below 1. It is not enforced after the fact — the parameterization admits no violating
setting.

**Two charts, one cell.** Chart B (`sigmoid(s) · MARGIN · Cayley(A) · diag(tanh(d))`) is
the training default, where the certificate is structural. Chart A stores `W` directly and
projects after each forward, for audit and reference. `equivalence_check()` verifies the
two agree to ~1e-7 — an atlas in the differential-geometry sense: two charts on one object.

**Stage 0** is a wiring gate — with the gate at zero, Aeon must reproduce the untouched R1
backbone, verified to ~1e-6 in fp32.

All modules compile cleanly. The test suite was **not** run here: this container has no
`torch`, so the tests are published unverified in this environment.

## Notebooks

Two threads run through the 96 notebooks, and they converge.

**The architecture.** From an early DPPU cell through recurrent variants (`DPPU-RNN`,
`DRNN`, an LSTM comparison) into the DPPU-VRU series — v11's five-dimensional fields,
v13, v21 ("teach the process, not the answer"), and finally **v23, the explicit register
architecture**, the model the trading work is built on. From its training log:

```
VRU v23  --  Explicit Register Architecture
Hidden: 192  |  Reg dim: 32  |  Device: cuda
PHI x ALPHA = 1.000000000000000  (dual identity)
Vocab: 47  |  Parameters: 81,775
```

Alongside it: batch and depth scaling sweeps across CPU, TPU, T4 and H100; spectral
pairing ablations; a six-way separation test; field attention; recursion-stability work.

**The applications.** A proof-of-cognition blockchain with governance voting, early
market-data prototypes, and the order-flow retargeting documented above.

Filenames are untouched, so the archive is full of `Copy of`, `Another copy of` and
`Untitled17`. The only change made: where Google Drive had stored a notebook without a
file extension, `.ipynb` was appended so GitHub renders it. Contents are byte-identical.

Notebooks were written for Colab and reference paths like `/content/drive/MyDrive/...`.
They will not run unmodified outside that environment.
[`notebooks/exports/`](./notebooks/exports) holds PDF exports of notebooks that also exist
in source form.

## Raw dumps (`raw/`)

`HZ.txt` is an email-to-self from March 2026 containing a working DPPU-VRU **v13** core —
constants, tokenizer, `DPPUCell`, `VRUModel`, training loop — followed by a conversation
summary that ends mid-thread. It is the clearest plain-source statement of the geometry in
the archive, and it is kept whole because the mixture of code and thinking is what the file
actually is. A Gmail header carrying a personal email address was removed; see
[`raw/README.md`](./raw/README.md).

## Training runs (`runs/dppu_vru/`)

`v17` through `v23`, each with a `_log.txt`; several with `_meta.json` (step, curriculum
level, streak, best score, teacher-forcing rate) and a `_checkpoint.pth`. `v21_log.txt` is
the longest run in the archive and logs DPPU against a vanilla baseline step by step.
`v23_audit.jsonl` is the start of the v23 curriculum audit trail.

### What the v21 log actually shows

`v21_log.txt` is the archive's longest run, and it does not support the headline claims.
Recomputed from the log itself (114 logged steps):

| | |
|---|---|
| Parameters | DPPU **229,890** vs vanilla **161,249** — 1.43x **more**, not fewer |
| Accuracy | DPPU exceeded vanilla in **3 of 114** steps |
| Final 20 steps | DPPU 24.5–58.9% vs vanilla 59.8–63.8% |
| Stability | DPPU standard deviation **6.9x** wider than vanilla over those steps |
| Probe at step 16,000 | DPPU **0/5**, vanilla **0/5** — neither model solved any problem |

**Scope of the parameter-efficiency claim.** "LSTM-class performance at ~4x fewer
parameters" was measured on the **sine-wave tasks of documents 04 and 05**. It does not
transfer to the arithmetic curriculum, where v21 uses more parameters than its own baseline
and trails it on accuracy.

### Anchor drift

The learnable anchor initialises at `phi = 1.2732`. Left unregularised in v20, v20b and
v21, it does not stay there:

| run | first | last | range |
|---|---|---|---|
| v20 | 1.2535 | 1.3245 | — |
| v20b | 1.2792 | 1.3505 | — |
| v21 | 1.2742 | 1.3284 | 1.2148 – 1.4238 |

Across v21's 123 anchor samples the mean is **1.3234, or +3.9% from 4/pi**, with the
first-half mean (1.2898) below the second-half mean (1.3566). The drift is **upward and
away from `4/pi`**, not toward it. It is not monotonic — the value decreases at 58 of 122
transitions — but the trend is unambiguous.

This is the archive's own evidence running against the spectral-attractor argument of
document 07. **It is unexplained here, and no attempt is made to explain it away.**

The v23 run here is a **beginning, not a finished result** — the log ends a few seconds
in, at "Stage 1: single-digit no carry".

## Status

Partial upload; further batches of source material are still to come.

- `poc/` — Aeon proof-of-concept
- `docs/foundations/` — five origin papers
- `docs/vru-trading-bot/` — complete (1–6)
- `docs/vru-architecture/` — **complete (1–9)**; a document 10 (VRU v11 scratchpad
  results) is named as planned but not yet written

## Authorship

Work by **Dylan Michael Scott** (Horizon Tech), per the bylines on the documents in
`docs/`. Two of the January 2025 foundation papers additionally carry a co-author byline.

## License

Not yet specified. All rights reserved by the author pending a license decision.
