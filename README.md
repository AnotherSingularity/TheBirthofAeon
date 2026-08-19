# The Birth of Aeon

The raw working files behind Aeon — Colab notebooks, training-run artifacts, and research
documents — published as-is.

This is an **archive, not a product**. Nothing here has been cleaned up, refactored, or
rewritten for an audience. It is the actual trail of work: dead ends, duplicated notebooks,
`Copy of Copy of` filenames, empty scratch files and all. That mess is the point. It is
what the process actually looked like.

## The architecture in one paragraph

The **VRU (Vitruvian Recurrent Unit)**, formerly the **DPPU (Dynamic Pi-Phi Processing
Unit)**, is a recurrent cell built around a single fixed geometric constant:

```
phi = 4/pi = 1.2732...      alpha = pi/4 = 0.7854...      phi x alpha = 1
```

`phi` is the circle-square ratio taken from Da Vinci's Vitruvian Man — hence the rename.
It scales the recurrent weight path; its reciprocal dual `alpha` scales the input
projection. The claim is that this proportion, applied as a fixed scalar with no gates,
normalization or constraint machinery, holds gradients stable across sequences of
arbitrary length. The reported result is LSTM-class performance at roughly 4x fewer
parameters, stable to 15,000 timesteps under full BPTT.

Whether `phi = 4/pi` is *derived* or merely *decorative* is the open question the archive
keeps circling. It is not settled here, and the documents say so.

## What's in here

| Path | Contents |
|---|---|
| [`notebooks/`](./notebooks) | 96 Google Colab notebooks. See [`notebooks/INDEX.md`](./notebooks/INDEX.md) for a grouped index. |
| [`runs/dppu_vru/`](./runs/dppu_vru) | Training logs, checkpoints and metadata for runs **v17–v23**. |
| [`docs/vru-architecture/`](./docs/vru-architecture) | The core research series — what the architecture is and why it works. |
| [`docs/vru-trading-bot/`](./docs/vru-trading-bot) | A complete six-part series applying it to markets. |
| [`raw/`](./raw) | Verbatim source dumps that are not notebooks. |

## Documents

Two **separate** numbered series live here. Both have a document 6, which is why they sit
in different folders.

### `docs/vru-architecture/` — the core research

Documents 1–9, of which **5 through 9 are present**. Document 6 records the rename from
DPPU to VRU; documents 1–5 used the DPPU designation. Document 9 names a Document 10 as
planned but not yet written.

| # | Document | What it does |
|---|---|---|
| 05 | Extreme Sequence Stress Test | seq_len 5,000 full BPTT, three-way comparison |
| 06 | Comprehensive GPU Experiment | Four tasks, three models, RTX 5090, to 15,000 timesteps |
| 07 | Why It Works | Spectral probing of `W_h` — the geometric attractor explanation |
| 08 | The Probe Journey | Five probes attempting to prove document 7's claim |
| 09 | The Arithmetic Benchmark | Carry propagation, and where the mechanism stops working |
| — | Architecture Diagram | One-page cell diagram: fields, gate, anchor, carry state |

**Missing: documents 1–4.** Per the series listings in documents 5 and 9, those are
1 Origin (the derivation of phi from Da Vinci's geometry), 2 Mathematical formalization,
3 RNN experiments v2–v5, and 4 the LSTM comparison. The two listings disagree slightly on
document 2 — document 5 calls it "v1 LaTeX", document 9 calls it "Field Simulation" —
which is preserved here rather than reconciled.

Read in order, these get **less** conclusive, deliberately, and that is the most
interesting thing about them:

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

The v23 run here is a **beginning, not a finished result** — the log ends a few seconds
in, at "Stage 1: single-digit no carry".

## Status

Partial upload; further batches of source material are still to come.

- `docs/vru-trading-bot/` — complete (1–6)
- `docs/vru-architecture/` — documents 5–9 of 9; **1–4 not yet uploaded**, and a
  document 10 (VRU v11 scratchpad results) is named as planned but not yet written

## License

Not yet specified. All rights reserved by the author pending a license decision.
