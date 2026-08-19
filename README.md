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

## Documents

Two **separate** numbered series live here. Both have a document 6, which is why they sit
in different folders.

### `docs/vru-architecture/` — the core research

Documents 1–8, of which **6, 7 and 8 are present**. Document 6 records the rename from
DPPU to VRU; documents 1–5 used the DPPU designation. Per document 6, the missing entries
are: 1 Origin (Da Vinci geometry, derivation of phi), 2 Mathematical formalization,
3 RNN experiments v2–v5, 4 LSTM comparison, 5 Stress test at seq_len 5,000.

| # | Document | What it does |
|---|---|---|
| 06 | Comprehensive GPU Experiment | Four tasks, three models, RTX 5090, sequences to 15,000 timesteps |
| 07 | Why It Works | Spectral probing of `W_h` — the geometric attractor explanation |
| 08 | The Probe Journey | Five probes attempting to prove document 7's claim |
| — | Architecture Diagram | One-page cell diagram: fields, gate, anchor, carry state |

These three are worth reading in order, because they get **less** conclusive as they go,
deliberately:

- **06** reports the headline result — a consistent 2x loss advantage over a vanilla RNN
  at extreme sequence length — then immediately undercuts it: a vanilla RNN *also* stayed
  stable at 15,000 timesteps, "unexpected based on conventional gradient theory," raising
  the question of whether gradient clipping alone explains the stability.
- **07** answers with a mechanism: `phi` drives `W_h` toward a spectral radius of
  `2/pi`, putting the effective recurrent scaling at `phi²/2 = 8/pi² ≈ 0.8106` — a
  contractive fixed point below 1.0.
- **08** then tries to prove that identity across five probes and **fails**. It says so in
  its own opening line: "A clean closed-form proof was not found." What survives is a
  weaker, broader claim — `phi` insulates the spectral radius from activation drift, with
  VRU drifting 7x less than vanilla. The document argues this is the stronger position
  precisely because it is more modest.

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
- `docs/vru-architecture/` — documents 6, 7, 8 of 8; **1–5 not yet uploaded**

## License

Not yet specified. All rights reserved by the author pending a license decision.
