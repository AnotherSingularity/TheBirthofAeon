# The Birth of Aeon

The raw working files behind Aeon — Colab notebooks, training-run artifacts, and
architecture documents — published as-is.

This is an **archive, not a product**. Nothing here has been cleaned up, refactored,
or rewritten for an audience. It is the actual trail of work: dead ends, duplicated
notebooks, `Copy of Copy of` filenames, empty scratch files and all. That mess is the
point. It is what the process actually looked like.

## What's in here

| Path | Contents |
|---|---|
| [`notebooks/`](./notebooks) | 96 Google Colab notebooks. See [`notebooks/INDEX.md`](./notebooks/INDEX.md) for a grouped index. |
| [`runs/dppu_vru/`](./runs/dppu_vru) | Training logs, checkpoints and metadata for runs **v17–v23**. |
| [`docs/`](./docs) | Architecture and strategy documents (Horizon Tech). |

## The work

Two threads run through the notebooks, and they converge.

**The architecture.** A line of development from an early DPPU cell through recurrent
variants (`DPPU-RNN`, `DRNN`, an LSTM comparison), into the DPPU-VRU series — v11's
five-dimensional fields, v13, v21 ("teach the process, not the answer"), and finally
**v23, the explicit register architecture**. v23 is the model the trading work is built
on. From its own training log:

```
VRU v23  --  Explicit Register Architecture
Hidden: 192  |  Reg dim: 32  |  Device: cuda
PHI x ALPHA = 1.000000000000000  (dual identity)
Vocab: 47  |  Parameters: 81,775
```

Alongside it: batch/depth scaling sweeps across CPU, TPU, T4 and H100; spectral pairing
ablations; a six-way separation test; field attention; and recursion-stability work.

**The applications.** A proof-of-cognition blockchain with governance voting, early
market-data prototypes, and — documented in `docs/` — a retargeting of v23 from
arithmetic reasoning to order-flow classification.

## Training runs (`runs/dppu_vru/`)

`v17` through `v23`, each with a `_log.txt`; several with `_meta.json` (step, curriculum
level, streak, best score, teacher-forcing rate) and a `_checkpoint.pth`. `v21_log.txt`
is the longest run in the archive and logs DPPU against a vanilla baseline step by step.
`v23_audit.jsonl` is the start of the v23 curriculum audit trail.

The v23 run here is a **beginning, not a finished result** — the log ends a few seconds
in, at "Stage 1: single-digit no carry".

## Documents (`docs/`)

- `04 - Order Flow Parasitism Trading Strategy.pdf`
- `05 - VRU Trading Bot Architecture Specification.pdf`
- `06 - Development Roadmap.pdf`

These are documents 4–6 of a longer numbered series; **1–3 are not in this repository yet.**

A note on the strategy document, since the name invites misreading: "order flow
parasitism" describes inferring institutional intent from *publicly available* market
data — consolidated-tape dark pool prints, exchange options tape, order book imbalance,
13F filings. The document is explicit that this is not front-running and involves no
material non-public information.

## Reading the notebooks

Filenames are untouched, so the archive is full of `Copy of`, `Another copy of` and
`Untitled17`. The only change made: where Google Drive had stored a notebook without a
file extension, `.ipynb` was appended so GitHub renders it. Contents are byte-identical
to the originals.

Notebooks were written for Colab and reference paths like `/content/drive/MyDrive/...`.
They will not run unmodified outside that environment.

## Status

This is a partial upload — further batches of source material are still to come.

## License

Not yet specified. All rights reserved by the author pending a license decision.
