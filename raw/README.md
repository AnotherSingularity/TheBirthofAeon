# raw/

Verbatim source dumps that are not notebooks — kept in whatever form they arrived in.

## `HZ.txt`

An email-to-self from **7 March 2026**, carrying a working DPPU-VRU **v13** core.
Despite the `.txt` extension it is a runnable PyTorch script for the first ~190 lines,
followed by a conversation summary.

**Contents, in order:**

1. **Constants and operators** — `PHI_CL = 4.0/math.pi`, `INV_PHI = math.pi/4.0`,
   `D_CAP = 4`, and the dynamic field functions `pi_dyn`, `phi_dyn`, `omega`,
   `compute_delta`. This is the clearest plain-source statement of the geometry in the
   archive.
2. **`MathTokenizer` / dataset** — a 21-character arithmetic vocabulary and the
   curriculum data generators.
3. **`DPPUCell` and `VRUModel`** — the cell and the model wrapped around it.
4. **`train()`** — the curriculum training loop.
5. **A trailing conversation summary** (not code) discussing thermodynamics, the five
   dimensional fields `D_0`–`D_4`, and the "consciousness anchor" `C` field. It ends
   mid-thread, on an offer to re-insert the cosine annealing scheduler and four-level
   curriculum.

It is kept whole rather than split into a `.py` and a note, because the mixture is what
the file actually is: a snapshot of the work in progress, sent to oneself at midnight.

**One modification:** a four-line Gmail header at the top — sender name, personal email
address, timestamp, recipient — was removed before publication. Nothing else was changed;
the body below that header is verbatim, trailing conversation included.

## Note on `DELTA_STAR` (line 17)

`HZ.txt` is preserved verbatim and is **not** edited. This note records a defect in it.

```python
DELTA_STAR = math.log((1.6180 - 1.0) / (PHI_CL - 1.0))    # = 0.8161396
```

1. **It is a decay time, by definition** — the `D` at which `f(D) = 1 + (C-1)e^(-D)`,
   started from the golden ratio 1.618, passes through `4/pi`. Evaluating it at that point
   restates its own definition; it confirms nothing independently.
2. **It is not a peak**, and is unrelated to the field system's critical point
   `D* = ln 2 = 0.693147`.
3. **It does not apply to the `phi_dyn` coded at line 24**, which starts at `4/pi` and
   decays to 1. That function at `DELTA_STAR` is `1.1208`, not `4/pi`.

Outside `raw/` this quantity is called **`GOLDEN_DECAY_TIME`**, so it cannot be confused
with the critical point. See [`../ERRATA.md`](../ERRATA.md) E4.

Also unmodified: `omega()` at line 25 carries a `(1 + D)` denominator that makes the
function monotonically decreasing, so the interior maximum attributed to it does not exist.
It has a live call site at line 107. See `ERRATA.md` E3.
