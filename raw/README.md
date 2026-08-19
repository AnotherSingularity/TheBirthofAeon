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
