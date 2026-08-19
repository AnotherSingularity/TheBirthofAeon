# Errata

Defects found in this archive's own documents, recorded rather than repaired.

**The source documents are not edited.** `docs/` holds the papers as written, and `raw/`
is verbatim by design. Rewriting a March 2026 research PDF to say something it did not say
would falsify the record this repository exists to preserve. Corrections live here instead,
each with a derivation the reader can run.

Every figure below was recomputed from the formulas as printed in the source documents.

---

## E1 — Document 01 §3: the limits are stated backwards

**Document:** `docs/vru-architecture/01 - Origin - Da Vinci Dual-Center Geometry.pdf`

The prose states that at `D = 0`, `pi_dynamic → 4`, and that as `D → ∞`,
`pi_dynamic → pi_classical`. The formula printed in the same section does the opposite:

```
pi_dynamic(D) = 4 - (4 - pi)·exp(-D)

  pi_dynamic(0)  = 4 - (4 - pi)  = pi = 3.141593
  pi_dynamic(∞)  = 4
```

`pi` rises toward 4 as folding increases. It does not fall from 4 toward `pi`.

**The formula is correct; the prose is wrong.** Three independent sources agree with the
formula:

- `docs/foundations/Dynamic Pi Transformation.pdf` (January 2025) — "when Delta = 0 … pi
  behaves classically", and describes pi *shifting toward 4* with increasing folding.
- `raw/HZ.txt` line 23 implements the rising form.
- Document 01's own Table 1 is the only artefact that follows the prose — see E2.

**Correct reading:** at `D = 0`, `pi_dynamic = pi_classical` and `phi_dynamic = 4/pi`
(classical pi, geometric phi). As `D → ∞`, `pi_dynamic → 4` and `phi_dynamic → 1`.

---

## E2 — Document 01 Table 1: generated from the mirrored formula

Table 1's `pi_dynamic` column runs 4.0000 → 3.143, i.e. it was produced by
`pi + (4 - pi)·exp(-D)` — the mirror of the formula printed beside it. The `phi_dynamic`
column does not match its own formula either.

Recomputed from the formulas **as printed**:

| D | pi_dyn | phi_dyn | Pi·Phi |
|---|---|---|---|
| 0.000000 | 3.141593 | 1.273240 | 4.000000 |
| 0.500000 | 3.479350 | 1.165728 | 4.055976 |
| 0.693147 | 3.570796 | 1.136620 | **4.058638** ← maximum |
| 2.000000 | 3.883827 | 1.036979 | 4.027447 |
| 5.000000 | 3.994216 | 1.001841 | 4.001570 |

The final column is the product `Pi·Phi`, **not** `Omega` — see E3.

---

## E3 — Document 01: the critical point D* ≈ 0.816 does not exist

Document 01 states that `Omega(D) = pi_dyn·phi_dyn/(1 + D)` is maximized at an interior
critical point `D* ≈ 0.816`, located by gradient ascent, and that the cell operates there.

**`Omega` has no interior maximum.** It decreases monotonically on `[0, ∞)`. Checked over
`D ∈ [0, 30]` at 600,001 sample points, in all four combinations of the two directions of
`pi_dyn` with the two candidate values of `phi` (4/pi and 1.618): the argmax is the
boundary `D = 0` in every case. Gradient ascent on `Omega` cannot find 0.816, nor any
interior point. **The `(1 + D)` denominator is the defect** — it is a monotone decay
imposed on a bounded product.

**What is actually there.** Drop the denominator and the product has an exact closed form.
Let `u = exp(-D)`, `A = 4 - pi`, `B = 4/pi - 1`. Then

```
Pi(D)·Phi(D) = (4 - A·u)(1 + B·u) = 4 + (4B - A)·u - A·B·u²
```

and the two coefficients coincide:

```
4B - A  =  A·B  =  16/pi + pi - 8  =  0.2345508325...      (exact identity)
```

so, writing `C = 16/pi + pi - 8`,

```
Pi·Phi = 4 + C·u·(1 - u)
```

This gives, with no numerical search:

- **exactly 4 at both limits** (`u = 0` and `u = 1`)
- a **unique maximum at `u = 1/2`**, i.e. `D* = ln 2 = 0.6931472`
- **peak value `4 + C/4 = 4.0586377`**

Verified: the closed form agrees with the direct product to `1.8e-15` over `D ∈ [0, 30]`,
and the identity holds to machine precision.

**Corrections to the document's wording:**

| stated | correct |
|---|---|
| `D* ≈ 0.816` | `D* = ln 2 = 0.693147` |
| "located through gradient ascent" | solved in closed form |
| "confirmed as the natural attractor" | *withdrawn* — see below |

The last one is not a wording fix. The product is **pinned to exactly 4 at both limits** and
bulges by `C/4` in between. That is a structural fact about how the two fields are defined:
their coefficients happen to coincide. It is **not** evidence that `4/pi` is an attractor,
and it must not be worded as though it were.

`D* = ln 2` is a cleaner result than the one it replaces — exact, derivable in three lines,
no numerics. It is recorded here as a correction, not substituted silently.

**Not changed:** `omega()` at `raw/HZ.txt` line 25 retains the `(1 + D)` denominator. It has
a live call site at line 107, and `raw/` is preserved verbatim. The defect is recorded, not
patched.

---

## E4 — `DELTA_STAR` is mislabelled, and unrelated to D*

**File:** `raw/HZ.txt` line 17 (unmodified)

```python
DELTA_STAR = math.log((1.6180 - 1.0) / (PHI_CL - 1.0))    # = 0.8161396
```

Three facts:

1. **It is a decay time, by definition.** Under `f(D) = 1 + (C - 1)·exp(-D)` starting from
   the golden ratio 1.618, `DELTA_STAR` is precisely the `D` at which `f` passes through
   `4/pi`. Evaluating `f(DELTA_STAR) = 4/pi` restates its own definition. It is not
   independent confirmation of anything.
2. **It is not a peak**, and has no relationship to `D* = ln 2 = 0.693147`.
3. **It does not apply to the `phi_dyn` actually coded** at line 24, which starts at `4/pi`
   and decays to 1. That function's value at `DELTA_STAR` is `1.1208`, not `4/pi`.

Outside `raw/`, this quantity is named **`GOLDEN_DECAY_TIME`** so it cannot again be
mistaken for the field system's critical point.

---

## E5 — Naming

`VRU` is expanded two ways in this archive:

- **"Vitruvian Recurrent Unit"** — documents 02, 06, 07, 08, 09, including document 06,
  which records the DPPU → VRU rename and its rationale.
- **"Variable Recurrent Unit"** — `VRU Architecture Diagram.pdf`, and nowhere else.

**Canonical: Vitruvian Recurrent Unit.** The diagram is the sole outlier and is left as
issued.
