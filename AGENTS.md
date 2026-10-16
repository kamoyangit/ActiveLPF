# AGENTS.md

Streamlit calculator that designs an NPN single-transistor active low-pass filter
(emitter follower + Sallen-Key), for 1st/2nd/3rd-order Butterworth, Bessel, or
Chebyshev response.

User-visible strings (labels, warnings, captions, markdown, `st.metric` titles) are
Japanese — keep new UI text Japanese. Comments in `app.py` are also Japanese; match
the language of the code you are editing. Identifiers and file paths stay English.

## Files

- `app.py` — the whole app: sidebar inputs, filter sizing math, hand-built SVG
  schematic, frequency-response simulation, parts table, LTspice netlist, docs.
- `app_old.py`, `app_org.py` — superseded snapshots of earlier designs. Nothing
  imports them. Make feature changes in `app.py`, not here.
- `requirements.txt`, `.python-version` (3.10.9), `venv/` — environment.

## Running

```bash
venv/bin/streamlit run app.py                       # interactive
venv/bin/streamlit run app.py --server.headless true --server.port 8599
```

Use the checked-in `venv/` (Python 3.10.9, streamlit 1.65.0); `venv/bin/python` for
everything else.

`requirements.txt` lists `streamlit>=1.33`, but `app.py` calls
`width="stretch"` on `st.dataframe`/`st.altair_chart` (the `use_container_width`
replacement), which is far newer than 1.33. Treat the venv as the supported target
and do not assume 1.33 compatibility. (Uncorrected stale floor — confirm before
relying on it.)

## Contracts worth knowing

- **Top-to-bottom script, no state.** The script re-executes on every interaction:
  one `with st.sidebar:` block near the top defines all inputs, everything below
  derives from those locals. No caching, no `session_state`, no callbacks. A new
  input goes in the sidebar block, and downstream code reads it as a plain variable.
- **`design_sections()` returns `(f0, Q)` pairs; `Q=None` means a 1st-order
  section.** For 3rd order it returns `[(2nd-order SK f0, Q), (1st-order RC f0, None)]`
  — the reverse of the order the UI displays. Read `sections` via the existing
  destructuring instead of assuming sort order; `f0_min`/`f0_max` come from it.
- **3rd-order topology:** 1st-order RC (`R3 = R / r3_div`, deliberately low
  impedance) → Sallen-Key 2nd order → emitter follower output.
- **E24 rounding is part of the result, not display-only.** With the "E24標準値に
  丸める" checkbox on, the parts table, schematic labels, SPICE netlist, and the
  `actual_H_order*` simulation all use the rounded `R1s/R2s/C1s/C2s/C3s/R3s`. Never
  feed the unrounded values into `actual_H_*` — the curves would stop matching the
  parts table and netlist.
- **Two curve families, both normalized.** `ideal_H` is the textbook response (DC =
  0 dB; the even-order Chebyshev DC-referencing is deliberate — see the comment, not
  a bug). `actual_H_order1/2/3` are nodal analyses including emitter-follower gain
  `K`, bias-network loading, `RL`, and optionally `Cin`/`Cout`. Both are plotted
  relative to the simulated passband reference `ref_db` sampled at `f0_min/100`, so
  the curves are comparable to each other but are not absolute gains.
- **`Cin`/`Cout`** are sized for an HPF corner at `f0/100` and only enter the
  simulation when the coupling checkbox is on.
- **`circuit_svg(p, order)`** is hand-written SVG string building with hard-coded
  coordinates; its labels come from `p`, already formatted by `fmt_r`/`fmt_c`. There
  is no netlist-to-drawing generation — schematic edits are coordinate-level.
- Units are SI throughout (Hz, Ω, F, V); scaling to kΩ/nF/µF happens only in
  `fmt_r`/`fmt_c`/`fmt_spice` at display time. `VT = 0.02585`.
- `st.error` + `st.stop()` guards invalid bias (`Vcc/2 − 0.65 ≤ 0`); the `correct_k`
  path has a warning branch for "no K-compensation solution". Keep both reachable
  instead of removing them.

## Verifying changes

No test suite, linter config, or git repository here (`pytest` is not installed), so
verify manually:

```bash
venv/bin/python -m py_compile app.py
venv/bin/python -c "import app; print(app.design_sections('バターワース', 1000.0, 0.5, 2))" 2>/dev/null
```

Importing runs the whole script in bare mode, which is harmless — it emits
`missing ScriptRunContext` warnings that can be ignored. This is the cheapest way to
check `design_sections`, `nearest_e24`, and `actual_H_order*` numerically for
several `(order, kind, fc, R)` combinations.

For a rendering or startup check, boot the server and confirm it serves, then kill
it:

```bash
venv/bin/streamlit run app.py --server.headless true --server.port 8599 &
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:8599/healthz   # expect 200
```

The app script itself only executes when a browser session attaches, so `healthz`
proves startup, not rendering — for rendering changes, also load the page or drive
the compute path directly. Sweep all three orders × the three response types before
declaring a sizing/graphing change done.
