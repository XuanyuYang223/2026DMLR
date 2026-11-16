# Readable calibration curve pages

These presentation-only figures redraw the saved pooled learning curves with
separate title, subtitle and legend bands. The earlier audited `final/` output,
its manifest and the frozen report generator are unchanged.

Each measured instance has the same four recorded curves: training and
development for the original initialization/data pairings 617 and 829. The
dotted line is the recorded mean development generation ceiling. Values are
read directly from `../final/learning_curves.csv` and
`../final/endpoint_summary.csv`; no smoothing, new averaging, checkpoint
selection, model execution, input generation or test evaluation is performed.
The context-rejected instance has no invented curve. Lengths 8, 10 and 12 all
occur in training, and these are the same data units as the v0.1 calibration.

The inherited eight-token generation budget limits some structured answers.
The majority-answer comparator is uncapped; see
`../../BASELINE_AND_REPRODUCTION_NOTES.md` before interpreting negative gains.
Numerical results and the complete endpoint table remain in
`../final/RESULT_REPORT.md` and its source CSVs.

Reproduce from the repository root into a **new** destination:

```powershell
.venv/Scripts/python.exe -B experiments/platform_calibration_v02/reuse_authorized_v1/plot_readable_curves.py --out experiments/platform_calibration_v02/reuse_authorized_v1/artifacts/presentation_reproduction
```

The writer rejects existing destinations. `MANIFEST.json` records source CSV
hashes, the unchanged final-manifest hash, script hash, figure inventory and
layout checks. Both PNG and SVG versions are provided.
