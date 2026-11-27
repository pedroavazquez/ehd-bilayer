# Paper draft: oscillatory onset of electroconvection in two stacked dielectric layers

First draft (22 Nov 2026), REVTeX 4.2, pof substyle (single column).

```
latexmk -pdf paper.tex
python3 compute_paper_data.py   # neutral curves and eigenfunctions -> data/paper_data.json (~1.5 min)
python3 make_figures.py         # all figures -> figures/
```

Data: `data/Tc_Gr.json` (T_c(Gr), mechanism tests), `data/delta_scan.json` (`../stability/delta_scan.py`),
`data/neutral_62x100.csv` and `data/fwd_32x40_*.csv` (Firedrake verification: `two_layer_eigen.py` and
`two_layer_forward.py` in `firedrake/two_layer/` of `pedroavazquez/nonLinearStabilityMonolithic`, branch
`two-layer`, because they use the monolithic EHD solver of that repository).
Open items are marked TODO in paper.tex (co-authors, acknowledgments).
