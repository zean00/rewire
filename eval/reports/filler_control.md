# Neutral-Filler THINK Control (Phase 2)

domains: ['arc_hard', 'arith_hard'], n=200, think budget=256, decision method: D2 alpha=1.0 pinned

| arm | accuracy | mean confidence |
|---|---|---|
| base | 0.660 | 0.444 |
| think | 0.625 | 0.476 |
| filler | 0.690 | 0.445 |

- Δacc(think) = **-0.035**
- Δacc(filler) = **+0.030**
- differential = **-0.065** → no evidence reasoning content (vs extra tokens) improves decisions (S5 NOT supported)
- filler tokens: 87
