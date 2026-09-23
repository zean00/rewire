# Baseline Table v1

Domains: arc_easy (100), arc_hard (100), arith_easy (100), arith_hard (100), intent (100). Generative cells greedy; D cells pinned no-think read point; PMI lambda=1.0 content-free 'N/A'.

## Main table

| cell | n | accuracy | p50 ms | p95 ms | mean tokens | ECE | premask p50 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | 500 | 0.308 | 258 | 1657 | 19 | — | — |
| C | 500 | 0.764 | 286 | 574 | 10 | — | — |
| G | 500 | 0.490 | 17287 | 17778 | 501 | — | — |
| B | 30 | 0.500 | 25648 | 26023 | 740 | — | — |
| D1 | 500 | 0.534 | 155 | 918 | 0 | 0.077 | 0.1497 |
| D2 | 500 | 0.752 | 190 | 235 | 0 | 0.223 | 0.1497 |
| D2P | 500 | 0.648 | 371 | 405 | 0 | 0.125 | 0.1497 |

## Per-domain accuracy

| cell | arc_easy | arc_hard | arith_easy | arith_hard | intent |
| --- | --- | --- | --- | --- | --- |
| A | 0.290 | 0.140 | 1.000 | 0.110 | 0.000 |
| C | 0.980 | 0.920 | 0.670 | 0.590 | 0.660 |
| G | 0.440 | 0.380 | 0.440 | 0.790 | 0.400 |
| B | 0.500 | 0.333 | 0.667 | 0.667 | 0.333 |
| D1 | 0.750 | 0.590 | 0.310 | 0.290 | 0.730 |
| D2 | 0.920 | 0.750 | 0.740 | 0.570 | 0.780 |
| D2P | 0.890 | 0.730 | 0.740 | 0.280 | 0.600 |

## ARC-Easy extras

| experiment | result |
| --- | --- |
| order-shuffle selection stability (D2 vs reshuffled, ARC-Easy) | 94/100 |
| premask p50: D2(arc_easy) vs D2-anchored(arc_easy) | 0.0164 vs 0.9123 |
| accuracy: D2(arc_easy) vs D2-anchored(arc_easy) | 0.920 vs 0.970 |

### D2 length-normalization sensitivity (ARC-Easy)

| alpha | accuracy |
| --- | --- |
| 0.0 | 0.980 |
| 0.5 | 0.970 |
| 1.0 | 0.920 |

B cell note: forced-think runs on a 30-item subsample (6/domain); G covers think-always at full n (Phase 0 measured G≡B behaviorally).

