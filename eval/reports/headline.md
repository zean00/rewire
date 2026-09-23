# Headline Comparison (S8/Q12)

gate: D2 confidence < 0.7 (margin < 0.1 also escalates; premask < 0.05 forces escalation); think budget 256/256. A and G reused from baseline_table records (same machine/config).
CAA = accuracy / p50 latency (s).

| arm | n | accuracy | p50 s | CAA | escalation rate |
| --- | --- | --- | --- | --- | --- |
| A never-think | 500 | 0.308 | 0.26 | 1.20 | 0% |
| C no-think generate | 500 | 0.764 | 0.29 | 2.67 | 0% |
| G think-always | 500 | 0.490 | 17.29 | 0.03 | 0% |
| H-redecide | 500 | 0.856 | 0.26 | 3.31 | 18% |
| H-generate | 500 | 0.786 | 0.25 | 3.13 | 18% |

## Per-domain accuracy

| arm | arc_easy | arc_hard | arith_easy | arith_hard | intent |
| --- | --- | --- | --- | --- | --- |
| A | 0.290 | 0.140 | 1.000 | 0.110 | 0.000 |
| C | 0.980 | 0.920 | 0.670 | 0.590 | 0.660 |
| G | 0.440 | 0.380 | 0.440 | 0.790 | 0.400 |
| H-redecide | 0.970 | 0.930 | 0.980 | 0.660 | 0.740 |
| H-generate | 0.940 | 0.830 | 0.940 | 0.500 | 0.720 |
