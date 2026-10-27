Profile of 2 training steps (after warmup).

| component | calls/step | CPU ms/step | CPU % | CUDA ms/step | CUDA % |
|---|---:|---:|---:|---:|---:|
| step | 1 | 350.48 | 100.0 | 0.00 | 0.0 |
| fwd/embed | 3 | 1.82 | 0.5 | 0.00 | 0.0 |
| fwd/engram | 6 | 15.27 | 4.4 | 0.00 | 0.0 |
| engram/compress | 6 | 0.26 | 0.1 | 0.00 | 0.0 |
| engram/hash | 6 | 1.05 | 0.3 | 0.00 | 0.0 |
| engram/fetch | 6 | 0.74 | 0.2 | 0.00 | 0.0 |
| engram/gate | 6 | 4.32 | 1.2 | 0.00 | 0.0 |
| engram/value | 6 | 1.10 | 0.3 | 0.00 | 0.0 |
| engram/short_conv | 6 | 6.90 | 2.0 | 0.00 | 0.0 |
| fwd/block | 9 | 25.60 | 7.3 | 0.00 | 0.0 |
| fwd/head | 3 | 174.98 | 49.9 | 0.00 | 0.0 |
| backward | 1 | 160.74 | 45.9 | 0.00 | 0.0 |
| optimizer | 1 | 107.63 | 30.7 | 0.00 | 0.0 |
