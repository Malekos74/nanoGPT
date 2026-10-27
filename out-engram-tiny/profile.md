Profile of 2 training steps (after warmup).

| component | calls/step | CPU ms/step | CPU % | CUDA ms/step | CUDA % |
|---|---:|---:|---:|---:|---:|
| step | 1 | 358.35 | 100.0 | 0.00 | 0.0 |
| &nbsp;&nbsp;· h2d copies, 0.00 MB | 0 |  |  | 0.00 | 0.0 |
| &nbsp;&nbsp;· d2h copies, 0.00 MB | 0 |  |  | 0.00 | 0.0 |
| &nbsp;&nbsp;· d2d copies, 0.00 MB | 0 |  |  | 0.00 | 0.0 |
| fwd/backbone | 5 | 61.22 | 17.1 | 0.00 | 0.0 |
| fwd/engram | 2 | 5.11 | 1.4 | 0.00 | 0.0 |
| &nbsp;&nbsp;engram/compress | 2 | 0.09 | 0.0 | 0.00 | 0.0 |
| &nbsp;&nbsp;engram/hash | 2 | 0.36 | 0.1 | 0.00 | 0.0 |
| &nbsp;&nbsp;engram/fetch | 2 | 0.19 | 0.1 | 0.00 | 0.0 |
| &nbsp;&nbsp;engram/gate | 2 | 1.42 | 0.4 | 0.00 | 0.0 |
| &nbsp;&nbsp;engram/value | 2 | 0.41 | 0.1 | 0.00 | 0.0 |
| &nbsp;&nbsp;engram/short_conv | 2 | 2.20 | 0.6 | 0.00 | 0.0 |
| backward | 1 | 147.55 | 41.2 | 0.00 | 0.0 |
| &nbsp;&nbsp;bwd/backbone | 4 | 97.61 | 27.2 | 0.00 | 0.0 |
| &nbsp;&nbsp;bwd/engram | 2 | 9.66 | 2.7 | 0.00 | 0.0 |
| optimizer | 1 | 134.14 | 37.4 | 0.00 | 0.0 |
| &nbsp;&nbsp;optimizer/muon | 1 | 45.01 | 12.6 | 0.00 | 0.0 |
| &nbsp;&nbsp;optimizer/adamw | 1 | 64.24 | 17.9 | 0.00 | 0.0 |
| &nbsp;&nbsp;optimizer/tables | 1 | 0.54 | 0.2 | 0.00 | 0.0 |
