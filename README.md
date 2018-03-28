## Location Tracking

This repository implemented

- *Square-root information filter*,
- *Backward square-root information filter*,
- *Square-root information smoother*,
- *Square-root information smoother as forward-backward*,
- *Square-root interacting multiple model (IMM) filter* and
- *Square-root IMM smoother*.

### How It Works

The algorithms are described in the [tex/algorithm.pdf](tex/algorithm.pdf).

### Example

A few examples are included:

- [Forward and Backward Square-root information filter](src/main/scala/srif/tracking/example/UniModelFilterExample.scala)
- [Square-root information smoother (including forward-backward algorithm)](src/main/scala/srif/tracking/example/UniModelSmootherExample.scala)
- [IMM filter and smoother](src/main/scala/srif/tracking/example/MultipleModelExample.scala).

### Reference

*Bierman, G.J. "Factorization Methods for Discrete Sequential Estimation."
Dover Books on Mathematics Series,
Dover Publications, 2006.*

### License

Apache License 2.0
