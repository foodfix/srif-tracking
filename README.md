## Location Tracking

This repository implemented

- *Square-root information filter and smoother*,
- *Square-root interacting multiple model (IMM) filter smoother*
- *Square-root IMM filter and smoother* and
- *Square-root Viterbi Algorithm*

### How It Works

The algorithms are described in the [tex/algorithm.pdf](tex/algorithm.pdf).

### Example

A few examples are included:

- [Uni-model filter and smoother](src/main/scala/srif/tracking/example/UniModelExample.scala)
- [Multiple model filter and smoother](src/main/scala/srif/tracking/example/MultipleModelExample.scala).

### Reference

*Bierman, G.J. "Factorization Methods for Discrete Sequential Estimation."
Dover Books on Mathematics Series,
Dover Publications, 2006.*

### License

Apache License 2.0
