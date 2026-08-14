# Tutorials

Background material for the paper, written for readers who want to understand the methods rather
than only use them. Each tutorial is a self-contained directory with its own figures and the script
that generates them.

| | |
| --- | --- |
| [`particle-filter/`](particle-filter/) | Why a bootstrap filter degenerates on the quantized Gaussian process, and how the fully adapted filter of Section 4 avoids it. Starts from the standard machinery. |

The figure scripts need the `egp` package (`pip install -e egp` from the repository root) and
matplotlib. Run one with, for example:

```sh
cd tutorials/particle-filter && python make_figures.py
```
