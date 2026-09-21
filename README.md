# GLARE

Code for **GLARE: Graph Learning through Affinity-guided Rewiring for Heterophilic Node Classification.**

On heterophilic graphs, connected nodes often have *different* labels — exactly the case where message-passing GNNs struggle. GLARE tackles this by rewiring the graph before you classify. It learns which nodes actually belong together (via a self-supervised similarity encoder and a neighbourhood-affinity signal), then runs an EM-style loop that alternates between training a small GNN and re-scoring every edge under a modularity-with-homophily objective. The result is a cleaner graph plus a fused feature representation you can drop into *any* downstream classifier.

Because labels only enter in two well-defined spots, the same code runs supervised (`--label_mask_ratio 1.0`) or completely label-free (`--label_mask_ratio 0.0`).

## What's in here

Four scripts, each self-contained. The model code is identical across them — only what's being measured changes.

- **`glare_benchmark_final.py`** — the main event. Runs GLARE with all five classifiers and reports accuracy, macro-F1 and runtime.
- **`benchmark_others.py`** — the competing rewiring / structure-learning baselines (IDGL, GADC, LPkG, DHGR, FoSR, ComFy), each paired with the classifier its own paper recommends.
- **`homophily_benchmark.py`** — rewires and reports homophily metrics for the original vs. rewired graph. No classifier trained.
- **`propagation_benchmark.py`** — label-propagation dynamics and community detection (NMI / ARI / accuracy) on the rewired graphs.

All four rely on a small `common/` package (progress bars, resumable checkpoints, tables, plots) that needs to sit next to them.

## Setup

Python 3.9+ with PyTorch and PyTorch Geometric:

```bash
pip install torch torch_geometric
pip install numpy scipy scikit-learn networkx pandas matplotlib
```

The six benchmark datasets — Actor, Squirrel-F, Chameleon-F, Roman-empire, Amazon-ratings, Tolokers — download automatically on first run.

## Running it

Every script picks up where it left off if interrupted, and there's a `--smoke_test` flag if you just want to check things work before committing to a full sweep.

```bash
# Supervised GLARE, then the label-free version
python glare_benchmark_final.py --label_mask_ratio 1.0 --device cuda
python glare_benchmark_final.py --label_mask_ratio 0.0 --device cuda

# Baselines
python benchmark_others.py --device cuda

# Analysis (appendix results)
python homophily_benchmark.py --device cuda
python propagation_benchmark.py --device cuda
```

Narrow things down anytime with `--datasets`, `--seeds`, `--methods` or `--classifiers`. Results land in each script's own folder as JSONL records, CSV tables and figures.

## A couple of things worth knowing

- The main cost is time — supervised GLARE runs an EM loop with an inner GNN, so rewiring takes a while (the label-free variant is ~2.7× faster).
- GADC and GRAPHITE don't change graph structure (they work on features), so their structural metrics match the original graph.

