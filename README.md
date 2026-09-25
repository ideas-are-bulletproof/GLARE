# GLARE: Graph Learning through Affinity-guided REwiring

Anonymous code release accompanying the submission *"GLARE: Graph Learning through Affinity-Guided Rewiring for Heterophilic Node Classification"* (under double-blind review).

GLARE is a classifier-agnostic rewiring method for heterophilic graphs. It alternates, in an EM-style loop, between learning node embeddings and pseudo-labels on the current graph and re-estimating the edges from them. It returns two outputs that any downstream classifier can consume:

- a **rewired graph** `Ã`, and
- a **node embedding** `Z` learned on that graph, fused with the input features as `X̂ = [ℓ2(X) ‖ ℓ2(Z)]`.

A fully unsupervised variant, **GLARE-U**, uses no labels of any split during rewiring.

## Method overview

1. **Similarity encoder.** An MLP is pretrained with NT-Xent on two feature-dropout views (no labels, no edges), propagated M hops over the graph, and optionally fine-tuned on visible training labels.
2. **Candidate pool and affinity.** Candidates are the original edges, 2-hop pairs, kNN pairs under the learned similarity, and random negatives. Each is scored by `w = λ_S · S + (1 − λ_S) · Φ`, where Φ is a neighbourhood-distribution affinity.
3. **EM loop (T outer iterations).**
   - *E-step:* train a weighted two-layer GraphSAGE on the current soft graph to get embeddings `H` and soft pseudo-labels.
   - *M-step:* update per-edge logits by minimising a modularity term plus a pseudo-label homophily term, regularised by entropy, a prior anchoring the original edges, and a degree budget. The temperature is cosine-annealed.
4. **Outputs.** Threshold the edge weights to get `Ã`, compute the two-hop embedding `Z` on `Ã`, and pass `(X̂, Ã)` to any classifier.

## Repository contents

| File | Purpose |
|---|---|
| `glare_benchmark_final.py` | Main GLARE pipeline and the 5-classifier harness (GCN, GAT, GraphSAGE, H2GCN, LINKX). Controls label visibility with `--label_mask_ratio`. |
| `glare_benchmark_final_unsupervised.py` | Same pipeline with the `--unsupervised` switch used for GLARE-U (dummy labels, empty train/val masks, last iterate kept). |
| `glare_fusion_ablation.py` | 2×2 ablation separating the rewired graph from the embedding (original/rewired graph × `ℓ2(X)`/`X̂`), with optional random-feature and graph-free MLP controls. |
| `benchmark_others.py` | Baselines, each with the classifier from its own paper: IDGL, DHGR, ComFy, FoSR, LPkG (plus GADC and GRAPHITE). |
| `homophily_benchmark.py` | Structural and feature homophily of original vs. rewired graphs (edge, node, adjusted, class-insensitive, LI). |
| `propagation_benchmark.py` | Label propagation, BFS reachability, and community detection / clustering on rewired graphs and embeddings. |
| `tolokers_auc_benchmark.py` | ROC-AUC on Tolokers for all methods. Loads the main scripts unchanged and applies small, verified in-memory hooks to read test scores. |

## Requirements

- Python ≥ 3.9
- PyTorch and PyTorch Geometric
- numpy, scipy, scikit-learn, networkx, pandas, matplotlib

```bash
pip install torch torch_geometric numpy scipy scikit-learn networkx pandas matplotlib
```

The scripts import a small helper package, `common/` (metrics, progress, checkpointing, reporting, plotting), which must sit next to the scripts. The DHGR baseline additionally needs the official DHGR repository; place it in `./DHGR` or point to it with the `DHGR_ROOT` environment variable.

## Datasets

All six benchmarks are downloaded automatically through PyTorch Geometric into `./data` and use their fixed public splits:

Actor, Squirrel-F, Chameleon-F (filtered versions without duplicate-node leakage), Roman-empire, Amazon-ratings, Tolokers.

## Usage

**GLARE (supervised, main results):**
```bash
python glare_benchmark_final.py --device cuda --label_mask_ratio 1.0
```

**GLARE-U (fully unsupervised rewiring):**
```bash
python glare_benchmark_final_unsupervised.py --device cuda --unsupervised
```

**Ablation (graph vs. embedding):** run from the same directory so cached GLARE rewirings are reused.
```bash
python glare_fusion_ablation.py --device cuda
python glare_fusion_ablation.py --summarize_only   # rebuild tables only
```

**Baselines:**
```bash
python benchmark_others.py --device cuda
python benchmark_others.py --methods idgl dhgr --datasets Actor Squirrel-F
```

**Homophily, propagation and Tolokers AUC:**
```bash
python homophily_benchmark.py --seeds 0 1 2 --device cuda
python propagation_benchmark.py --device cuda
python tolokers_auc_benchmark.py --device cuda
```

Most scripts accept `--datasets`, `--seeds`, `--classifiers` / `--methods`, `--device` and `--smoke_test` for a quick sanity run. GLARE hyperparameters are exposed as `--glare_*` / `--glare18_*` flags; defaults match the paper (see `--help`).

## Outputs and resuming

Each run appends one record to a JSONL file, so an interrupted sweep resumes where it stopped. GLARE rewirings are cached per (dataset, seed) and shared across classifiers and the ablation.

| Script | Output folder |
|---|---|
| GLARE / GLARE-U | `glare_benchmark_results/` (checkpoints in `checkpoints/`) |
| Ablation | `glare_fusion_ablation_results/` |
| Baselines | `others_benchmark_results/` |
| Homophily | `homophily_results/` |
| Propagation | `others_benchmark_results_propagation/` |
| Tolokers AUC | `tolokers_auc_results/` |

Tables (CSV/Markdown) and figures are regenerated at the end of each run.

## Main results

Mean test accuracy over the five downstream classifiers (3 seeds):

| Graph | Actor | Squirrel-F | Chameleon-F | Roman-emp. | Amazon-rat. | Tolokers | Mean |
|---|---|---|---|---|---|---|---|
| Original, ℓ2(X) | 0.315 | 0.334 | 0.463 | 0.701 | 0.484 | 0.788 | 0.514 |
| GLARE | 0.351 | 0.507 | 0.643 | 0.657 | 0.482 | 0.795 | 0.573 |
| GLARE-U (no labels) | 0.319 | 0.505 | 0.637 | 0.656 | 0.459 | 0.793 | 0.562 |

GLARE improves accuracy in 23 of 30 classifier–dataset combinations (mean +5.8 points) and reduces the spread across classifiers roughly fourfold. It does not help on Roman-empire and parts of Amazon-ratings, where the original graph combined with the GLARE embedding is the better choice. See the paper for full tables, baselines and ablations.

## Reproducibility notes

- Seeds `{0, 1, 2}`, fixed public splits, results reported as mean ± std.
- Downstream classifiers share one harness: hidden size 128, dropout 0.5, label smoothing 0.1, cosine-annealed Adam, gradient clipping, 300 epochs, validation-based model selection every 5 epochs.
- GLARE-U never reads a label of any split during rewiring; an assertion in the code enforces this. Only the downstream classifiers use training labels.

## License and citation

Code is released for review purposes. License and citation information will be added after the review period.
