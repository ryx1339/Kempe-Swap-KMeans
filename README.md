# Kempe-Swap-KMeans (KSKM)

A scalable, near-optimal algorithm for semi-supervised (constrained) clustering under hard **must-link (ML)** and **cannot-link (CL)** constraints.

> Based on the paper: **"Kempe Swap K-Means: A Scalable Near-Optimal Solution for Semi-Supervised Clustering"**  
> Yuxuan Ren, Shijie Deng — Georgia Institute of Technology  
> [arXiv:2603.27417](https://arxiv.org/abs/2603.27417)

---

## Overview

Standard K-Means ignores prior knowledge about data relationships. **KSKM** incorporates pairwise constraints:

- **Must-link (ML):** two data points must be in the same cluster.
- **Cannot-link (CL):** two data points must be in different clusters.

KSKM bridges the gap between greedy heuristics (e.g., COP-K-Means) and expensive integer-programming-based methods by leveraging **Kempe chain swaps** — a classical technique from graph coloring — to efficiently explore the constrained solution space while always maintaining feasibility.

---

## Algorithm

### Core Idea

Constrained clustering under CL constraints is equivalent to a **graph k-coloring problem**, where:
- **Vertices** are super-nodes formed by merging must-linked data points (exploiting ML transitivity).
- **Edges** represent cannot-link relationships between super-nodes.
- **Colors** correspond to cluster assignments.

KSKM iterates between two phases (K-Means style):

1. **Assignment step (KSAssignment):** Identifies all improving **Kempe chains** between cluster pairs and selects a compatible subset by solving a **Maximum Weight Independent Set (MWIS)** problem. This guarantees that each swap preserves the valid k-coloring (feasibility is never violated).

2. **Centroid update step:** Recomputes cluster centroids from the current assignment, minimizing WCSS.

### Escape from Local Optima

To avoid poor local optima, KSKM applies two controlled centroid mutations:

- **Centroid perturbation (KSPerturb):** Samples perturbed centroids from the empirical cluster distribution (multivariate t-distribution), where clusters with higher variance receive stronger perturbations.
- **Centroid reposition (KSShift):** Repositions the best-fitting cluster centroid to the worst-fitting cluster's location for larger jumps.

After each mutation, KSAssignment is re-applied to restore local optimality.

### Initialization

Cluster assignments are initialized using a **DSATUR-inspired heuristic** (Algorithm 5): vertices are processed in DSATUR order (non-increasing saturation degree) and assigned to their nearest feasible centroid. Centroids are initialized via **k-means++**.

### Extension: KSKM-E

`KSKM_E` adds an optional **exact GCP refinement step** (via Gurobi ILP) after Kempe-swap descent converges. This helps escape Kempe-swap local optima when the chromatic number is close to k, at the cost of additional computation.

---

## Installation

### Requirements

```
numpy
scipy
scikit-learn
numba
gurobipy  # requires a valid Gurobi license
```

Install dependencies:

```bash
pip install numpy scipy scikit-learn numba gurobipy
```

> **Note:** Gurobi requires a license. Academic licenses are available free of charge at [gurobi.com](https://www.gurobi.com/academia/academic-program-and-licenses/).

---

## Usage

```python
import numpy as np
from KSKM import KSKM

# Data points to cluster
data = np.array([...])  # shape (n, d)

# Pairwise constraints (0-indexed)
ml = [(2, 6), (9, 10)]   # must-link pairs
cl = [(2, 7), (6, 7)]    # cannot-link pairs

# Run KSKM
membership = KSKM(
    random_state=42,
    data=data,
    ml=ml,
    cl=cl,
    steps_mutation=200,    # number of perturbation iterations
    k=4,                   # number of clusters
    steps_back_to_best=5,  # revert to best after this many non-improving steps
    steps_no_improvement=10,
    verbose=False,
    time_limit=3600        # seconds
)
```

See [`example.py`](example.py) for a complete runnable example with evaluation via Adjusted Rand Index (ARI).

---

## API Reference

### `KSKM(random_state, data, ml, cl, steps_mutation, k, steps_back_to_best, steps_no_improvement, verbose, time_limit)`

Main entry point. Runs the full KSKM algorithm (Kempe-swap local search with centroid mutations).

**Returns:** `ndarray` of shape `(n,)` — cluster label for each data point (0-indexed), or `[]` if no feasible solution was found.

---

### `KSKM_E(random_state, data, ml, cl, steps_mutation, k, steps_back_to_best, steps_no_improvement, verbose, time_limit)`

Extended version with Gurobi-based exact GCP refinement. Produces higher quality solutions at the cost of longer runtime.

**Returns:** `ndarray` of shape `(n,)` — cluster label for each data point.

---

## Key Internal Components

| Function | Role |
|---|---|
| `preprocessing` | Merges ML groups into super-nodes; builds CL graph |
| `sub_adj_classification` | Decomposes constraint graph into singletons, cliques, and general components |
| `DSATUR_KM` | Cost-aware DSATUR initialization (nearest feasible centroid) |
| `KSAssignment` | Core Kempe-swap assignment via MWIS (Algorithm 2) |
| `build_MWSP_core` | Enumerates candidate swaps and conflict structure (Numba-accelerated) |
| `KempeChainMWSP` | Full descent loop: alternates centroid update + KSAssignment |
| `CentroidUpdate` | Recomputes centroids and distance matrix; optionally performs reposition |
| `CentroidUpdate_random` | Samples perturbed centroids for KSPerturb mutation |
| `ExactAssignment` | Gurobi-based GCP refinement (used in KSKM-E) |
| `recover_ml_from_membership` | Expands super-node assignments back to individual data points |

---

## Performance

KSKM consistently outperforms state-of-the-art constrained clustering benchmarks in:
- **Solution quality** (WCSS, ARI)
- **Computational efficiency** (especially for large-scale, dense constraint graphs)

The MWIS subproblem solved per iteration is significantly cheaper than the full fixed-centroid graph coloring ILP used by competing methods, enabling KSKM to scale to datasets where ILP-based approaches are intractable.

---

## Citation

```bibtex
@article{ren2026kskm,
  title   = {Kempe Swap K-Means: A Scalable Near-Optimal Solution for Semi-Supervised Clustering},
  author  = {Ren, Yuxuan and Deng, Shijie},
  journal = {arXiv preprint arXiv:2603.27417},
  year    = {2026}
}
```
