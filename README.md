# Thermodynamic Sampling Lab

Massive random sampling can collapse the search space for sparse polynomials. This lab shows how thermodynamic sampling units (TSUs) can learn supervised models by identifying the interaction structure—then optionally export the resulting polynomial to a deterministic CPU implementation. The goal is not production quality; it is to demonstrate the style of thinking this approach enables.

## Quick Start

**What you'll see:** TSUs discovering that XOR needs a 2-way interaction, 
Iris needing 3-5 key features, and breast cancer reducing 127 potential 
edges to ~12 significant ones—then deploying each as a fast sparse polynomial.

Install the package (installs all runtime dependencies):

```bash
pip install -e .
```

Then launch the core experiments:

```bash
python -m thermolab.experiments.xor.subset_scan
python -m thermolab.experiments.xor.thrml
python -m thermolab.experiments.iris
python -m thermolab.experiments.breast_cancer
```

Optional explorations:

```bash
python -m thermolab.experiments.xor.high_order   # explicit three-spin XOR factor
python -m thermolab.experiments.ising            # toy Ising sampler + benchmarking
```

## Expected Runtime & Results

| Command | Approx. runtime* | Key metrics (train / test) | Dense vs. sparse time (speedup) |
| --- | --- | --- | --- |
| `python -m thermolab.experiments.xor.subset_scan` | < 1 s | Consistency report only | n/a |
| `python -m thermolab.experiments.xor.thrml` | ~1 s | 1.00 / 1.00 accuracy | 0.0023 s vs. 0.0012 s (1.9×) |
| `python -m thermolab.experiments.iris` | ~1 s | 1.00 / 1.00 accuracy | 0.0016 s vs. 0.0036 s (0.46×) |
| `python -m thermolab.experiments.breast_cancer` | ~1 min | 0.90 / 0.92 accuracy | 4.41 s vs. 0.24 s (18×) |

## Why These Experiments

Many TSU discussions fixate on the specific demos Extropic has demonstrated (e.g. Fashion MNIST denoising using DTMs). The key insight is broader: massive random sampling can collapse the search for high-order interactions, letting you learn sparse polynomials and move the final evaluation to conventional hardware. This lab walks through that progression.

### XOR First

Every new modelling idea I test starts with XOR—XOR is the simplest possible test of whether a method can discover non-linear structure. 

Four binary patterns—`[1,1,-1]`, `[1,-1,1]`, `[-1,1,1]`, `[-1,-1,-1]`—live on an XY plane where no linear separator exists. By scanning feature subsets we see immediately that single spins stay ambiguous while the two-spin interaction resolves the task. That’s the minimum bar: can the method recover the need for higher-order terms in the simplest non-linearly separable problem?

### Binary Classification

The Iris binary subset and the breast-cancer dataset illustrate classical “spin + polynomial” pipelines. Features are thresholded into bits, optional interaction columns are added where subset agreement indicates necessity, and pseudo-likelihood with L1 sparsity learns THRML weights. Each run reports deterministic accuracy and the speedup from dropping negligible edges—highlighting how a TSU can discover the structure and then hand it to a classical evaluator.

### Multi-Class via One-vs-Rest

When you pass `--classes 0,1,2` to `experiments/iris.py`, each run picks one class as positive and collapses the rest into a negative label. Repeating the pipeline for each class yields three polynomials that together cover the multi-class decision surface. This keeps the workflow simple while still showing how TSUs can support multi-way decisions.

### Complex Binning and Interactions

`experiments/breast_cancer.py` demonstrates richer binning. It thresholds dozens of features, then builds pairwise interaction terms for the most reliable single features. That mirrors real-world cases where you might quantise sensors into multiple bins or engineer domain-specific interactions. The lab exposes how those choices change disagreement, weight sparsity, and deterministic evaluation speed.

### Thermal-to-Classical Conversion

All experiments finish by benchmarking the learned polynomial in dense vs. sparsified form. That makes the optional conversion explicit: run the search on a TSU (or a simulator), keep only the significant edges, and deploy the lean polynomial on conventional hardware.

---

The lab is deliberately minimal. Use it as a sandbox for new binning strategies, higher-order factors, or alternative learning rules—and as a reminder that thermodynamic computing is ultimately about shrinking search, not just recreating familiar demos.

---

### Author

**Sam Martin** ([@_sammartin](https://twitter.com/_sammartin))

Licensed under the MIT License (see `LICENSE`).
