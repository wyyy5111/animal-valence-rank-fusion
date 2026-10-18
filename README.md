# Animal Valence Rank Fusion

Core implementation of **per-class rank fusion (RF)** for low-label classification of valence in animal vocalizations. Frozen acoustic representations feed lightweight supervised members; visible validation data select a small ensemble; class-wise ordinal ranks combine their predictions without fitting fusion weights.

## Installation

Python 3.10 or later:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

For MLP, attention MIL and SER sequence members:

```bash
python -m pip install -e '.[neural]'
```

Run an end-to-end engineering example and the test suite:

```bash
python examples/synthetic_demo.py
python -m unittest discover -s tests -v
animal-valence --help
```

The example generates synthetic features in a temporary directory. Its output is a software check, not a result from either research dataset.

## Core workflow

1. Prepare frozen vector or frame caches and explicit labelled training (**L**) and visible-validation (**V**) manifests.
2. Fit each supervised member on L. Choose its hyperparameters or checkpoint on V.
3. Export member scores on V and select the top-3/top-4 ensemble using V only. Save that decision before applying it to a held-out batch.
4. Predict a shared, unlabelled batch with the selected members and fuse their class-wise ranks. The binary low-budget policy may additionally match per-domain V priors.
5. Evaluate the saved predictions against held-out labels in a separate command.

```bash
animal-valence fit --cache data/vectors.npz \
  --train data/L.json --validation data/V.json \
  --labels Negative Neutral Positive --architecture svm \
  --output models/svm.joblib

animal-valence predict --model models/svm.joblib --cache data/vectors.npz \
  --records data/batch.json --member svm --output outputs/svm.json

animal-valence select --validation-units data/validation_units.json \
  --output outputs/selection.json

animal-valence fuse --predictions outputs/svm.json outputs/mlp.json outputs/mil.json \
  --selection outputs/selection.json --output outputs/fused.json

animal-valence evaluate --predictions outputs/fused.json \
  --labels data/evaluation_labels.json --output outputs/metrics.json
```

The three artifacts in the fusion command must come from separately fitted members. The example script demonstrates how to construct validation units. See [input formats](docs/input-format.md) for the complete schema and [method](docs/method.md) for the information boundaries and decision rules.

## Included components

| Component | Implementation |
| --- | --- |
| Supervised vector members | RBF SVM, logistic regression, L-only imputation and scaling |
| Neural vector member | Deep MLP |
| Temporal members | Attention MIL, aligned dual-stream MIL inputs, SER sequence encoder with attentive statistics pooling |
| Ensemble | Per-class ordinal rank averaging and row normalization |
| Member choice | Validation-only top-k comparison; optional leave-one-repeat-out exclusion |
| Binary decisions | Per-domain prior estimates from V and unlabelled-batch quantile thresholds |
| Evaluation | Breed/species-equal UAR, pooled UAR, macro F1, confusion matrices, complete-repeat aggregation |

Multiple vector caches are concatenated by ID. Multiple sequence caches require aligned frame grids and are concatenated along the feature dimension for dual-stream MIL. `fit --architecture mlp|mil|ser` accepts architecture and training JSON files through `--config` and `--training`.

## Data and representations

Supply data obtained from their original providers: EmotionalCanines and the seven-species ungulate vocalization collection. The ungulate source is [Zenodo record 14636641](https://zenodo.org/records/14636641). Frozen encoder weights and extraction tools remain with their respective providers. This repository begins at the cached-representation interface, so different upstream encoders can share the same training and fusion implementation.

Audio, model weights, real feature caches, experimental logs, exploratory scripts and ablation campaigns are not distributed here. [Provenance and scope](docs/provenance.md) describes the relationship between this compact implementation and the research project.

## Interpretation

- The primary metric is the equally weighted mean of within-domain UAR: breeds for canine recordings and species for ungulates. It differs from pooled UAR.
- Rank fusion depends on the shared prediction batch. Fix the batch and tie handling when comparing results. The normalized ranks are scores, not calibrated posterior probabilities.
- Strictly increasing transformations preserve each member's within-class ordering. This property applies to fixed members and a fixed batch; it does not imply invariance to changing the prediction batch.
- Use known recording groups for split isolation. The canine data do not provide verifiable individual-animal identities.
- The research test sets had been inspected historically. The archived study uses a frozen report; this software's separated stages do not make that historical evaluation a blind test.

## Citation and license

Software citation metadata are provided in [CITATION.cff](CITATION.cff). The core code is released under the [MIT license](LICENSE). Dataset and pretrained-model licenses remain with their original providers.
