# Method

## Supervised members

Each frozen representation supplies either a recording vector or a temporal sequence. Vector members include an RBF SVM, logistic regression and an MLP. Temporal members project frame features into a compact state, encode their order, and pool over valid frames. Attention MIL uses a bidirectional recurrent encoder; the SER member supports a Transformer or recurrent encoder with multi-head attentive statistics pooling. Aligned feature streams can be concatenated before temporal encoding.

Imputation and normalization are fitted on L. Training weights balance domain/class cells and the known recording groups within each cell. Vector members receive sample weights; neural training uses weighted replacement sampling. V selects the vector regularization strength or neural checkpoint. Optional arousal targets belong to L only.

## Per-class rank fusion

For N recordings and C classes, member m assigns a score s(m,i,c). Rank recordings separately for each member and class. The stable ordinal rank r(m,i,c) runs from 0 to N-1; ties follow sorted recording IDs. Average r/(N-1) over members, then normalize each recording's C averaged ranks by their sum. The predicted class is the largest normalized rank unless a separately declared binary prior rule is used.

This operation preserves predictions under strictly increasing member/class score transformations, with the members, recording batch and tie rule held fixed. It fits no fusion weights. Ranks are recomputed over the supplied batch, so batch composition is part of the prediction protocol.

## Validation-only member choice

For each available V unit, score each member using its single-member rank-fusion output and the primary metric. Average individual scores across units, rank the candidates, and compare the top-3 and top-4 sets using fused V scores. Average candidate scores across eligible units with sufficient shared recordings. Ties prefer fewer members and then deterministic lexical order.

The canine full-budget policy retains the predefined full pool. The other supplied study policies use V-selected sets; the lowest-budget ungulate policy also uses per-species V priors. Config files identify these policies and archived member names without distributing experimental outputs. Pooled validation and leave-one-repeat-out decisions must be distinguished by the caller; the selector provides explicit repeat exclusion.

## Information boundaries

| Stage | Information consumed |
| --- | --- |
| Upstream frozen extraction | Audio and pretrained encoder, no task labels |
| Supervised fit and normalization | L features, labels and available grouping information |
| Hyperparameter/checkpoint choice | V features and labels |
| Fusion-member choice | V member scores, V labels and domain identifiers |
| Prior estimation | Binary V labels and domain identifiers |
| Frozen prediction and ranking | Shared unlabelled batch, fitted members and frozen policy |
| Binary quantile decision | Unlabelled batch scores, domain identifiers and V priors |
| Final evaluation | Saved predictions and held-out labels |

No target labels enter `predict`, `fuse`, or `apply_prior`. A held-out prediction batch is used jointly by the rank operation and, if enabled, the quantile decision. It is not single-record inference. The caller must keep evaluation artifacts out of model and policy selection.

## Evaluation

For each declared class, compute recall within each domain. UAR is the average across the full declared class list, assigning zero recall to absent classes. The primary score averages these domain UARs equally. Pooled UAR computes recall after combining domains and is reported separately.

For an out-of-fold evaluation, concatenate all expected folds of one repeat and verify exact, duplicate-free coverage before calculating its metric. Summaries use sample standard deviation across complete repeats, never fold-level standard deviation. No statistical significance claim is inferred from the software example.

For the supervised SER recipe, valence mixup retains the original sequence mask. The optional arousal loss uses the original sampled L target, and activates only when at least ten L arousal labels are available. No auxiliary label is taken from V or the prediction batch. Architecture and training defaults are configurable source defaults; checkpoint files and metadata record the actual run settings.
