# Provenance and release scope

This is a compact extraction and refactoring of the animal-valence research implementation. It keeps supervised cached-feature members, validation-only member choice, per-class rank fusion, binary prior matching, and the associated evaluation interface. It removes campaign orchestration, exploratory methods, ablation drivers, report generation and local machine configuration.

## Source mapping

| Public module | Research implementation |
| --- | --- |
| `fusion.py` | `scripts/rank_fusion.py`, `class_rank_fusion` |
| `selection.py` | `scripts/v_selected_rank_fusion.py` |
| `prior.py` | `scripts/fusion_prior_matching.py` |
| `members.py` | `animal_affect/baselines.py` |
| `neural.py` | `animal_affect/mlp.py`, `sequence.py`, `ser_model.py` |
| `metrics.py` | `animal_affect/metrics.py` and complete-repeat aggregation |
| `io.py`, `cli.py` | Portable cache interface and explicit stage separation |

The ordinal fusion kernel was compared with the original implementation on binary and three-class score arrays with different batch and member counts. Guards reject undefined rank normalization and inconsistent input metadata. Public unit tests use synthetic values only.

The research artifacts were produced through many historical runs. Current source defaults are not evidence that every historical run used identical defaults. This release does not bundle the original trained models, their predictions, cached representations, splits or full experiment registry; it is a core implementation rather than an archive of every reported experiment. The configurations describe core policies and source defaults, not newly measured results. The study's existing test-history and identity-availability disclosures remain applicable.

Only load model files from a trusted source: vector models use joblib serialization and neural models use PyTorch serialization. Audio and encoder weights should be obtained under their providers' licenses. No third-party encoder source or weights are vendored.

Forward-pass checks with identical weights and inputs matched the original MLP, MIL (mean, attention, attention-plus-max), and SER (Transformer and BiGRU) paths exactly on synthetic CPU inputs. The finite-frame normalization path also matched. SER retains the original mask during mixup and uses unmixed auxiliary arousal targets; its auxiliary head requires at least ten valid L arousal labels. No research measurements were regenerated for this software release.

## Portable API differences

A neural fit trains one explicitly supplied architecture configuration. Historical training drivers could compare several configurations; callers can perform those comparisons on V before freezing a member. Public V inference is batched, and sinusoidal positions are generated at runtime instead of read from a fixed-length buffer. The vector normalizer uses explicit float64 L statistics and a small-variance guard, whereas the original classical driver used scikit-learn preprocessing. These choices can introduce numerical differences, particularly for tiny-variance columns or across devices. Forward/kernel checks do not assert bitwise reproduction of historical training trajectories or reported measurements.
