# Input formats

## Feature caches

NPZ files are read with `allow_pickle=False`. IDs are unique Unicode strings. All feature blocks must refer to the same recording content for a given ID.

Vector cache:

```python
np.savez_compressed('vectors.npz', ids=np.array(ids), X=vectors,
                    audio_sha256=np.array(audio_digests))
```

`X` has shape `[recordings, features]`. Missing vector values may be NaN and are imputed from L only; infinities are rejected.

Frame cache:

```python
np.savez_compressed('frames.npz', ids=np.array(ids), X_cat=all_frames,
                    offsets=np.array(offsets, dtype=np.int64),
                    audio_sha256=np.array(audio_digests))
```

`X_cat` has shape `[total_frames, features]`; `offsets` has length `recordings + 1`, begins at zero, ends at `len(X_cat)`, and strictly increases. Frames must be finite. For aligned dual-stream caches, each recording must have the same frame count across streams. Cache preparation must align the frame grids before concatenation.

`audio_sha256` is optional for independently prepared caches. If present, it contains one hexadecimal SHA256 digest per recording. Supplied manifest digests must match. Hashes across feature blocks must agree.

## Manifests

L and V manifests are JSON arrays of rows:

```json
[
  {"id": "recording-001", "domain": "husky", "y_true": 0, "group": "session-001"},
  {"id": "recording-002", "domain": "shiba", "y_true": 2, "group": "session-002"}
]
```

`y_true` is an integer indexing the explicitly supplied class order. `group` and `audio_sha256` are optional. The fit command rejects shared L/V IDs, shared known hashes, and shared known groups. Use globally unique group identifiers. It cannot infer identities absent from source data. The caller constructs budget-specific L/V and held-out manifests according to the dataset protocol; this repository does not reconstruct unavailable individual identities or historical random splits.

Prediction manifests need only `id` and `domain` (plus an optional digest). The prediction command discards any labels before processing the records.

## Prediction artifacts

```json
{
  "member": "encoder_svm",
  "label_names": ["Negative", "Positive"],
  "predictions": [
    {"id": "recording-101", "domain": "Goat", "probabilities": [0.8, 0.2], "y_pred": 0},
    {"id": "recording-102", "domain": "Goat", "probabilities": [0.3, 0.7], "y_pred": 1}
  ]
}
```

All fused artifacts must use identical class orders, recording sets and domains. A shared batch must contain at least two recordings. A zero sum of class ranks makes row normalization undefined and raises an error. This includes degenerate constant-score batches; the library does not invent replacement scores.

The field name `probabilities` is the common serialization interface. After rank fusion it contains normalized ordinal scores, not calibrated probabilities. Native SVM decisions are retained separately as `y_pred` when evaluating a single member.

## Validation units and selected members

```json
{
  "label_names": ["Negative", "Positive"],
  "units": [
    {"repeat": 0, "fold": 0, "members": {
      "encoder_svm": [
        {"id": "V-001", "domain": "Goat", "y_true": 0, "probabilities": [0.8, 0.2]},
        {"id": "V-002", "domain": "Goat", "y_true": 1, "probabilities": [0.3, 0.7]}
      ]
    }}
  ]
}
```

The compact schema above illustrates the fields. The default candidate comparison requires at least ten shared V recordings per unit; realistic input includes the available members and all eligible units. `select` writes member names, class order, individual V scores and candidate-set V scores. `--exclude-repeat R` excludes every V unit from repeat R. Use it when constructing a decision for an excluded repeat. Never put held-out-test labels or test scores into these units.

For fixed full-pool fusion, omit `--selection` and supply the predefined member artifacts. For the binary prior rule, run `animal-valence prior --validation data/V.json --output outputs/priors.json`, then pass `--priors outputs/priors.json` to `fuse`. Column 1 must be Positive. Rates are estimated by domain from V; unknown domains use 0.5. Equal-to-threshold scores predict Positive, so ties and finite batches can prevent an exact positive fraction.
