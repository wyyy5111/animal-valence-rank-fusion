"""Phase-separated commands for training, selection, prediction and evaluation."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from .fusion import rank_fusion
from .selection import select_members
from .prior import estimate_priors, apply_prior
from .metrics import label_metrics
from .io import read_json, write_json, index_rows, load_features, check_split


def artifact(path):
    value = read_json(path)
    index_rows(value["predictions"])
    names = value["label_names"]
    if (not isinstance(names, list) or len(names) < 2 or
        any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names)):
        raise ValueError("Artifacts require an ordered list of unique classes")
    for row in value["predictions"]:
        scores = np.asarray(row["probabilities"], dtype=float)
        if scores.shape != (len(names),) or not np.isfinite(scores).all():
            raise ValueError("Score columns must match the declared class order")
        if "y_pred" in row and (isinstance(row["y_pred"], bool) or
            not isinstance(row["y_pred"], int) or not 0 <= row["y_pred"] < len(names)):
            raise ValueError("Predicted indices must belong to the declared classes")
    return value


def fit(args):
    from .members import fit_vector
    train, validation = read_json(args.train), read_json(args.validation)
    check_split(train, validation)
    names = args.labels
    for row in train + validation:
        if not 0 <= row["y_true"] < len(names):
            raise ValueError("Label index is outside declared class order")
    x_train = load_features(args.cache, train)
    x_val = load_features(args.cache, validation)
    options = dict(train_domains=[r["domain"] for r in train],
                   val_domains=[r["domain"] for r in validation],
                   train_groups=[r.get("group", r["id"]) for r in train],
                   label_names=names, seed=args.seed)
    if args.architecture in ("svm", "logistic"):
        model = fit_vector(x_train, [r["y_true"] for r in train], x_val,
                           [r["y_true"] for r in validation], kind=("linear" if args.architecture == "logistic" else "svm"), **options)
        backend = "vector"
    else:
        from .neural import fit_neural
        model = fit_neural(x_train, [r["y_true"] for r in train], x_val,
                           [r["y_true"] for r in validation], architecture=args.architecture,
                           config=read_json(args.config) if args.config else None,
                           training=read_json(args.training) if args.training else None,
                           device=args.device, **options)
        backend = "neural"
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    write_json(str(args.output) + ".json", {"backend": backend, "architecture": args.architecture,
               "label_names": names, "training_ids": [r["id"] for r in train],
               "validation_ids": [r["id"] for r in validation], "metadata": model.metadata})


def predict(args):
    metadata = read_json(str(args.model) + ".json")
    if metadata["backend"] == "vector":
        from .members import load_vector
        model = load_vector(args.model)
    else:
        from .neural import load_neural
        model = load_neural(args.model, device=args.device)
    # Deliberately discard labels before applying a fitted model.
    rows = [{k: r[k] for k in ("id", "domain", "audio_sha256") if k in r}
            for r in read_json(args.records)]
    index_rows(rows)
    features = load_features(args.cache, rows)
    scores, decisions = model.predict_proba(features), model.predict(features)
    output = [{"id": r["id"], "domain": r["domain"], "probabilities": p.tolist(), "y_pred": int(d)}
              for r, p, d in zip(rows, scores, decisions)]
    write_json(args.output, {"member": args.member, "label_names": metadata["label_names"], "predictions": output})


def select(args):
    value = read_json(args.validation_units)
    result = select_members(value["units"], value["label_names"],
                            candidate_sizes=args.sizes, min_shared=args.min_shared,
                            exclude_repeat=args.exclude_repeat)
    result["label_names"] = value["label_names"]
    write_json(args.output, result)


def fuse(args):
    members = [artifact(path) for path in args.predictions]
    if len({m["member"] for m in members}) != len(members):
        raise ValueError("Member names must be unique")
    names = members[0]["label_names"]
    if any(m["label_names"] != names for m in members):
        raise ValueError("Member class orders differ")
    if args.selection:
        selection = read_json(args.selection)
        if selection.get("label_names", names) != names:
            raise ValueError("Selection and prediction class orders differ")
        chosen = selection["chosen_members"]
        by_name = {m["member"]: m for m in members}
        if not set(chosen) <= by_name.keys():
            raise ValueError("A selected member has no prediction artifact")
        members = [by_name[name] for name in chosen]
    maps = [index_rows(m["predictions"]) for m in members]
    if any(set(maps[0]) != set(other) for other in maps[1:]):
        raise ValueError("Prediction batches must have identical recording IDs")
    for key, row in maps[0].items():
        if any(other[key]["domain"] != row["domain"] for other in maps[1:]):
            raise ValueError("Member domains differ")
    ids, scores = rank_fusion([{key: r["probabilities"] for key, r in m.items()} for m in maps])
    rows = [{"id": key, "domain": maps[0][key]["domain"], "probabilities": p.tolist(), "y_pred": int(p.argmax())}
            for key, p in zip(ids, scores)]
    if args.priors:
        if len(names) != 2:
            raise ValueError("Prior matching is binary only")
        rows = apply_prior(rows, read_json(args.priors))
    write_json(args.output, {"member": "rank_fusion", "members": [m["member"] for m in members],
                            "label_names": names, "predictions": rows})


def evaluate(args):
    predictions = artifact(args.predictions)
    truth = index_rows(read_json(args.labels), True)
    rows = predictions["predictions"]
    if {r["id"] for r in rows} != set(truth):
        raise ValueError("Prediction and evaluation IDs must match exactly")
    if any(r["domain"] != truth[r["id"]]["domain"] for r in rows):
        raise ValueError("Evaluation domains differ")
    scores = np.asarray([r["probabilities"] for r in rows])
    metrics = label_metrics([truth[r["id"]]["y_true"] for r in rows],
                            [r.get("y_pred", int(scores[i].argmax())) for i, r in enumerate(rows)],
                            [r["domain"] for r in rows], predictions["label_names"], scores)
    write_json(args.output, metrics)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("fit", help="Fit one supervised member on L and select on V")
    p.add_argument("--cache", nargs="+", required=True)
    p.add_argument("--train", required=True)
    p.add_argument("--validation", required=True)
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--architecture", choices=["svm", "logistic", "mlp", "mil", "ser"], required=True)
    p.add_argument("--config"); p.add_argument("--training")
    p.add_argument("--device", default="cpu"); p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--output", required=True); p.set_defaults(action=fit)
    p = commands.add_parser("predict", help="Apply a fitted member without reading labels")
    p.add_argument("--model", required=True); p.add_argument("--cache", nargs="+", required=True)
    p.add_argument("--records", required=True); p.add_argument("--member", required=True)
    p.add_argument("--device", default="cpu"); p.add_argument("--output", required=True)
    p.set_defaults(action=predict)
    p = commands.add_parser("select", help="Select fusion members using V-only prediction units")
    p.add_argument("--validation-units", required=True); p.add_argument("--sizes", nargs="+", type=int, default=[3, 4])
    p.add_argument("--min-shared", type=int, default=10); p.add_argument("--exclude-repeat", type=int)
    p.add_argument("--output", required=True); p.set_defaults(action=select)
    p = commands.add_parser("prior", help="Estimate binary domain priors using V labels")
    p.add_argument("--validation", required=True); p.add_argument("--output", required=True)
    p.set_defaults(action=lambda a: write_json(a.output, estimate_priors(read_json(a.validation))))
    p = commands.add_parser("fuse", help="Fuse an unlabelled batch using a frozen member list")
    p.add_argument("--predictions", nargs="+", required=True); p.add_argument("--selection")
    p.add_argument("--priors"); p.add_argument("--output", required=True); p.set_defaults(action=fuse)
    p = commands.add_parser("evaluate", help="Evaluate saved, fixed decisions")
    p.add_argument("--predictions", required=True); p.add_argument("--labels", required=True)
    p.add_argument("--output", required=True); p.set_defaults(action=evaluate)
    args = parser.parse_args(argv)
    args.action(args)
