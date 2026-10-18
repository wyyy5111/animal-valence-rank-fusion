"""End-to-end CLI smoke check on synthetic data; not a research result."""
from pathlib import Path
import hashlib
import tempfile
import numpy as np

from animal_valence.cli import main
from animal_valence.io import read_json, write_json


def run():
    rng = np.random.default_rng(2026)
    with tempfile.TemporaryDirectory(prefix="animal-valence-demo-") as directory:
        root = Path(directory)
        records, features = [], []
        for i in range(120):
            target = i % 2
            records.append({"id": f"synthetic-{i:03d}", "domain": ["domain_a", "domain_b"][i % 3 == 0],
                            "y_true": target, "group": f"group-{i:03d}"})
            features.append(rng.normal(target * 0.7, 1, 8))
        ids = [r["id"] for r in records]
        cache = root / "features.npz"
        np.savez_compressed(cache, ids=np.asarray(ids), X=np.asarray(features),
                            audio_sha256=np.asarray([hashlib.sha256(i.encode()).hexdigest() for i in ids]))
        splits = {"L": records[:60], "V": records[60:90], "T": records[90:]}
        for key, rows in splits.items():
            write_json(root / f"{key}.json", rows)
        # The prediction command accepts features/IDs only; labels are separate.
        write_json(root / "batch.json", [{"id": r["id"], "domain": r["domain"]} for r in splits["T"]])
        units = {"label_names": ["Negative", "Positive"], "units": [{"repeat": 0, "fold": 0, "members": {}}]}
        targets = []
        for member, architecture in [("svm", "svm"), ("linear_a", "logistic"), ("linear_b", "logistic")]:
            model = root / f"{member}.joblib"
            main(["fit", "--cache", str(cache), "--train", str(root / "L.json"),
                  "--validation", str(root / "V.json"), "--labels", "Negative", "Positive",
                  "--architecture", architecture, "--seed", str(2026 + len(targets)), "--output", str(model)])
            for split, manifest in [("V", "V.json"), ("T", "batch.json")]:
                output = root / f"{member}_{split}.json"
                main(["predict", "--model", str(model), "--cache", str(cache),
                      "--records", str(root / manifest), "--member", member, "--output", str(output)])
                if split == "V":
                    labels = {r["id"]: r["y_true"] for r in splits["V"]}
                    units["units"][0]["members"][member] = [dict(r, y_true=labels[r["id"]]) for r in read_json(output)["predictions"]]
                else:
                    targets.append(str(output))
        write_json(root / "units.json", units)
        main(["select", "--validation-units", str(root / "units.json"), "--output", str(root / "selection.json")])
        main(["prior", "--validation", str(root / "V.json"), "--output", str(root / "priors.json")])
        main(["fuse", "--predictions", *targets, "--selection", str(root / "selection.json"),
              "--priors", str(root / "priors.json"), "--output", str(root / "fused.json")])
        main(["evaluate", "--predictions", str(root / "fused.json"), "--labels", str(root / "T.json"),
              "--output", str(root / "metrics.json")])
        assert len(read_json(root / "fused.json")["predictions"]) == 30
        print("Synthetic training, V selection, prior matching, fusion and evaluation: passed.")
        print("These generated data are an engineering example, not experimental evidence.")


if __name__ == "__main__":
    run()
