"""Lightweight supervised heads over frozen vector or frame features.

Only L fits preprocessing and parameters. Visible V selects the checkpoint by
domain-equal UAR and macro-F1. Import this optional module when PyTorch is needed.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .members import LOnlyNormalizer, _domains, _matrix, prepare_labels, training_weights
from .metrics import label_metrics


class FeatureMLP(nn.Module):
    """GELU multilayer classifier for pooled frozen vectors."""

    def __init__(self, input_dim, hidden=(512, 256), classes=2, dropout=0.2):
        super().__init__()
        layers, previous = [], input_dim
        for width in hidden:
            layers.extend([nn.Linear(previous, width), nn.GELU(), nn.Dropout(dropout)])
            previous = width
        self.net = nn.Sequential(*layers, nn.Linear(previous, classes))

    def forward(self, x):
        return self.net(x)


class AttentionMIL(nn.Module):
    """Frame projection, packed BiGRU and masked mean/attention pooling."""

    def __init__(self, input_dim, hidden=128, classes=2, dropout=0.2,
                 rnn_layers=1, pooling="mean"):
        super().__init__()
        if pooling not in {"mean", "attn", "attn_max"}:
            raise ValueError("pooling must be mean, attn or attn_max")
        self.pooling = pooling
        self.proj = nn.Sequential(nn.Linear(input_dim, hidden), nn.GELU(), nn.Dropout(dropout))
        self.rnn = nn.GRU(hidden, hidden, num_layers=rnn_layers, batch_first=True,
                          bidirectional=True, dropout=dropout if rnn_layers > 1 else 0.0)
        if pooling != "mean":
            self.attn = nn.Sequential(nn.Linear(2 * hidden, 128), nn.Tanh(), nn.Linear(128, 1))
        self.head = nn.Sequential(nn.Dropout(dropout),
                                  nn.Linear((4 if pooling == "attn_max" else 2) * hidden, classes))

    def forward(self, frames, mask):
        lengths = _valid_lengths(frames, mask)
        x = self.proj(frames)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True,
                                                   enforce_sorted=False)
        states, _ = self.rnn(packed)
        states, _ = nn.utils.rnn.pad_packed_sequence(states, batch_first=True,
                                                     total_length=frames.shape[1])
        if self.pooling == "mean":
            pooled = (states * mask.unsqueeze(-1)).sum(1) / lengths.unsqueeze(-1)
        else:
            scores = self.attn(states).squeeze(-1).masked_fill(~mask, float("-inf"))
            pooled = (states * torch.softmax(scores, dim=1).unsqueeze(-1)).sum(1)
            if self.pooling == "attn_max":
                maximum = states.masked_fill(~mask.unsqueeze(-1), float("-inf")).max(1).values
                pooled = torch.cat([pooled, maximum], dim=-1)
        return self.head(pooled)


def _valid_lengths(frames, mask):
    if frames.ndim != 3 or mask.shape != frames.shape[:2] or mask.dtype != torch.bool:
        raise ValueError("Expected B x T x D frames and a B x T Boolean mask")
    lengths = mask.sum(1)
    if (lengths <= 0).any():
        raise ValueError("Every sequence needs at least one valid frame")
    expected = torch.arange(frames.shape[1], device=mask.device)[None, :] < lengths[:, None]
    if not torch.equal(mask, expected):
        raise ValueError("Masks must describe contiguous frames followed by right padding")
    return lengths


class MultiHeadAttentiveStatsPooling(nn.Module):
    """Attention-weighted mean and standard deviation for every head."""

    def __init__(self, dim, heads=4):
        super().__init__()
        self.attention = nn.Linear(dim, heads)

    def forward(self, x, mask):
        scores = self.attention(x).masked_fill(~mask.unsqueeze(-1), float("-inf"))
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        expanded = x.unsqueeze(2)
        mean = (weights * expanded).sum(1)
        variance = (weights * (expanded - mean.unsqueeze(1)).square()).sum(1)
        return torch.cat([mean, torch.sqrt(variance + 1e-6)], dim=-1).flatten(1)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        positions = torch.arange(x.shape[1], device=x.device, dtype=x.dtype).unsqueeze(1)
        frequencies = torch.exp(torch.arange(0, self.dim, 2, device=x.device, dtype=x.dtype)
                                 * (-math.log(10000.0) / self.dim))
        pe = torch.zeros(x.shape[1], self.dim, device=x.device, dtype=x.dtype)
        pe[:, 0::2] = torch.sin(positions * frequencies)
        pe[:, 1::2] = torch.cos(positions * frequencies[:self.dim // 2])
        return x + pe.unsqueeze(0)


class SERSequenceNet(nn.Module):
    """Transformer/BiGRU context with multi-head attentive statistics pooling."""

    def __init__(self, input_dim, dim=256, layers=2, heads=4, pool_heads=4,
                 classes=2, dropout=0.2, encoder="transformer", rnn_layers=1,
                 arousal_classes=0):
        super().__init__()
        if encoder not in {"transformer", "bigru"}:
            raise ValueError("encoder must be transformer or bigru")
        if dim % heads != 0 and encoder == "transformer":
            raise ValueError("Transformer dim must be divisible by heads")
        if dim % 2 and encoder == "bigru":
            raise ValueError("BiGRU dim must be even")
        self.proj = nn.Sequential(nn.Linear(input_dim, dim), nn.GELU(), nn.Dropout(dropout))
        self.encoder = encoder
        if encoder == "transformer":
            self.pos = SinusoidalPositionalEncoding(dim)
            layer = nn.TransformerEncoderLayer(dim, heads, 2 * dim, dropout,
                                               batch_first=True, norm_first=False)
            self.context = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        else:
            self.context = nn.GRU(dim, dim // 2, num_layers=rnn_layers, batch_first=True,
                                  bidirectional=True, dropout=dropout if rnn_layers > 1 else 0.0)
        self.pool = MultiHeadAttentiveStatsPooling(dim, pool_heads)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(2 * pool_heads * dim, classes)
        self.arousal_head = nn.Linear(2 * pool_heads * dim, arousal_classes) if arousal_classes else None

    def forward(self, frames, mask):
        lengths = _valid_lengths(frames, mask)
        x = self.proj(frames)
        if self.encoder == "transformer":
            x = self.context(self.pos(x), src_key_padding_mask=~mask)
        else:
            packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True,
                                                       enforce_sorted=False)
            x, _ = self.context(packed)
            x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True,
                                                   total_length=frames.shape[1])
        pooled = self.dropout(self.pool(x, mask))
        return self.head(pooled), self.arousal_head(pooled) if self.arousal_head is not None else None


def _sequences(values, dimension=None):
    try:
        sequences = [_matrix(value, dimension=dimension, name="frame sequence") for value in values]
    except TypeError:
        raise ValueError("Expected a sequence of nonempty T x D feature matrices") from None
    if not sequences or any(s.shape[1] != sequences[0].shape[1] for s in sequences):
        raise ValueError("Frame sequences must share a nonzero feature dimension")
    if any(not np.isfinite(sequence).all() for sequence in sequences):
        raise ValueError("Frame features must be finite; missing-value imputation is vector-only")
    return sequences


def _fit_frame_normalizer(sequences):
    """Retain the original frame-cache float32 preprocessing arithmetic."""
    frames = np.concatenate(sequences, axis=0).astype(np.float32)
    if not np.isfinite(frames).all():
        raise ValueError("Frame features must be representable as finite float32 values")
    mean, std = frames.mean(axis=0), frames.std(axis=0)
    std = np.where(std > 1e-8, std, 1.0)
    return LOnlyNormalizer(np.zeros_like(mean), mean, std)


def _collate(sequences, device):
    frames = torch.zeros(len(sequences), max(len(s) for s in sequences), sequences[0].shape[1])
    mask = torch.zeros(frames.shape[:2], dtype=torch.bool)
    for i, sequence in enumerate(sequences):
        frames[i, :len(sequence)] = torch.as_tensor(sequence, dtype=torch.float32)
        mask[i, :len(sequence)] = True
    return frames.to(device), mask.to(device)


_CONFIG_DEFAULTS = {
    "mlp": {"hidden": [512, 256], "dropout": 0.2},
    "mil": {"hidden": 128, "dropout": 0.2, "rnn_layers": 1, "pooling": "mean"},
    "ser": {"dim": 256, "layers": 2, "heads": 4, "pool_heads": 4, "dropout": 0.2,
            "encoder": "transformer", "rnn_layers": 1},
}


def _configuration(architecture, config):
    if architecture not in _CONFIG_DEFAULTS:
        raise ValueError("architecture must be mlp, mil or ser")
    settings = dict(_CONFIG_DEFAULTS[architecture])
    unknown = set(config or {}) - set(settings)
    if unknown:
        raise ValueError(f"Unknown {architecture} configuration keys: {sorted(unknown)}")
    settings.update(config or {})
    if architecture == "mlp":
        hidden = settings["hidden"]
        if not isinstance(hidden, (list, tuple)) or not hidden:
            raise ValueError("MLP hidden must be a nonempty list of positive integers")
        settings["hidden"] = [_positive_int(value, "hidden") for value in hidden]
    for key in {"dim", "layers", "heads", "pool_heads", "rnn_layers"} & settings.keys():
        settings[key] = _positive_int(settings[key], key)
    if architecture == "mil":
        settings["hidden"] = _positive_int(settings["hidden"], "hidden")
    settings["dropout"] = float(settings["dropout"])
    if not np.isfinite(settings["dropout"]) or not 0 <= settings["dropout"] < 1:
        raise ValueError("dropout must be in [0, 1)")
    return settings


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _training(architecture, training):
    settings = {"batch": 64 if architecture == "mlp" else 16,
                "max_epochs": 100 if architecture == "mlp" else 60,
                "min_epochs": 20 if architecture == "mlp" else 10,
                "patience": 15 if architecture == "mlp" else 12,
                "lr": 1e-3, "weight_decay": 1e-4,
                "label_smoothing": 0.1 if architecture == "ser" else 0.0,
                "mixup_alpha": 0.2 if architecture == "ser" else 0.0,
                "arousal_weight": 0.3}
    unknown = set(training or {}) - set(settings)
    if unknown:
        raise ValueError(f"Unknown training keys: {sorted(unknown)}")
    settings.update(training or {})
    for key in ("batch", "max_epochs", "min_epochs", "patience"):
        settings[key] = _positive_int(settings[key], key)
    settings["min_epochs"] = min(settings["min_epochs"], settings["max_epochs"])
    for key in ("lr", "weight_decay", "label_smoothing", "mixup_alpha", "arousal_weight"):
        settings[key] = float(settings[key])
        if not np.isfinite(settings[key]) or settings[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if settings["lr"] == 0 or settings["label_smoothing"] >= 1:
        raise ValueError("lr must be positive and label_smoothing must be less than one")
    if architecture != "ser" and (settings["mixup_alpha"] or settings["label_smoothing"]):
        raise ValueError("Mixup and label smoothing belong to the supervised SER path")
    return settings


def _build_net(architecture, input_dim, classes, config, arousal_classes=0):
    if architecture == "mlp":
        return FeatureMLP(input_dim, classes=classes, **config)
    if architecture == "mil":
        return AttentionMIL(input_dim, classes=classes, **config)
    return SERSequenceNet(input_dim, classes=classes, arousal_classes=arousal_classes, **config)


@dataclass
class NeuralMember:
    net: nn.Module
    normalizer: LOnlyNormalizer
    label_names: tuple[str, ...]
    metadata: dict
    architecture: str
    config: dict
    arousal_classes: int = 0

    def predict_proba(self, features, batch_size=64):
        batch_size = _positive_int(batch_size, "batch_size")
        if self.architecture == "mlp":
            values = self.normalizer.transform(features).astype(np.float32)
        else:
            values = [self.normalizer.transform(s).astype(np.float32)
                      for s in _sequences(features, len(self.normalizer.mean))]
        return _predict(self.net, self.architecture, values, batch_size)

    def predict(self, features):
        return self.predict_proba(features).argmax(axis=1)

    def save(self, path):
        """Save tensors and primitive metadata, compatible with weights-only loading."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"format": "animal-valence-neural-v1", "architecture": self.architecture,
                    "config": self.config, "arousal_classes": self.arousal_classes,
                    "input_dim": len(self.normalizer.mean), "label_names": list(self.label_names),
                    "normalizer": {key: getattr(self.normalizer, key).tolist()
                                   for key in ("median", "mean", "std")},
                    "metadata": self.metadata,
                    "state_dict": {key: value.detach().cpu() for key, value in self.net.state_dict().items()}}, path)


def load_neural(path, device="cpu"):
    """Restore a saved supervised member without unpickling arbitrary classes."""
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    if bundle.get("format") != "animal-valence-neural-v1":
        raise ValueError("Unsupported neural artifact format")
    config = _configuration(bundle["architecture"], bundle["config"])
    net = _build_net(bundle["architecture"], bundle["input_dim"], len(bundle["label_names"]),
                     config, bundle["arousal_classes"])
    net.load_state_dict(bundle["state_dict"])
    net.to(device).eval()
    dtype = np.float64 if bundle["architecture"] == "mlp" else np.float32
    normalizer = LOnlyNormalizer(**{key: np.asarray(value, dtype=dtype)
                                   for key, value in bundle["normalizer"].items()})
    return NeuralMember(net, normalizer, tuple(bundle["label_names"]), bundle["metadata"],
                        bundle["architecture"], config, bundle["arousal_classes"])


def _predict(net, architecture, values, batch_size):
    net.eval()
    device = next(net.parameters()).device
    probabilities = []
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            batch = values[start:start + batch_size]
            if architecture == "mlp":
                logits = net(torch.as_tensor(batch, dtype=torch.float32, device=device))
            else:
                frames, mask = _collate(batch, device)
                outputs = net(frames, mask)
                logits = outputs[0] if architecture == "ser" else outputs
            probabilities.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(probabilities)


def _auxiliary_loss(logits, targets):
    valid = targets >= 0
    if not valid.any():
        return logits.sum() * 0
    return F.cross_entropy(logits[valid], targets[valid])


def fit_neural(x_train, y_train, x_val, y_val, *, train_domains=None, val_domains=None,
               train_groups=None, label_names=None, architecture="mlp", config=None,
               training=None, device="cpu", seed=2026, arousal_targets=None,
               arousal_classes=3):
    """Train one frozen-feature head with V checkpoint selection and early stopping.

    ``mlp`` consumes N x D vectors. ``mil`` and ``ser`` consume lists of T_i x D
    matrices. Optional auxiliary arousal labels are L-only integers (``-1`` is
    missing), available for SER. No U or test labels enter this interface.
    """
    config = _configuration(architecture, config)
    settings = _training(architecture, training)
    torch_device = torch.device(device)
    if torch_device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; choose device='cpu'")
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; choose device='cpu'")
    if architecture == "mlp":
        raw_train = _matrix(x_train)
        raw_val = _matrix(x_val, dimension=raw_train.shape[1])
        normalizer = LOnlyNormalizer.fit(raw_train)
        train_x = normalizer.transform(raw_train).astype(np.float32)
        val_x = normalizer.transform(raw_val).astype(np.float32)
    else:
        raw_train = _sequences(x_train)
        raw_val = _sequences(x_val, raw_train[0].shape[1])
        normalizer = _fit_frame_normalizer(raw_train)
        train_x = [normalizer.transform(s).astype(np.float32) for s in raw_train]
        val_x = [normalizer.transform(s).astype(np.float32) for s in raw_val]
    train_y, val_y, names = prepare_labels(y_train, len(train_x), y_val, len(val_x), label_names)
    domains = _domains(val_domains, len(val_x), "val_domains")
    weights = training_weights(train_y, train_domains, train_groups)
    probability = weights / weights.sum()
    aux_targets = None
    if arousal_targets is not None:
        if architecture != "ser":
            raise ValueError("Auxiliary arousal labels are supported only for SER")
        arousal_classes = _positive_int(arousal_classes, "arousal_classes")
        raw_aux = np.asarray(arousal_targets)
        try:
            aux_targets = raw_aux.astype(np.int64)
        except (TypeError, ValueError, OverflowError):
            raise ValueError("arousal_targets must be integer L-only labels") from None
        if (raw_aux.shape != (len(train_x),) or not np.array_equal(raw_aux, aux_targets)
                or np.any((aux_targets < -1) | (aux_targets >= arousal_classes))):
            raise ValueError("arousal_targets must align with L and use -1 or declared class indices")
        # Original supervised SER enables the task only with at least ten
        # labeled-training arousal annotations.
        if int((aux_targets >= 0).sum()) < 10:
            aux_targets = None
    if aux_targets is None:
        arousal_classes = 0
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    rng = np.random.default_rng(seed)
    net = _build_net(architecture, len(normalizer.mean), len(names), config, arousal_classes).to(torch_device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"])
    steps = max(1, math.ceil(len(train_x) / settings["batch"]))
    history, best, stale = [], None, 0
    for epoch in range(1, settings["max_epochs"] + 1):
        net.train()
        order = rng.choice(len(train_x), size=settings["batch"] * steps, replace=True, p=probability)
        for step in range(steps):
            idx = order[step * settings["batch"]:(step + 1) * settings["batch"]]
            targets = torch.as_tensor(train_y[idx], dtype=torch.long, device=torch_device)
            if architecture == "mlp":
                frames = torch.as_tensor(train_x[idx], dtype=torch.float32, device=torch_device)
                mask = None
            else:
                frames, mask = _collate([train_x[int(i)] for i in idx], torch_device)
            lam, perm = 1.0, None
            if settings["mixup_alpha"] > 0:
                lam = float(rng.beta(settings["mixup_alpha"], settings["mixup_alpha"]))
                perm = torch.randperm(len(idx), device=torch_device)
                frames = lam * frames + (1 - lam) * frames[perm]
                # Match the original supervised SER: keep the original mask.
            optimizer.zero_grad(set_to_none=True)
            outputs = net(frames) if mask is None else net(frames, mask)
            logits = outputs[0] if architecture == "ser" else outputs
            loss = lam * F.cross_entropy(logits, targets, label_smoothing=settings["label_smoothing"])
            if perm is not None:
                loss = loss + (1 - lam) * F.cross_entropy(logits, targets[perm],
                                                         label_smoothing=settings["label_smoothing"])
            if architecture == "ser" and aux_targets is not None:
                aux = torch.as_tensor(aux_targets[idx], dtype=torch.long, device=torch_device)
                # The published main path mixes valence targets only; its
                # auxiliary CE uses the original sampled L arousal targets.
                if (aux >= 0).any():
                    aux_loss = _auxiliary_loss(outputs[1], aux)
                    loss = loss + settings["arousal_weight"] * aux_loss
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss; inspect input features and training settings")
            loss.backward()
            optimizer.step()
        probabilities = _predict(net, architecture, val_x, settings["batch"])
        metric = label_metrics(val_y, probabilities.argmax(1), domains=domains,
                               label_names=names, probabilities=probabilities)
        history.append({"epoch": epoch, "validation_primary": metric["primary"],
                        "validation_macro_f1": metric["macro_f1"]})
        key = (metric["primary"], metric["macro_f1"])
        if best is None or key > best[0]:
            best = (key, copy.deepcopy(net.state_dict()), epoch, metric)
            stale = 0
        else:
            stale += 1
        if epoch >= settings["min_epochs"] and stale >= settings["patience"]:
            break
    net.load_state_dict(best[1])
    net.eval()
    metadata = {"kind": architecture, "seed": int(seed), "device": str(torch_device),
                "config": config, "training": settings, "best_epoch": best[2], "epochs_run": epoch,
                "selection_rule": "visible-V domain-equal UAR, then macro-F1; ties keep earliest epoch",
                "selected_validation": {key: best[3][key] for key in ("primary", "macro_f1", "accuracy")},
                "history": history, "decision_rule": "probability_argmax",
                "preprocessing_fit": ("L only: median imputation and float64 standardization"
                                      if architecture == "mlp" else "L only: float32 frame standardization"),
                "sampling": "weighted replacement; equal domain-class cells and groups within cells",
                "label_names": list(names), "n_train": len(train_x), "n_validation": len(val_x),
                "input_dim": len(normalizer.mean),
                "arousal_auxiliary": {"enabled": aux_targets is not None,
                                      "n_labeled_train": int((aux_targets >= 0).sum()) if aux_targets is not None else 0},
                "n_parameters": sum(value.numel() for value in net.parameters())}
    return NeuralMember(net, normalizer, names, metadata, architecture, config, arousal_classes)
