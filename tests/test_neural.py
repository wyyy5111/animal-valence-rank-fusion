import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

try:
    import torch
    from animal_valence.neural import AttentionMIL, SERSequenceNet, fit_neural, load_neural
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Optional PyTorch dependency is not installed")
class NeuralMemberTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        rng = np.random.default_rng(3)
        self.x = rng.normal(size=(8, 4)).astype(np.float32)
        self.y = np.asarray([0, 1] * 4)
        self.v = rng.normal(size=(4, 4)).astype(np.float32)
        self.yv = [0, 1, 0, 1]
        self.sequences = [rng.normal(size=(i % 3 + 2, 4)).astype(np.float32) for i in range(8)]
        self.validation = [rng.normal(size=(i % 3 + 2, 4)).astype(np.float32) for i in range(4)]
        self.training = {"max_epochs": 2, "min_epochs": 1, "patience": 1, "batch": 4}

    def _roundtrip(self, member, features):
        probabilities = member.predict_proba(features, batch_size=2)
        np.testing.assert_allclose(probabilities.sum(1), 1, atol=1e-6)
        self.assertTrue(np.isfinite(probabilities).all())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "member.pt"
            member.save(path)
            restored = load_neural(path)
            np.testing.assert_allclose(restored.predict_proba(features, batch_size=2), probabilities,
                                       atol=1e-6)
        self.assertLessEqual(member.metadata["best_epoch"], 2)
        from animal_valence.metrics import label_metrics
        score = label_metrics(self.yv, probabilities.argmax(1), label_names=["0", "1"])
        self.assertEqual(score["primary"], member.metadata["selected_validation"]["primary"])

    def test_mlp_fits_l_statistics_and_restores_checkpoint(self):
        member = fit_neural(self.x, self.y, self.v + 100, self.yv, architecture="mlp",
                            config={"hidden": [8, 4], "dropout": 0}, training=self.training)
        np.testing.assert_allclose(member.normalizer.mean, self.x.astype(np.float64).mean(0), atol=1e-6)
        self._roundtrip(member, self.v + 100)

    def test_attention_mil_variable_lengths(self):
        member = fit_neural(self.sequences, self.y, self.validation, self.yv, architecture="mil",
                            config={"hidden": 4, "pooling": "attn_max", "dropout": 0},
                            training=self.training)
        self._roundtrip(member, self.validation)
        raw = np.concatenate(self.sequences).astype(np.float64)
        np.testing.assert_allclose(member.normalizer.mean, raw.mean(0), atol=1e-6)
        self.assertEqual(member.normalizer.mean.dtype, np.float32)
        frames = np.concatenate(self.sequences)
        np.testing.assert_array_equal(member.normalizer.mean, frames.mean(0))
        np.testing.assert_array_equal(member.normalizer.std, frames.std(0))

    def test_ser_transformer_auxiliary_mixup(self):
        member = fit_neural(self.sequences * 2, self.y.tolist() * 2, self.validation, self.yv, architecture="ser",
                            config={"dim": 8, "layers": 1, "heads": 2, "pool_heads": 2, "dropout": 0},
                            arousal_targets=[0, 1, 2, -1, 0, 1, 2, -1] * 2, training=self.training)
        self.assertEqual(member.metadata["arousal_auxiliary"]["n_labeled_train"], 12)
        self._roundtrip(member, self.validation)

    def test_original_ser_mixup_mask_and_unmixed_auxiliary_targets(self):
        from animal_valence import neural
        sequences = self.sequences * 2
        arousal = np.asarray([0, 1, 2, -1] * 4)
        seed = 17
        indices = np.random.default_rng(seed).choice(16, size=16, replace=True,
                                                     p=np.full(16, 1 / 16))
        lengths = np.asarray([len(sequences[i]) for i in indices])
        expected_mask = np.arange(lengths.max())[None, :] < lengths[:, None]
        captured_masks = []
        original_forward = SERSequenceNet.forward

        def capture_forward(net, frames, mask):
            if net.training:
                captured_masks.append(mask.detach().cpu().numpy().copy())
            return original_forward(net, frames, mask)

        with patch.object(SERSequenceNet, "forward", capture_forward), \
                patch.object(neural, "_auxiliary_loss", wraps=neural._auxiliary_loss) as auxiliary:
            member = fit_neural(sequences, self.y.tolist() * 2, self.validation, self.yv,
                                architecture="ser", seed=seed,
                                config={"dim": 8, "layers": 1, "heads": 2, "pool_heads": 1, "dropout": 0},
                                arousal_targets=arousal,
                                training={"batch": 16, "max_epochs": 1, "min_epochs": 1})
        self.assertTrue(member.metadata["arousal_auxiliary"]["enabled"])
        self.assertEqual(auxiliary.call_count, 1)
        np.testing.assert_array_equal(auxiliary.call_args.args[1].cpu().numpy(), arousal[indices])
        np.testing.assert_array_equal(captured_masks[0], expected_mask)

    def test_original_auxiliary_requires_ten_labeled_training_rows(self):
        member = fit_neural(self.sequences, self.y, self.validation, self.yv, architecture="ser",
                            config={"dim": 8, "encoder": "bigru", "pool_heads": 1},
                            arousal_targets=[0, 1, 2, -1, 0, 1, 2, -1], training=self.training)
        self.assertFalse(member.metadata["arousal_auxiliary"]["enabled"])

    def test_ser_bigru_main_path(self):
        member = fit_neural(self.sequences, self.y, self.validation, self.yv, architecture="ser",
                            config={"dim": 8, "encoder": "bigru", "pool_heads": 1, "dropout": 0},
                            training={**self.training, "mixup_alpha": 0})
        self._roundtrip(member, self.validation)

    def test_padding_does_not_change_outputs(self):
        frames = torch.randn(2, 4, 4)
        mask = torch.tensor([[True, True, False, False], [True, True, True, True]])
        changed = frames.clone()
        changed[0, 2:] = 9999
        nets = [AttentionMIL(4, hidden=4, pooling=pooling, dropout=0)
                for pooling in ("mean", "attn", "attn_max")]
        nets += [SERSequenceNet(4, dim=8, heads=2, layers=1, pool_heads=2, dropout=0,
                                encoder=encoder) for encoder in ("transformer", "bigru")]
        for net in nets:
            net.eval()
            with torch.no_grad():
                first, second = net(frames, mask), net(changed, mask)
                if isinstance(first, tuple):
                    first, second = first[0], second[0]
                torch.testing.assert_close(first, second)

    def test_invalid_inputs_rejected(self):
        with self.assertRaises(ValueError):
            fit_neural(self.x, self.y, self.v, self.yv, config={"unknown": 3})
        with self.assertRaises(ValueError):
            fit_neural(self.x, self.y, self.v, self.yv, training={"max_epochs": 0})
        with self.assertRaises(ValueError):
            fit_neural(self.sequences, self.y, self.validation, self.yv, architecture="mil",
                       arousal_targets=[0] * 8)
        with self.assertRaises(ValueError):
            fit_neural([np.empty((0, 4))], [0], self.validation, self.yv, architecture="mil")


if __name__ == "__main__":
    unittest.main()
