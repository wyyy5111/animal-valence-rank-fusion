import subprocess
import sys
import tempfile
from pathlib import Path
import unittest

import numpy as np

from animal_valence.members import LOnlyNormalizer, fit_vector, load_vector, training_weights


class VectorMemberTests(unittest.TestCase):
    def setUp(self):
        self.x = np.asarray([[-2, 1, np.nan], [-1, 2, np.nan], [-1.5, 1.5, np.nan],
                             [1, 5, np.nan], [2, 4, np.nan], [1.5, 4.5, np.nan]])
        self.y = np.asarray([0, 0, 0, 1, 1, 1])
        self.v = np.asarray([[-1.2, 2, 20], [1.2, 4, 20]])

    def test_training_statistics_do_not_include_validation(self):
        model = fit_vector(self.x, self.y, self.v, [0, 1], c_values=(1,), label_names=["negative", "positive"])
        shifted = fit_vector(self.x, self.y, self.v + 10000, [0, 1], c_values=(1,))
        np.testing.assert_array_equal(model.normalizer.median, [0, 3, 0])
        np.testing.assert_array_equal(model.normalizer.mean, [0, 3, 0])
        np.testing.assert_array_equal(model.normalizer.mean, shifted.normalizer.mean)
        self.assertEqual(model.predict_proba(self.v).shape, (2, 2))
        np.testing.assert_allclose(model.predict_proba(self.v).sum(1), 1)

    def test_svm_native_decision_and_roundtrip(self):
        model = fit_vector(self.x, self.y, self.v, [0, 1], kind="svm", c_values=(0.1, 1, 10))
        expected = model.estimator.predict(model.normalizer.transform(self.v))
        np.testing.assert_array_equal(model.predict(self.v), expected)
        self.assertEqual(model.metadata["decision_rule"], "native_predict")
        trials = model.metadata["selection_trials"]
        best = max(enumerate(trials), key=lambda pair: (pair[1]["validation_primary"],
                                                       pair[1]["validation_macro_f1"], -pair[0]))[1]
        self.assertEqual(model.metadata["selected_C"], best["C"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "member.joblib"
            model.save(path)
            restored = load_vector(path)
            np.testing.assert_array_equal(restored.predict(self.v), expected)
            np.testing.assert_allclose(restored.predict_proba(self.v), model.predict_proba(self.v))

    def test_cell_and_group_balancing(self):
        y = [0, 0, 0, 1, 1, 0]
        domains = ["a", "a", "a", "a", "a", "b"]
        groups = ["animal1", "animal1", "animal2", None, None, None]
        weights = training_weights(y, domains, groups)
        self.assertAlmostEqual(weights.mean(), 1)
        self.assertAlmostEqual(weights[0] + weights[1], weights[2])
        self.assertAlmostEqual(weights[:3].sum(), weights[3:5].sum())
        self.assertAlmostEqual(weights[:3].sum(), weights[5])

    def test_linear_and_invalid_inputs(self):
        member = fit_vector(self.x, self.y, self.v, [0, 1], kind="linear", c_values=(1,))
        np.testing.assert_array_equal(member.predict(self.v), member.predict_proba(self.v).argmax(1))
        alias = fit_vector(self.x, self.y, self.v, [0, 1], kind="logistic", c_values=(1,))
        np.testing.assert_allclose(alias.predict_proba(self.v), member.predict_proba(self.v))
        with self.assertRaises(ValueError):
            fit_vector(self.x, self.y, self.v, [0, 1], c_values=(0,))
        with self.assertRaises(ValueError):
            fit_vector(self.x, self.y, self.v, [0, 1], label_names=["a", "b", "c"])
        with self.assertRaises(ValueError):
            fit_vector(self.x, self.y, self.v, [0, 1], val_domains=["too short"])
        with self.assertRaises(ValueError):
            LOnlyNormalizer.fit([[np.inf, 0]])

    def test_vector_module_does_not_import_torch(self):
        result = subprocess.run([sys.executable, "-c",
                                 "import sys; import animal_valence.members; assert 'torch' not in sys.modules"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
