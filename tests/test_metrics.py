import unittest

import numpy as np

from animal_valence.metrics import combine_oof, label_metrics, summarize_repeats


class MetricsTests(unittest.TestCase):
    def test_fixed_class_macro_and_domain_weighting(self):
        result = label_metrics([0, 0, 1, 0], [0, 1, 1, 1],
                               ["large", "large", "large", "small"], ["negative", "positive"])
        self.assertEqual(result["confusion"], [[1, 2], [0, 1]])
        self.assertAlmostEqual(result["uar"], 2 / 3)
        self.assertAlmostEqual(result["macro_f1"], .5)
        self.assertAlmostEqual(result["accuracy"], .5)
        self.assertAlmostEqual(result["primary"], .375)
        self.assertEqual(result["primary"], result["macro_domain_uar"])
        self.assertEqual(result["per_domain"]["small"]["confusion"], [[0, 1], [0, 0]])
        self.assertEqual(result["per_domain"]["small"]["missing_classes"], ["positive"])

    def test_absent_classes_remain_in_denominator(self):
        result = label_metrics([0, 0], [0, 0], label_names=["zero", "one", "two"])
        self.assertEqual(result["accuracy"], 1)
        self.assertAlmostEqual(result["uar"], 1 / 3)
        self.assertAlmostEqual(result["macro_f1"], 1 / 3)
        self.assertEqual(result["confusion"], [[2, 0, 0], [0, 0, 0], [0, 0, 0]])

    def test_probabilities_do_not_override_decisions(self):
        result = label_metrics([0, 1], [1, 0], label_names=["negative", "positive"],
                               probabilities=[[.9, .1], [.1, .9]])
        self.assertEqual(result["accuracy"], 0)
        self.assertEqual(result["proba_argmax_accuracy"], 1)
        self.assertAlmostEqual(result["cross_entropy"], -np.log(.9))

    def test_invalid_labels_and_probability_matrix(self):
        for truth, pred in (([0, 1], [.2, 1]), ([0, 2], [0, 1]), ([], []), ([0, 1], [0])):
            with self.subTest(truth=truth, pred=pred), self.assertRaises(ValueError):
                label_metrics(truth, pred, label_names=["a", "b"])
        for p in ([[2, -1], [0, 1]], [[.2, .2], [.1, .9]], [[np.nan, 0], [0, 1]]):
            with self.assertRaises(ValueError):
                label_metrics([0, 1], [0, 1], label_names=["a", "b"], probabilities=p)
        with self.assertRaises(ValueError):
            label_metrics([0], [0], label_names=["a", "a"])

    def test_oof_requires_exactly_one_complete_repeat(self):
        folds = [[{"id": "c", "y_true": 1}], {"predictions": [{"id": "a", "y_true": 0}]},
                 [{"id": "b", "y_true": 1}]]
        joined = combine_oof(folds, ["a", "b", "c"])
        self.assertEqual([row["id"] for row in joined], ["a", "b", "c"])
        self.assertEqual(joined[0]["y_true"], 0)
        for invalid, expected in ((folds[:2], ["a", "b", "c"]),
                                  (folds, ["a", "b", "missing"]),
                                  (folds, ["a", "a", "b", "c"]),
                                  ([[{"id": "a"}], [{"id": "a"}], [{"id": "c"}]], ["a", "b", "c"])):
            with self.assertRaises(ValueError):
                combine_oof(invalid, expected)

    def test_repeat_summary_uses_sample_standard_deviation(self):
        result = summarize_repeats([{"primary": .2}, {"primary": .4}, {"primary": .6}])
        self.assertEqual(result["n_repeats"], 3)
        self.assertAlmostEqual(result["primary"]["mean"], .4)
        self.assertAlmostEqual(result["primary"]["std"], .2)
        self.assertEqual(summarize_repeats([{"primary": 1}])["primary"]["std"], 0)
        for values in ([], [{"primary": np.nan}]):
            with self.assertRaises(ValueError):
                summarize_repeats(values)


if __name__ == "__main__":
    unittest.main()
