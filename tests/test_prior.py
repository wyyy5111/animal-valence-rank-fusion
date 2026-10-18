import unittest

import numpy as np

from animal_valence.prior import apply_prior, estimate_priors


class PriorTests(unittest.TestCase):
    def rows(self, domain="a"):
        return [{"id": str(index), "domain": domain, "probabilities": [1 - p, p]}
                for index, p in enumerate((.1, .2, .8, .9))]

    def test_estimate_uses_validation_labels_by_domain(self):
        rows = [{"domain": "a", "y_true": 1}, {"domain": "a", "y_true": 0},
                {"domain": "a", "y_true": 1}, {"domain": "b", "y_true": 0}]
        self.assertEqual(estimate_priors(rows), {"a": 2 / 3, "b": 0})
        for rows in ([], [{"domain": "a", "y_true": 2}], [{"domain": "a", "y_true": None}]):
            with self.assertRaises(ValueError):
                estimate_priors(rows)

    def test_quantile_rule_needs_no_target_truth(self):
        rows = self.rows()
        original = [dict(row) for row in rows]
        actual = apply_prior(rows, {"a": .5})
        self.assertEqual([row["y_pred"] for row in actual], [0, 0, 1, 1])
        self.assertEqual(rows, original)
        self.assertEqual([row["id"] for row in actual], ["0", "1", "2", "3"])
        # Supplying arbitrary target labels cannot affect the decisions.
        labeled = [{**row, "y_true": object()} for row in rows]
        self.assertEqual([row["y_pred"] for row in apply_prior(labeled, {"a": .5})], [0, 0, 1, 1])

    def test_rate_edges_unseen_domain_and_ties(self):
        for rate, expected in ((0, [0] * 4), (1, [1] * 4)):
            self.assertEqual([row["y_pred"] for row in apply_prior(self.rows(), {"a": rate})], expected)
        self.assertEqual([row["y_pred"] for row in apply_prior(self.rows(), {})], [0, 0, 1, 1])
        tied = [{"domain": "a", "probabilities": [.5, .5]} for _ in range(4)]
        self.assertEqual([row["y_pred"] for row in apply_prior(tied, {"a": .25})], [1] * 4)

    def test_domain_order_preserved(self):
        a, b = self.rows("a"), self.rows("b")
        rows = [a[0], b[0], a[1], b[1]]
        result = apply_prior(rows, {"a": 0, "b": 1})
        self.assertEqual([row["domain"] for row in result], ["a", "b", "a", "b"])
        self.assertEqual([row["y_pred"] for row in result], [0, 1, 0, 1])

    def test_invalid_rates_and_scores(self):
        for priors in ({"a": -1}, {"a": 2}, {"a": np.nan}):
            with self.assertRaises(ValueError):
                apply_prior(self.rows(), priors)
        for p in ([np.nan, .5], [.1, .1], [1], [-1, 2]):
            with self.assertRaises(ValueError):
                apply_prior([{"domain": "a", "probabilities": p}], {"a": .5})


if __name__ == "__main__":
    unittest.main()
