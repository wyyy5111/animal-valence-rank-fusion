import unittest

import numpy as np

from animal_valence.fusion import rank_fusion


class RankFusionTests(unittest.TestCase):
    def setUp(self):
        self.scores = {"c": [.6, .4], "b": [.4, .6], "a": [.4, .6], "d": [.7, .3]}

    def test_stable_ordinal_ties_and_sorted_ids(self):
        ids, actual = rank_fusion([self.scores])
        self.assertEqual(ids, ["a", "b", "c", "d"])
        np.testing.assert_allclose(actual, [[0, 1], [.25, .75], [2 / 3, 1 / 3], [1, 0]])
        reordered = dict(reversed(list(self.scores.items())))
        other_ids, other = rank_fusion([reordered], ids=["d", "c", "b", "a"])
        self.assertEqual(ids, other_ids)
        np.testing.assert_array_equal(actual, other)

    def test_strictly_monotone_transforms_preserve_fusion(self):
        transformed = {key: [np.exp(3 * row[0]), 100 + 7 * row[1]] for key, row in self.scores.items()}
        _, original = rank_fusion([self.scores, self.scores])
        _, actual = rank_fusion([transformed, self.scores])
        np.testing.assert_array_equal(actual, original)

    def test_class_axis_and_shared_batch(self):
        first = {"a": [.7, .2, .1], "b": [.2, .6, .2], "c": [.1, .2, .7]}
        second = {**first, "extra": [.1, .8, .1]}
        ids, actual = rank_fusion([first, second])
        self.assertEqual(ids, ["a", "b", "c"])
        np.testing.assert_allclose(actual.sum(axis=1), 1)
        permutation = [2, 0, 1]
        permuted = {key: [row[index] for index in permutation] for key, row in first.items()}
        _, changed = rank_fusion([permuted])
        np.testing.assert_array_equal(changed, actual[:, permutation])

    def test_requested_subset_controls_ranking(self):
        ids, result = rank_fusion([self.scores], ids=["d", "a"])
        self.assertEqual(ids, ["a", "d"])
        np.testing.assert_array_equal(result, [[0, 1], [1, 0]])

    def test_undefined_or_malformed_inputs_raise(self):
        cases = [[], [{}], [{"a": [.5, .5]}],
                 [{"a": [.5, .5], "b": [.5, .5]}],
                 [{"a": [np.nan, 1], "b": [1, 0]}],
                 [{"a": [np.inf, 1], "b": [1, 0]}],
                 [{"a": [1, 0], "b": [0, .5, .5]}],
                 [{"a": [1, 0]}, {"b": [0, 1]}]]
        for maps in cases:
            with self.subTest(maps=maps), self.assertRaises(ValueError):
                rank_fusion(maps)
        for ids in (["a", "a"], ["a", "missing"]):
            with self.assertRaises(ValueError):
                rank_fusion([self.scores], ids=ids)


if __name__ == "__main__":
    unittest.main()
