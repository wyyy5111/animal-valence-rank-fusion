import inspect
import unittest

from animal_valence.selection import select_members


def rows(good=True):
    output = []
    for index in range(12):
        positive = .05 + .9 * index / 11
        if not good:
            positive = 1 - positive
        output.append({"id": f"row-{index:02d}", "y_true": int(index >= 6), "domain": "animal",
                       "probabilities": [1 - positive, positive]})
    return output


def unit(repeat=0, fold=0):
    return {"repeat": repeat, "fold": fold,
            "members": {"c": rows(), "b": rows(), "a": rows(), "d": rows(False)}}


class SelectionTests(unittest.TestCase):
    def test_top_three_four_rule_and_smaller_tie(self):
        result = select_members([unit()], ["negative", "positive"])
        self.assertEqual(result["chosen_members"], ["a", "b", "c"])
        self.assertEqual(result["validation_score"], 1)
        self.assertEqual(result["individual_scores"], {"a": 1, "b": 1, "c": 1, "d": 0})
        self.assertEqual([item["members"] for item in result["candidate_scores"]],
                         [["a", "b", "c"], ["a", "b", "c", "d"]])
        self.assertEqual([item["score"] for item in result["candidate_scores"]], [1, 1])

    def test_no_test_input_and_excluded_repeat_is_not_read(self):
        self.assertNotIn("test", inspect.signature(select_members).parameters)
        regular = unit()
        regular["test"] = object()
        excluded = {"repeat": 5, "fold": 0, "members": None}
        result = select_members([excluded, regular], ["negative", "positive"], exclude_repeat=5)
        self.assertEqual(result, select_members([unit()], ["negative", "positive"]))
        with self.assertRaises(TypeError):
            select_members([unit()], ["negative", "positive"], test_rows=[])

    def test_fallback_and_member_name_ties_are_deterministic(self):
        tiny = {"repeat": 0, "fold": 0, "members": {"z": rows(), "a": rows()}}
        fallback = select_members([tiny], ["negative", "positive"])
        self.assertEqual(fallback["chosen_members"], ["a", "z"])
        one = select_members([tiny], ["negative", "positive"], candidate_sizes=(1,))
        self.assertEqual(one["chosen_members"], ["a"])

    def test_shared_coverage_and_metadata_must_match(self):
        insufficient = unit()
        insufficient["members"]["b"] = rows()[:9]
        with self.assertRaisesRegex(ValueError, "enough shared"):
            select_members([insufficient], ["negative", "positive"])
        inconsistent = unit()
        inconsistent["members"]["b"][0]["domain"] = "other"
        with self.assertRaisesRegex(ValueError, "disagree"):
            select_members([inconsistent], ["negative", "positive"])

    def test_reject_duplicate_units_rows_and_wrong_classes(self):
        with self.assertRaisesRegex(ValueError, "Duplicate validation unit"):
            select_members([unit(), unit()], ["negative", "positive"])
        duplicate = unit()
        duplicate["members"]["a"].append(dict(duplicate["members"]["a"][0]))
        with self.assertRaisesRegex(ValueError, "Duplicate validation ID"):
            select_members([duplicate], ["negative", "positive"])
        with self.assertRaisesRegex(ValueError, "class count"):
            select_members([unit()], ["negative", "neutral", "positive"])


if __name__ == "__main__":
    unittest.main()
