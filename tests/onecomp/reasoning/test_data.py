import unittest

from onecomp.reasoning.data import extract_answer, grade_answer


class DataTests(unittest.TestCase):
    def test_balanced_box_and_numeric_equivalence(self):
        self.assertEqual(extract_answer(r"x \boxed{\frac{1}{2}}"), r"\frac{1}{2}")
        self.assertTrue(grade_answer(r"\boxed{\frac{1}{2}}", ".5"))
        self.assertTrue(grade_answer(r"\boxed{042}", "42"))
        self.assertFalse(grade_answer("I tried 42\nbut still thinking 73", "73"))
        self.assertFalse(grade_answer("", ""))
        self.assertTrue(grade_answer("Answer: B.", "B"))


if __name__ == "__main__":
    unittest.main()
