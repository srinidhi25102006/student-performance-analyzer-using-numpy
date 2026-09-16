"""
test_analyzer.py

Unit test suite for the Student Performance Analyzer.
Tests all NumPy analysis functions in analyzer.py and input validation in data.py.
"""

import sys
import os
import unittest
import numpy as np

# Add src directory to path to enable clean imports when running tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from data import get_sample_dataset, validate_mark
from analyzer import (
    convert_to_array,
    get_array_properties,
    calculate_student_totals,
    calculate_student_averages,
    assign_grades,
    determine_pass_fail,
    calculate_subject_averages,
    get_subject_highest,
    get_subject_lowest,
    get_top_student,
    get_lowest_student,
    get_high_performers,
    calculate_class_std,
    rank_students,
    count_pass_fail,
    get_best_subject,
    get_worst_subject,
    calculate_subject_deviations,
)


class TestStudentPerformanceAnalyzer(unittest.TestCase):
    """Test suite verifying correctness of NumPy computations."""

    def setUp(self) -> None:
        """Sets up default test data prior to each test method."""
        self.students, self.subjects, self.marks_list = get_sample_dataset()
        self.marks_array = convert_to_array(self.marks_list)

    def test_convert_to_array(self) -> None:
        """Tests list to NumPy array conversion and data type."""
        self.assertIsInstance(self.marks_array, np.ndarray)
        self.assertEqual(self.marks_array.shape, (5, 5))
        self.assertEqual(self.marks_array.dtype, np.float64)

    def test_get_array_properties(self) -> None:
        """Tests extraction of array shape, size, ndim, dtype."""
        props = get_array_properties(self.marks_array)
        self.assertEqual(props["shape"], (5, 5))
        self.assertEqual(props["size"], 25)
        self.assertEqual(props["ndim"], 2)
        self.assertEqual(props["dtype"], "float64")

    def test_calculate_student_totals(self) -> None:
        """Tests row-wise sum calculation (student total marks)."""
        totals = calculate_student_totals(self.marks_array)
        expected_totals = np.array([438.0, 390.0, 457.0, 350.0, 292.0])
        np.testing.assert_array_almost_equal(totals, expected_totals)

    def test_calculate_student_averages(self) -> None:
        """Tests row-wise mean calculation (student average marks)."""
        averages = calculate_student_averages(self.marks_array)
        expected_averages = np.array([87.6, 78.0, 91.4, 70.0, 58.4])
        np.testing.assert_array_almost_equal(averages, expected_averages)

    def test_assign_grades(self) -> None:
        """Tests grade assignment based on average cutoffs."""
        averages = calculate_student_averages(self.marks_array)
        grades = assign_grades(averages)
        expected_grades = np.array(['A', 'B', 'A+', 'B', 'D'])
        np.testing.assert_array_equal(grades, expected_grades)

    def test_determine_pass_fail(self) -> None:
        """Tests pass/fail determination (pass require >=40 in all subjects)."""
        status = determine_pass_fail(self.marks_array, pass_mark=40.0)
        expected_status = np.array(['Pass', 'Pass', 'Pass', 'Pass', 'Pass'])
        np.testing.assert_array_equal(status, expected_status)

        # Test fail case when one subject < 40
        fail_marks = np.array([[85, 35, 92, 88, 95]])
        fail_status = determine_pass_fail(fail_marks, pass_mark=40.0)
        self.assertEqual(fail_status[0], 'Fail')

    def test_calculate_subject_averages(self) -> None:
        """Tests column-wise mean calculation (subject class averages)."""
        subj_avg = calculate_subject_averages(self.marks_array)
        expected_subj_avg = np.array([73.4, 76.2, 76.4, 79.0, 80.4])
        np.testing.assert_array_almost_equal(subj_avg, expected_subj_avg)

    def test_get_subject_highest_and_lowest(self) -> None:
        """Tests finding highest and lowest mark in each subject."""
        highest = get_subject_highest(self.marks_array)
        lowest = get_subject_lowest(self.marks_array)

        expected_highest = np.array([90.0, 93.0, 92.0, 94.0, 95.0])
        expected_lowest = np.array([55.0, 60.0, 58.0, 62.0, 57.0])

        np.testing.assert_array_almost_equal(highest, expected_highest)
        np.testing.assert_array_almost_equal(lowest, expected_lowest)

    def test_get_top_and_lowest_student(self) -> None:
        """Tests finding student with highest and lowest average using argmax/argmin."""
        averages = calculate_student_averages(self.marks_array)
        top_name, top_avg = get_top_student(self.students, averages)
        low_name, low_avg = get_lowest_student(self.students, averages)

        self.assertEqual(top_name, "Karthik")
        self.assertAlmostEqual(top_avg, 91.4)
        self.assertEqual(low_name, "Rahul")
        self.assertAlmostEqual(low_avg, 58.4)

    def test_get_high_performers(self) -> None:
        """Tests boolean mask filtering for averages >= threshold."""
        averages = calculate_student_averages(self.marks_array)
        performers = get_high_performers(self.students, averages, threshold=75.0)

        expected = [("Arun", 87.6), ("Divya", 78.0), ("Karthik", 91.4)]
        self.assertEqual(len(performers), len(expected))
        for (name, avg), (exp_name, exp_avg) in zip(performers, expected):
            self.assertEqual(name, exp_name)
            self.assertAlmostEqual(avg, exp_avg)

    def test_calculate_class_std(self) -> None:
        """Tests overall standard deviation calculation."""
        std_val = calculate_class_std(self.marks_array)
        self.assertTrue(std_val > 0)
        self.assertAlmostEqual(std_val, 12.55681488, places=4)

    def test_rank_students(self) -> None:
        """Tests index sorting (np.argsort) for ranking students."""
        averages = calculate_student_averages(self.marks_array)
        rankings = rank_students(self.students, averages)

        # Karthik (91.4), Arun (87.6), Divya (78.0), Meena (70.0), Rahul (58.4)
        expected_names = ["Karthik", "Arun", "Divya", "Meena", "Rahul"]
        actual_names = [name for rank, name, avg in rankings]
        self.assertEqual(actual_names, expected_names)

    def test_count_pass_fail(self) -> None:
        """Tests counting passed and failed student statuses."""
        pass_fail_array = np.array(['Pass', 'Pass', 'Fail', 'Pass'])
        p_count, f_count = count_pass_fail(pass_fail_array)
        self.assertEqual(p_count, 3)
        self.assertEqual(f_count, 1)

    def test_get_best_and_worst_subject(self) -> None:
        """Tests argmax/argmin for best and worst performing subject."""
        subj_avg = calculate_subject_averages(self.marks_array)
        best_name, best_avg = get_best_subject(self.subjects, subj_avg)
        worst_name, worst_avg = get_worst_subject(self.subjects, subj_avg)

        self.assertEqual(best_name, "Computer Networks")
        self.assertAlmostEqual(best_avg, 80.4)
        self.assertEqual(worst_name, "Python")
        self.assertAlmostEqual(worst_avg, 73.4)

    def test_calculate_subject_deviations(self) -> None:
        """Tests broadcasting subtractions (marks_array - subject_averages)."""
        subj_avg = calculate_subject_averages(self.marks_array)
        deviations = calculate_subject_deviations(self.marks_array, subj_avg)

        # For Arun: Python (85 - 73.4 = +11.6)
        self.assertAlmostEqual(deviations[0, 0], 11.6)

    def test_validate_mark_valid(self) -> None:
        """Tests mark validator with valid inputs."""
        self.assertEqual(validate_mark("0"), 0.0)
        self.assertEqual(validate_mark("100"), 100.0)
        self.assertEqual(validate_mark("85.5"), 85.5)

    def test_validate_mark_invalid(self) -> None:
        """Tests mark validator with out of bounds or non-numeric inputs."""
        with self.assertRaises(ValueError):
            validate_mark("-5")
        with self.assertRaises(ValueError):
            validate_mark("105")
        with self.assertRaises(ValueError):
            validate_mark("abc")


if __name__ == "__main__":
    unittest.main()
