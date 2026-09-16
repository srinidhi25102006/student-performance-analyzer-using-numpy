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

import tempfile
from data import get_sample_dataset, validate_mark, load_dataset_from_csv
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
    predict_single_student,
    predict_all_students,
    get_student_subject_breakdown,
    get_all_students_subject_breakdown,
    get_overall_subject_analytics,
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

    def test_predict_single_student_improving(self) -> None:
        """Tests linear regression prediction for an improving student trend."""
        improving_marks = np.array([50.0, 60.0, 70.0, 80.0, 90.0])
        pred = predict_single_student(improving_marks)
        self.assertEqual(pred["trend"], "Improving")
        self.assertAlmostEqual(pred["predicted_score"], 100.0)
        self.assertEqual(pred["risk_level"], "Low Risk")

    def test_predict_single_student_declining(self) -> None:
        """Tests linear regression prediction for a declining student trend."""
        declining_marks = np.array([90.0, 80.0, 70.0, 60.0, 50.0])
        pred = predict_single_student(declining_marks)
        self.assertEqual(pred["trend"], "Declining")
        self.assertAlmostEqual(pred["predicted_score"], 40.0)
        self.assertEqual(pred["risk_level"], "High Risk")

    def test_predict_all_students(self) -> None:
        """Tests predicting performance trends for full class dataset."""
        preds = predict_all_students(self.students, self.marks_array)
        self.assertEqual(len(preds), 5)
        self.assertEqual(preds[0]["student"], "Arun")
        self.assertIn(preds[0]["trend"], ["Improving", "Declining", "Stable"])
        self.assertIn(preds[0]["risk_level"], ["Low Risk", "Moderate Risk", "High Risk"])

    def test_student_subject_breakdown_unique(self) -> None:
        """Tests strongest and weakest subject identification for a single student with unique max/min."""
        subjects = ["Math", "DBMS", "OS", "Python"]
        marks = np.array([85.0, 72.0, 91.0, 88.0])
        res = get_student_subject_breakdown(subjects, marks)
        self.assertEqual(res["strongest"], "OS")
        self.assertEqual(res["weakest"], "DBMS")
        self.assertFalse(res["is_strongest_tied"])
        self.assertFalse(res["is_weakest_tied"])

    def test_student_subject_breakdown_tied(self) -> None:
        """Tests strongest and weakest subject identification when subjects have tied scores."""
        subjects = ["Math", "DBMS", "OS", "Python"]
        tied_marks = np.array([90.0, 70.0, 90.0, 70.0])
        res = get_student_subject_breakdown(subjects, tied_marks)
        self.assertEqual(res["strongest"], "Math, OS")
        self.assertEqual(res["weakest"], "DBMS, Python")
        self.assertTrue(res["is_strongest_tied"])
        self.assertTrue(res["is_weakest_tied"])

    def test_overall_subject_analytics_tied(self) -> None:
        """Tests class overall subject averages and tied highest/lowest performing subjects."""
        subjects = ["Math", "DBMS", "OS", "Python"]
        # Matrix where Math & Python have same average (80.0), DBMS & OS have same average (60.0)
        matrix = np.array([
            [80.0, 60.0, 60.0, 80.0],
            [80.0, 60.0, 60.0, 80.0]
        ])
        res = get_overall_subject_analytics(subjects, matrix)
        self.assertEqual(res["highest_subjects"], "Math, Python")
        self.assertEqual(res["lowest_subjects"], "DBMS, OS")
        self.assertTrue(res["is_highest_tied"])
        self.assertTrue(res["is_lowest_tied"])

    def test_all_students_subject_breakdown(self) -> None:
        """Tests subject breakdown for multiple students."""
        breakdowns = get_all_students_subject_breakdown(self.students, self.subjects, self.marks_array)
        self.assertEqual(len(breakdowns), 5)
        self.assertEqual(breakdowns[0]["student"], "Arun")
        self.assertEqual(breakdowns[0]["strongest"], "Computer Networks")
        self.assertEqual(breakdowns[0]["weakest"], "Mathematics")

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

    def test_load_csv_success(self) -> None:
        """Tests successful loading of sample students CSV dataset."""
        csv_path = os.path.join(os.path.dirname(__file__), "../data/students.csv")
        students, subjects, marks = load_dataset_from_csv(csv_path)
        self.assertEqual(len(students), 10)
        self.assertEqual(subjects, ["Math", "DBMS", "OS", "Python"])
        self.assertEqual(students[0], "Aarav")
        self.assertEqual(marks[0], [88.0, 92.0, 85.0, 90.0])

    def test_load_csv_file_not_found(self) -> None:
        """Tests error handling for non-existent CSV file path."""
        with self.assertRaises(FileNotFoundError):
            load_dataset_from_csv("non_existent_file_xyz.csv")

    def test_load_csv_missing_columns(self) -> None:
        """Tests error handling for CSV file missing required Name/Student_ID column."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tf:
            tf.write("Score1,Score2\n80,90\n")
            temp_path = tf.name

        try:
            with self.assertRaises(ValueError):
                load_dataset_from_csv(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_load_csv_invalid_marks(self) -> None:
        """Tests error handling for CSV containing invalid out-of-bounds mark."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tf:
            tf.write("Student_ID,Name,Math\n101,Aarav,150\n")
            temp_path = tf.name

        try:
            with self.assertRaises(ValueError):
                load_dataset_from_csv(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_load_csv_missing_values(self) -> None:
        """Tests error handling for CSV containing missing mark values."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tf:
            tf.write("Student_ID,Name,Math,Python\n101,Aarav,85,\n")
            temp_path = tf.name

        try:
            with self.assertRaises(ValueError):
                load_dataset_from_csv(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


if __name__ == "__main__":
    unittest.main()



