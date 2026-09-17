"""
test_visualization.py

Unit test suite for visualization data preparation and Matplotlib chart helpers.
Verifies data transformation, student selection, risk aggregations, and empty dataset safeguards.
"""

import sys
import os
import unittest
import numpy as np

# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from analyzer import convert_to_array
from data import get_sample_dataset
from visualization import (
    prepare_subject_avg_data,
    prepare_student_avg_data,
    prepare_performance_trend_data,
    prepare_at_risk_summary_data,
    plot_subject_averages,
    plot_student_comparison,
    plot_student_trend,
    plot_at_risk_summary,
)


class TestVisualization(unittest.TestCase):
    """Unit tests for visualization data prep functions and safeguards."""

    def setUp(self):
        """Set up standard sample dataset for testing."""
        self.students, self.subjects, self.marks_list = get_sample_dataset()
        self.marks_array = convert_to_array(self.marks_list)

    def test_prepare_subject_avg_data(self):
        """Test subject average data extraction."""
        subj_names, averages = prepare_subject_avg_data(self.subjects, self.marks_array)
        self.assertEqual(subj_names, self.subjects)
        self.assertEqual(len(averages), len(self.subjects))
        # Expected averages for Arun, Divya, Karthik, Meena, Rahul
        # Python avg: (85+72+90+65+55)/5 = 73.4
        self.assertAlmostEqual(averages[0], 73.4, places=2)

    def test_prepare_student_avg_data(self):
        """Test student average data extraction."""
        st_names, averages = prepare_student_avg_data(self.students, self.marks_array)
        self.assertEqual(st_names, self.students)
        self.assertEqual(len(averages), len(self.students))
        # Arun average: (85+78+92+88+95)/5 = 87.6
        self.assertAlmostEqual(averages[0], 87.6, places=2)

    def test_prepare_performance_trend_data_default(self):
        """Test performance trend data default student selection (first student)."""
        subj_names, scores, target_name = prepare_performance_trend_data(
            self.students, self.subjects, self.marks_array
        )
        self.assertEqual(target_name, "Arun")
        self.assertEqual(subj_names, self.subjects)
        np.testing.assert_array_equal(scores, np.array([85.0, 78.0, 92.0, 88.0, 95.0]))

    def test_prepare_performance_trend_data_specific_student(self):
        """Test performance trend data for a specified student."""
        subj_names, scores, target_name = prepare_performance_trend_data(
            self.students, self.subjects, self.marks_array, student_name="Meena"
        )
        self.assertEqual(target_name, "Meena")
        np.testing.assert_array_equal(scores, np.array([65.0, 70.0, 68.0, 72.0, 75.0]))

    def test_prepare_at_risk_summary_data(self):
        """Test at-risk summary count data generation."""
        categories, counts = prepare_at_risk_summary_data(
            self.students, self.subjects, self.marks_array
        )
        self.assertEqual(len(categories), 3)
        # All sample dataset students have average >= 50, so Low Risk = 5
        self.assertEqual(counts, [0, 0, 5])

    def test_prepare_at_risk_summary_data_mixed(self):
        """Test at-risk summary count data with mixed risk students."""
        students = ["HighRiskSt", "ModRiskSt", "LowRiskSt"]
        subjects = ["Math", "Sci"]
        marks_array = convert_to_array([[30.0, 35.0], [45.0, 48.0], [80.0, 85.0]])
        categories, counts = prepare_at_risk_summary_data(students, subjects, marks_array)
        self.assertEqual(counts, [1, 1, 1])

    def test_empty_data_handling(self):
        """Test data preparation functions gracefully return empty structures when given empty dataset."""
        empty_students = []
        empty_subjects = []
        empty_array = np.array([], dtype=np.float64).reshape(0, 0)

        subj_names, subj_avgs = prepare_subject_avg_data(empty_subjects, empty_array)
        self.assertEqual(subj_names, [])
        self.assertEqual(subj_avgs.size, 0)

        st_names, st_avgs = prepare_student_avg_data(empty_students, empty_array)
        self.assertEqual(st_names, [])
        self.assertEqual(st_avgs.size, 0)

        subj_names, scores, target_name = prepare_performance_trend_data(
            empty_students, empty_subjects, empty_array
        )
        self.assertEqual(subj_names, [])
        self.assertEqual(scores.size, 0)
        self.assertEqual(target_name, "")

        categories, counts = prepare_at_risk_summary_data(
            empty_students, empty_subjects, empty_array
        )
        self.assertEqual(counts, [0, 0, 0])

    def test_plotting_non_blocking(self):
        """Verify plotting functions execute without error when show=False."""
        plot_subject_averages(self.subjects, self.marks_array, show=False)
        plot_student_comparison(self.students, self.marks_array, show=False)
        plot_student_trend(self.students, self.subjects, self.marks_array, show=False)
        plot_at_risk_summary(self.students, self.subjects, self.marks_array, show=False)
        from visualization import plot_all_visualizations
        plot_all_visualizations(self.students, self.subjects, self.marks_array, show=False)


if __name__ == '__main__':
    unittest.main()
