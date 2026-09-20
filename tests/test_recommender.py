"""
test_recommender.py

Unit test suite for the Personalized Student Improvement Recommendation Engine.
Tests recommendation logic for:
- Improving student
- Declining student
- Stable student
- Weak subject identification
- High-risk student
- Multiple recommendations count constraint (2-4 items)
- Tied weakest subjects
- Batch recommendation generation
- Report output formatting
"""

import sys
import os
import unittest
import numpy as np

# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from data import get_sample_dataset
from analyzer import convert_to_array
from recommender import (
    generate_recommendations,
    generate_student_recommendations,
    generate_all_recommendations,
    format_recommendations_report,
    normalize_risk_level,
)


class TestStudentRecommendations(unittest.TestCase):
    """Test suite for student personalized improvement recommendations."""

    def setUp(self) -> None:
        """Sets up default dataset prior to each test."""
        self.students, self.subjects, self.marks_list = get_sample_dataset()
        self.marks_array = convert_to_array(self.marks_list)

    def test_improving_student(self) -> None:
        """Tests that an improving student receives momentum-building recommendations."""
        # Consecutive improving marks across subjects
        subjects = ["Math", "DBMS", "OS", "Python"]
        improving_marks = np.array([55.0, 65.0, 78.0, 90.0])
        rec = generate_student_recommendations("Priya", subjects, improving_marks)

        self.assertEqual(rec["student"], "Priya")
        self.assertEqual(rec["trend"], "Improving")
        # Verify recommendation contains momentum guidance
        recs_text = " ".join(rec["recommendations"])
        self.assertIn("momentum", recs_text.lower())
        # Verify recommendation count is within bounds
        self.assertTrue(2 <= len(rec["recommendations"]) <= 4)

    def test_declining_student(self) -> None:
        """Tests that a declining student receives review and practice recommendations."""
        subjects = ["Math", "DBMS", "OS", "Python"]
        declining_marks = np.array([92.0, 80.0, 68.0, 52.0])
        rec = generate_student_recommendations("Rohan", subjects, declining_marks)

        self.assertEqual(rec["student"], "Rohan")
        self.assertEqual(rec["trend"], "Declining")
        recs_text = " ".join(rec["recommendations"])
        self.assertTrue(
            "recent assessment topics" in recs_text.lower() or
            "practice frequency" in recs_text.lower()
        )
        self.assertTrue(2 <= len(rec["recommendations"]) <= 4)

    def test_stable_student(self) -> None:
        """Tests that a student with consistent performance receives goal-setting advice."""
        subjects = ["Math", "DBMS", "OS", "Python"]
        stable_marks = np.array([75.0, 76.0, 74.0, 75.0])
        rec = generate_student_recommendations("Ananya", subjects, stable_marks)

        self.assertEqual(rec["student"], "Ananya")
        self.assertEqual(rec["trend"], "Stable")
        recs_text = " ".join(rec["recommendations"])
        self.assertTrue(
            "targeted goals" in recs_text.lower() or
            "consistent performance" in recs_text.lower()
        )
        self.assertTrue(2 <= len(rec["recommendations"]) <= 4)

    def test_weak_subject_recommendation(self) -> None:
        """Tests that recommendations specifically identify and target the weakest subject."""
        subjects = ["Math", "DBMS", "OS", "Python"]
        # DBMS is distinctly the lowest mark (48.0)
        marks = np.array([82.0, 48.0, 75.0, 80.0])
        rec = generate_student_recommendations("Rahul", subjects, marks)

        self.assertEqual(rec["weakest_subject"], "DBMS")
        # Check that DBMS is explicitly mentioned in the recommendations
        dbms_recs = [r for r in rec["recommendations"] if "DBMS" in r]
        self.assertGreaterEqual(len(dbms_recs), 1)
        self.assertTrue(any("DBMS fundamentals" in r for r in dbms_recs))

    def test_high_risk_student(self) -> None:
        """Tests that a high-risk student receives focused study plan recommendations."""
        subjects = ["Math", "DBMS", "OS", "Python"]
        # Overall average well below 40.0
        high_risk_marks = np.array([32.0, 35.0, 28.0, 30.0])
        rec = generate_student_recommendations("Vikram", subjects, high_risk_marks)

        self.assertEqual(rec["risk_level"], "High")
        recs_text = " ".join(rec["recommendations"])
        self.assertIn("focused study plan", recs_text.lower())
        self.assertTrue(2 <= len(rec["recommendations"]) <= 4)

    def test_moderate_risk_student(self) -> None:
        """Tests that a moderate-risk student receives appropriate monitoring and revision advice."""
        recs = generate_recommendations(
            weakest_subject="Python",
            trend="Declining",
            risk_level="Moderate",
            overall_avg=65.0,
            weakest_mark=58.0
        )
        self.assertTrue(2 <= len(recs) <= 4)
        recs_text = " ".join(recs)
        self.assertIn("Python", recs_text)
        self.assertIn("next assessment", recs_text.lower())

    def test_multiple_recommendations_count(self) -> None:
        """Tests that the number of generated recommendations is strictly between 2 and 4 for all test cases."""
        test_scenarios = [
            ("Math", "Improving", "Low", 88.0, 80.0),
            ("DBMS", "Declining", "Moderate", 68.5, 55.0),
            ("OS", "Stable", "High", 38.0, 30.0),
            ("Python", "Improving", "High", 45.0, 35.0),
            ("Computer Networks", "Declining", "Low", 82.0, 75.0),
            ("Math", "Stable", "Low", 95.0, 90.0),
            ("None", "Stable", "Low", 70.0, 70.0),
        ]

        for subj, trend, risk, avg, min_m in test_scenarios:
            with self.subTest(scenario=(subj, trend, risk)):
                recs = generate_recommendations(subj, trend, risk, avg, min_m)
                self.assertGreaterEqual(len(recs), 2, f"Fewer than 2 recs for {subj}, {trend}, {risk}")
                self.assertLessEqual(len(recs), 4, f"More than 4 recs for {subj}, {trend}, {risk}")

    def test_tied_weakest_subjects(self) -> None:
        """Tests recommendation generation when multiple subjects are tied for lowest score."""
        subjects = ["Math", "DBMS", "OS", "Python"]
        # Both DBMS and OS have minimum score 50.0
        tied_marks = np.array([80.0, 50.0, 50.0, 80.0])
        rec = generate_student_recommendations("Suresh", subjects, tied_marks)

        self.assertIn("DBMS", rec["weakest_subject"])
        self.assertIn("OS", rec["weakest_subject"])
        self.assertTrue(2 <= len(rec["recommendations"]) <= 4)

    def test_generate_all_recommendations(self) -> None:
        """Tests batch generation of recommendations for an entire class."""
        all_recs = generate_all_recommendations(self.students, self.subjects, self.marks_array)
        self.assertEqual(len(all_recs), len(self.students))

        for rec in all_recs:
            self.assertIn("student", rec)
            self.assertIn("average", rec)
            self.assertIn("weakest_subject", rec)
            self.assertIn("trend", rec)
            self.assertIn("risk_level", rec)
            self.assertIn("recommendations", rec)
            self.assertTrue(2 <= len(rec["recommendations"]) <= 4)

    def test_generate_all_recommendations_empty(self) -> None:
        """Tests that batch generation gracefully handles empty inputs."""
        empty_recs = generate_all_recommendations([], [], np.array([]))
        self.assertEqual(empty_recs, [])

    def test_normalize_risk_level(self) -> None:
        """Tests risk label normalization."""
        self.assertEqual(normalize_risk_level("High Risk"), "High")
        self.assertEqual(normalize_risk_level("HIGH RISK"), "High")
        self.assertEqual(normalize_risk_level("Moderate Risk"), "Moderate")
        self.assertEqual(normalize_risk_level("LOW RISK"), "Low")
        self.assertEqual(normalize_risk_level("Low"), "Low")

    def test_format_recommendations_report(self) -> None:
        """Tests formatting of student recommendation report string."""
        sample_rec = {
            "student": "Rahul",
            "average": 68.5,
            "weakest_subject": "DBMS",
            "trend": "Declining",
            "risk_level": "Moderate",
            "recommendations": [
                "Focus on DBMS fundamentals.",
                "Practice more DBMS questions.",
                "Review recent assessment topics and increase practice frequency.",
                "Monitor performance in the next assessment."
            ]
        }
        report_str = format_recommendations_report(sample_rec)

        self.assertIn("Student: Rahul", report_str)
        self.assertIn("Overall Average: 68.5", report_str)
        self.assertIn("Weak Subject: DBMS", report_str)
        self.assertIn("Performance Trend: Declining", report_str)
        self.assertIn("Risk Level: Moderate", report_str)
        self.assertIn("1. Focus on DBMS fundamentals.", report_str)
        self.assertIn("2. Practice more DBMS questions.", report_str)
        self.assertIn("3. Review recent assessment topics and increase practice frequency.", report_str)
        self.assertIn("4. Monitor performance in the next assessment.", report_str)


if __name__ == "__main__":
    unittest.main()
