"""
main.py

Main entry point for the Student Performance Analyzer CLI application.
Integrates data loading, NumPy analysis, report formatting, and interactive CLI menu.
"""

import sys
from typing import List
import numpy as np

from data import get_sample_dataset, get_custom_dataset
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
    predict_all_students,
)


def print_divider(character: str = "=", length: int = 80) -> None:
    """Prints a decorative section divider."""
    print(character * length)


def print_table_header(students: List[str], subjects: List[str]) -> None:
    """Formats and prints the main marks table header."""
    header_str = f"{'Student':<12} | " + " | ".join(f"{subj:>12}" for subj in subjects)
    print(header_str)
    print("-" * len(header_str))


def display_marks_table(students: List[str], subjects: List[str], marks_array: np.ndarray) -> None:
    """Displays student marks in a readable tabular format."""
    print_divider("=")
    print("                  STUDENT MARKS MATRIX                  ")
    print_divider("=")
    print_table_header(students, subjects)
    for idx, student in enumerate(students):
        row_marks = marks_array[idx]
        marks_str = " | ".join(f"{val:>12.2f}" for val in row_marks)
        print(f"{student:<12} | {marks_str}")
    print_divider("-")


def display_array_metadata(marks_array: np.ndarray) -> None:
    """Displays NumPy array properties (shape, size, ndim, dtype)."""
    props = get_array_properties(marks_array)
    print("\n--- NumPy Array Metadata ---")
    print(f"  Shape (Rows x Cols) : {props['shape']}")
    print(f"  Total Size (Count)  : {props['size']}")
    print(f"  Dimensions (ndim)   : {props['ndim']}")
    print(f"  Data Type (dtype)   : {props['dtype']}")


def display_student_report(
    students: List[str],
    subjects: List[str],
    marks_array: np.ndarray
) -> None:
    """Calculates and displays student totals, averages, grades, and pass/fail status."""
    totals = calculate_student_totals(marks_array)
    averages = calculate_student_averages(marks_array)
    grades = assign_grades(averages)
    pass_fail = determine_pass_fail(marks_array)

    print_divider("=")
    print("              INDIVIDUAL STUDENT PERFORMANCE REPORT             ")
    print_divider("=")
    header = f"{'Rank':<6} | {'Student':<12} | {'Total':>8} | {'Average':>8} | {'Grade':>6} | {'Status':>6}"
    print(header)
    print("-" * len(header))

    rankings = rank_students(students, averages)
    # Create dictionary for quick lookup by student name
    rank_dict = {student: rank for rank, student, _ in rankings}

    for idx, student in enumerate(students):
        s_rank = rank_dict[student]
        print(
            f"{s_rank:<6} | {student:<12} | {totals[idx]:>8.2f} | "
            f"{averages[idx]:>8.2f} | {grades[idx]:>6} | {pass_fail[idx]:>6}"
        )
    print_divider("-")


def display_subject_report(
    subjects: List[str],
    marks_array: np.ndarray
) -> None:
    """Displays subject-wise metrics: average, highest mark, and lowest mark."""
    subj_averages = calculate_subject_averages(marks_array)
    subj_max = get_subject_highest(marks_array)
    subj_min = get_subject_lowest(marks_array)
    best_subj, best_avg = get_best_subject(subjects, subj_averages)
    worst_subj, worst_avg = get_worst_subject(subjects, subj_averages)

    print_divider("=")
    print("                SUBJECT-WISE PERFORMANCE ANALYTICS               ")
    print_divider("=")
    header = f"{'Subject':<20} | {'Class Average':>14} | {'Highest Mark':>12} | {'Lowest Mark':>12}"
    print(header)
    print("-" * len(header))

    for idx, subj in enumerate(subjects):
        print(
            f"{subj:<20} | {subj_averages[idx]:>14.2f} | "
            f"{subj_max[idx]:>12.2f} | {subj_min[idx]:>12.2f}"
        )
    print_divider("-")
    print(f" Best Performing Subject  : {best_subj} (Class Avg: {best_avg:.2f})")
    print(f" Worst Performing Subject : {worst_subj} (Class Avg: {worst_avg:.2f})")
    print_divider("-")


def display_class_insights(
    students: List[str],
    subjects: List[str],
    marks_array: np.ndarray
) -> None:
    """Displays class-level statistics, rankings, high performers, pass/fail summary, and std dev."""
    averages = calculate_student_averages(marks_array)
    pass_fail = determine_pass_fail(marks_array)
    num_passed, num_failed = count_pass_fail(pass_fail)
    top_student, top_avg = get_top_student(students, averages)
    low_student, low_avg = get_lowest_student(students, averages)
    high_performers = get_high_performers(students, averages, threshold=75.0)
    class_std = calculate_class_std(marks_array)

    print_divider("=")
    print("                 CLASS STATISTICS & HIGHLIGHTS                   ")
    print_divider("=")
    print(f" Student with Highest Average : {top_student} ({top_avg:.2f})")
    print(f" Student with Lowest Average  : {low_student} ({low_avg:.2f})")
    print(f" Class Standard Deviation     : {class_std:.2f}")
    print(f" Pass / Fail Summary          : {num_passed} Passed, {num_failed} Failed")
    print_divider("-")

    print("\nHigh Performers (Average >= 75.0):")
    if high_performers:
        for name, avg in high_performers:
            print(f"  - {name:<12} : {avg:.2f}")
    else:
        print("  None")

    print("\nStudent Rankings (Highest to Lowest Average):")
    rankings = rank_students(students, averages)
    for rank, name, avg in rankings:
        print(f"  Rank {rank:<2} : {name:<12} (Avg: {avg:.2f})")
    print_divider("-")


def display_deviations_report(
    students: List[str],
    subjects: List[str],
    marks_array: np.ndarray
) -> None:
    """Displays subject mark deviations demonstrating Array Broadcasting."""
    subj_averages = calculate_subject_averages(marks_array)
    deviations = calculate_subject_deviations(marks_array, subj_averages)

    print_divider("=")
    print("      SCORE DEVIATIONS FROM SUBJECT AVERAGES (Broadcasting Demo) ")
    print_divider("=")
    print("Positive (+) indicates scoring above subject average; Negative (-) below.")
    print_divider("-")
    print_table_header(students, subjects)

    for idx, student in enumerate(students):
        dev_row = deviations[idx]
        dev_str = " | ".join(f"{val:>+12.2f}" for val in dev_row)
        print(f"{student:<12} | {dev_str}")
    print_divider("-")


def display_prediction_report(students: List[str], marks_array: np.ndarray) -> None:
    """Displays student performance trend, predicted score, and risk status using linear regression."""
    predictions = predict_all_students(students, marks_array)

    print_divider("=")
    print("      STUDENT PERFORMANCE PREDICTIONS (NumPy Linear Regression) ")
    print_divider("=")
    header = f"{'Student':<12} | {'Current Avg':>11} | {'Trend':<10} | {'Predicted Score':>15} | {'Risk Level':<13}"
    print(header)
    print("-" * len(header))

    for p in predictions:
        print(
            f"{p['student']:<12} | {p['current_avg']:>11.2f} | "
            f"{p['trend']:<10} | {p['predicted_score']:>15.2f} | {p['risk_level']:<13}"
        )
    print_divider("-")


def run_full_report(students: List[str], subjects: List[str], marks_array: np.ndarray) -> None:
    """Runs all analytical reports in sequence."""
    display_marks_table(students, subjects, marks_array)
    display_array_metadata(marks_array)
    display_student_report(students, subjects, marks_array)
    display_subject_report(subjects, marks_array)
    display_class_insights(students, subjects, marks_array)
    display_deviations_report(students, subjects, marks_array)
    display_prediction_report(students, marks_array)


def main() -> None:
    """CLI Menu Loop for Student Performance Analyzer."""
    print_divider("=")
    print("      WELCOME TO STUDENT PERFORMANCE ANALYZER (NumPy Edition)     ")
    print_divider("=")

    # Load initial default sample dataset
    students, subjects, marks_list = get_sample_dataset()
    marks_array = convert_to_array(marks_list)

    while True:
        print("\n=== MAIN MENU ===")
        print("1. View Marks Table & Array Metadata")
        print("2. View Individual Student Performance (Totals, Averages, Grades, Status)")
        print("3. View Subject-wise Analytics & Best/Worst Subjects")
        print("4. View Class Insights, High Performers & Rankings")
        print("5. View Subject Deviations Matrix (NumPy Broadcasting Demo)")
        print("6. View Student Performance Predictions (NumPy Linear Regression)")
        print("7. Run Complete Full Analytical Report")
        print("8. Switch to Custom Student Data Entry")
        print("9. Reset to Default Sample Dataset")
        print("10. Exit Application")

        choice = input("\nEnter your choice (1-10): ").strip()

        if choice == "1":
            display_marks_table(students, subjects, marks_array)
            display_array_metadata(marks_array)
        elif choice == "2":
            display_student_report(students, subjects, marks_array)
        elif choice == "3":
            display_subject_report(subjects, marks_array)
        elif choice == "4":
            display_class_insights(students, subjects, marks_array)
        elif choice == "5":
            display_deviations_report(students, subjects, marks_array)
        elif choice == "6":
            display_prediction_report(students, marks_array)
        elif choice == "7":
            run_full_report(students, subjects, marks_array)
        elif choice == "8":
            students, subjects, marks_list = get_custom_dataset()
            marks_array = convert_to_array(marks_list)
            print("\nCustom data loaded successfully!")
            run_full_report(students, subjects, marks_array)
        elif choice == "9":
            students, subjects, marks_list = get_sample_dataset()
            marks_array = convert_to_array(marks_list)
            print("\nReset back to default sample dataset.")
        elif choice == "10":
            print("\nThank you for using Student Performance Analyzer! Goodbye.")
            sys.exit(0)
        else:
            print("Invalid choice! Please select an option from 1 to 10.")


if __name__ == "__main__":
    main()

