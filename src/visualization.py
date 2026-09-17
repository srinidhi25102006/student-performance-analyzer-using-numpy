"""
visualization.py

Provides Matplotlib visual chart generation for student and subject performance analysis.
Keeps plotting and graphical routines separate from core numerical analyzer logic.
Demonstrates NumPy aggregations, data preparation for charts, and Matplotlib plotting.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt

from analyzer import (
    calculate_subject_averages,
    calculate_student_averages,
    detect_all_at_risk_students,
)


def prepare_subject_avg_data(
    subjects: List[str], marks_array: np.ndarray
) -> Tuple[List[str], np.ndarray]:
    """
    Extracts and prepares subject-wise average performance data for plotting.

    Returns:
        Tuple containing list of subject names and 1D array of subject average marks.
    """
    if len(subjects) == 0 or marks_array.size == 0:
        return [], np.array([], dtype=np.float64)

    subject_avgs = calculate_subject_averages(marks_array)
    return list(subjects), subject_avgs


def prepare_student_avg_data(
    students: List[str], marks_array: np.ndarray
) -> Tuple[List[str], np.ndarray]:
    """
    Extracts and prepares overall average marks for each student for plotting.

    Returns:
        Tuple containing list of student names and 1D array of overall average marks.
    """
    if len(students) == 0 or marks_array.size == 0:
        return [], np.array([], dtype=np.float64)

    student_avgs = calculate_student_averages(marks_array)
    return list(students), student_avgs


def prepare_performance_trend_data(
    students: List[str],
    subjects: List[str],
    marks_array: np.ndarray,
    student_name: Optional[str] = None,
) -> Tuple[List[str], np.ndarray, str]:
    """
    Extracts performance marks across subjects/assessments for a target student.
    If student_name is None, defaults to the first student.

    Returns:
        Tuple containing subject list, student's subject marks array, and selected student name.
    """
    if len(students) == 0 or marks_array.size == 0 or len(subjects) == 0:
        return [], np.array([], dtype=np.float64), ""

    selected_name = student_name if student_name in students else students[0]
    student_idx = students.index(selected_name)
    student_marks = marks_array[student_idx]
    return list(subjects), student_marks, selected_name


def prepare_at_risk_summary_data(
    students: List[str], subjects: List[str], marks_array: np.ndarray
) -> Tuple[List[str], List[int]]:
    """
    Counts the number of High Risk, Moderate Risk, and Low Risk students.

    Returns:
        Tuple containing risk category names and list of integer counts [High, Moderate, Low].
    """
    categories = ["High Risk (<40)", "Moderate Risk (40-49)", "Low Risk (>=50)"]
    if len(students) == 0 or marks_array.size == 0:
        return categories, [0, 0, 0]

    risk_data = detect_all_at_risk_students(students, subjects, marks_array)
    summary = risk_data["summary"]
    return categories, [summary["high_risk"], summary["moderate_risk"], summary["low_risk"]]


# --- PLOTTING ROUTINES ---


def plot_subject_averages(
    subjects: List[str],
    marks_array: np.ndarray,
    show: bool = True,
    save_path: Optional[str] = None,
) -> None:
    """Renders a Bar Chart of subject-wise average marks."""
    subj_names, averages = prepare_subject_avg_data(subjects, marks_array)

    if not subj_names or averages.size == 0:
        print("Warning: Cannot display chart. Dataset is empty.")
        return

    plt.figure(figsize=(8, 5))
    bars = plt.bar(subj_names, averages, color="#3498db", edgecolor="#2980b9")
    plt.title("Subject-Wise Average Performance", fontsize=14, fontweight="bold")
    plt.xlabel("Subjects", fontsize=12)
    plt.ylabel("Average Marks", fontsize=12)
    plt.ylim(0, 100)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1,
            f"{height:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()


def plot_student_comparison(
    students: List[str],
    marks_array: np.ndarray,
    show: bool = True,
    save_path: Optional[str] = None,
) -> None:
    """Renders a Bar Chart comparing student overall averages with student names on the x-axis."""
    names, averages = prepare_student_avg_data(students, marks_array)

    if not names or averages.size == 0:
        print("Warning: Cannot display chart. Dataset is empty.")
        return

    plt.figure(figsize=(9, 5))
    bars = plt.bar(names, averages, color="#2ecc71", edgecolor="#27ae60")
    plt.title("Student Performance Comparison", fontsize=14, fontweight="bold")
    plt.xlabel("Student Name", fontsize=12)
    plt.ylabel("Overall Average Marks", fontsize=12)
    plt.ylim(0, 100)
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1,
            f"{height:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()


def plot_student_trend(
    students: List[str],
    subjects: List[str],
    marks_array: np.ndarray,
    student_name: Optional[str] = None,
    show: bool = True,
    save_path: Optional[str] = None,
) -> None:
    """Renders a Line Chart showing a student's marks across subjects/test assessments."""
    subj_names, scores, target_name = prepare_performance_trend_data(
        students, subjects, marks_array, student_name
    )

    if not subj_names or scores.size == 0:
        print("Warning: Cannot display chart. Dataset is empty.")
        return

    plt.figure(figsize=(8, 5))
    plt.plot(
        subj_names,
        scores,
        marker="o",
        linewidth=2.5,
        color="#e74c3c",
        label=f"{target_name}'s Marks",
    )

    # Compute regression line using NumPy polyfit if > 1 data point
    if len(subj_names) > 1:
        x_indices = np.arange(len(subj_names))
        slope, intercept = np.polyfit(x_indices, scores, 1)
        trend_y = slope * x_indices + intercept
        plt.plot(
            subj_names,
            trend_y,
            linestyle="--",
            color="#8e44ad",
            label=f"Trend (Slope: {slope:+.2f})",
        )

    plt.title(f"Performance Trend across Subjects ({target_name})", fontsize=14, fontweight="bold")
    plt.xlabel("Subjects / Assessments", fontsize=12)
    plt.ylabel("Marks", fontsize=12)
    plt.ylim(0, 100)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc="upper right")

    for i, mark in enumerate(scores):
        plt.text(i, mark + 1.5, f"{mark:.1f}", ha="center", fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()


def plot_at_risk_summary(
    students: List[str],
    subjects: List[str],
    marks_array: np.ndarray,
    show: bool = True,
    save_path: Optional[str] = None,
) -> None:
    """Renders a Bar Chart of At-Risk student classifications."""
    categories, counts = prepare_at_risk_summary_data(students, subjects, marks_array)

    if sum(counts) == 0:
        print("Warning: Cannot display chart. Dataset is empty.")
        return

    colors = ["#e74c3c", "#f39c12", "#2ecc71"]  # Red for High, Orange for Mod, Green for Low

    plt.figure(figsize=(8, 5))
    bars = plt.bar(categories, counts, color=colors, edgecolor="#34495e")
    plt.title("At-Risk Student Status Breakdown", fontsize=14, fontweight="bold")
    plt.xlabel("Risk Classification", fontsize=12)
    plt.ylabel("Number of Students", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.05,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()


def plot_all_visualizations(
    students: List[str],
    subjects: List[str],
    marks_array: np.ndarray,
    show: bool = True,
    save_path: Optional[str] = None,
) -> None:
    """Renders a 2x2 subplot dashboard summarizing all 4 performance charts."""
    if len(students) == 0 or marks_array.size == 0:
        print("Warning: Cannot display dashboard. Dataset is empty.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Student Performance Analytics Dashboard", fontsize=16, fontweight="bold"
    )

    # Subplot 1: Subject Averages
    subj_names, subj_avgs = prepare_subject_avg_data(subjects, marks_array)
    axes[0, 0].bar(subj_names, subj_avgs, color="#3498db", edgecolor="#2980b9")
    axes[0, 0].set_title("Subject-Wise Averages")
    axes[0, 0].set_ylabel("Average Marks")
    axes[0, 0].set_ylim(0, 100)
    axes[0, 0].grid(axis="y", linestyle="--", alpha=0.5)

    # Subplot 2: Student Averages
    st_names, st_avgs = prepare_student_avg_data(students, marks_array)
    axes[0, 1].bar(st_names, st_avgs, color="#2ecc71", edgecolor="#27ae60")
    axes[0, 1].set_title("Student Averages Comparison")
    axes[0, 1].set_ylabel("Overall Average")
    axes[0, 1].set_ylim(0, 100)
    axes[0, 1].tick_params(axis="x", rotation=30)
    axes[0, 1].grid(axis="y", linestyle="--", alpha=0.5)

    # Subplot 3: Performance Trend (First Student)
    subj_names, st_marks, target_name = prepare_performance_trend_data(
        students, subjects, marks_array
    )
    axes[1, 0].plot(
        subj_names, st_marks, marker="o", color="#e74c3c", label=target_name
    )
    if len(subj_names) > 1:
        x = np.arange(len(subj_names))
        slope, intercept = np.polyfit(x, st_marks, 1)
        axes[1, 0].plot(
            subj_names,
            slope * x + intercept,
            "--",
            color="#8e44ad",
            label="Trend",
        )
    axes[1, 0].set_title(f"Performance Trend ({target_name})")
    axes[1, 0].set_ylabel("Marks")
    axes[1, 0].set_ylim(0, 100)
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)
    axes[1, 0].legend()

    # Subplot 4: At-Risk Summary
    cat_names, counts = prepare_at_risk_summary_data(students, subjects, marks_array)
    colors = ["#e74c3c", "#f39c12", "#2ecc71"]
    axes[1, 1].bar(cat_names, counts, color=colors, edgecolor="#34495e")
    axes[1, 1].set_title("At-Risk Student Summary")
    axes[1, 1].set_ylabel("Student Count")
    axes[1, 1].grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()
