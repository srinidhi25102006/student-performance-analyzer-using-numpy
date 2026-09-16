"""
analyzer.py

Contains reusable NumPy analysis functions for student performance evaluation.
Includes detailed comments demonstrating core NumPy concepts like multi-dimensional arrays,
vectorized operations, axis-wise aggregations, boolean masking, index sorting, and broadcasting.
"""

from typing import List, Tuple, Dict, Any
import numpy as np


def convert_to_array(data: List[List[float]]) -> np.ndarray:
    """
    CONCEPT: Converting lists into arrays & specifying dtype.
    Converts a 2D Python list of student marks into a 2D NumPy float ndarray.
    """
    # np.array() creates an N-dimensional array object from a Python sequence.
    return np.array(data, dtype=np.float64)


def get_array_properties(marks_array: np.ndarray) -> Dict[str, Any]:
    """
    CONCEPT: Array shape, size, ndim, and dtype.
    Inspects and returns fundamental metadata of the NumPy array.
    """
    return {
        # .shape returns a tuple representing array dimensions (e.g. (rows, cols))
        "shape": marks_array.shape,
        # .size returns total number of elements in the array
        "size": marks_array.size,
        # .ndim returns number of dimensions (axes) of the array (e.g. 2 for matrix)
        "ndim": marks_array.ndim,
        # .dtype returns the data type of the array elements (e.g. float64)
        "dtype": str(marks_array.dtype)
    }


def calculate_student_totals(marks_array: np.ndarray) -> np.ndarray:
    """
    CONCEPT: np.sum() across axis=1 (horizontal / row-wise).
    Calculates total marks for each student across all subjects.
    
    Axis explanation:
    - axis=1 sums across columns for each row (student totals).
    """
    return np.sum(marks_array, axis=1)


def calculate_student_averages(marks_array: np.ndarray) -> np.ndarray:
    """
    CONCEPT: np.mean() across axis=1 (horizontal / row-wise).
    Calculates average marks for each student.
    """
    return np.mean(marks_array, axis=1)


def assign_grades(averages: np.ndarray) -> np.ndarray:
    """
    CONCEPT: Boolean masking & np.select / np.where for grading logic.
    Assigns letter grades based on student average marks:
        - A+ : average >= 90
        - A  : average >= 80
        - B  : average >= 70
        - C  : average >= 60
        - D  : average >= 50
        - F  : average < 50
    """
    # Define boolean conditions based on threshold rules
    conditions = [
        averages >= 90.0,
        averages >= 80.0,
        averages >= 70.0,
        averages >= 60.0,
        averages >= 50.0
    ]
    choices = ['A+', 'A', 'B', 'C', 'D']
    
    # np.select evaluates conditions list in order and selects corresponding choice.
    # Default is 'F' for any average < 50.
    grades = np.select(conditions, choices, default='F')
    return grades


def determine_pass_fail(marks_array: np.ndarray, pass_mark: float = 40.0) -> np.ndarray:
    """
    CONCEPT: Boolean masking, np.all(), and np.where().
    A student passes ONLY if EVERY subject mark is at least pass_mark (40.0).
    
    - marks_array >= pass_mark creates a boolean array of same shape.
    - np.all(..., axis=1) checks if ALL elements in a row are True.
    - np.where(condition, x, y) returns 'Pass' where True, 'Fail' where False.
    """
    passed_all_subjects = np.all(marks_array >= pass_mark, axis=1)
    status_array = np.where(passed_all_subjects, 'Pass', 'Fail')
    return status_array


def calculate_subject_averages(marks_array: np.ndarray) -> np.ndarray:
    """
    CONCEPT: np.mean() across axis=0 (vertical / column-wise).
    Calculates the class average for each subject.
    
    Axis explanation:
    - axis=0 aggregates down rows for each column (subject averages).
    """
    return np.mean(marks_array, axis=0)


def get_subject_highest(marks_array: np.ndarray) -> np.ndarray:
    """
    CONCEPT: np.max() across axis=0.
    Finds the highest mark obtained in each subject across all students.
    """
    return np.max(marks_array, axis=0)


def get_subject_lowest(marks_array: np.ndarray) -> np.ndarray:
    """
    CONCEPT: np.min() across axis=0.
    Finds the lowest mark obtained in each subject across all students.
    """
    return np.min(marks_array, axis=0)


def get_top_student(students: List[str], averages: np.ndarray) -> Tuple[str, float]:
    """
    CONCEPT: np.argmax() and 1D indexing.
    Finds the student with the highest overall average.
    
    - np.argmax() returns the index of the maximum value in the array.
    """
    top_idx = int(np.argmax(averages))
    return students[top_idx], float(averages[top_idx])


def get_lowest_student(students: List[str], averages: np.ndarray) -> Tuple[str, float]:
    """
    CONCEPT: np.argmin() and 1D indexing.
    Finds the student with the lowest overall average.
    
    - np.argmin() returns the index of the minimum value in the array.
    """
    lowest_idx = int(np.argmin(averages))
    return students[lowest_idx], float(averages[lowest_idx])


def get_high_performers(
    students: List[str],
    averages: np.ndarray,
    threshold: float = 75.0
) -> List[Tuple[str, float]]:
    """
    CONCEPT: Boolean masking on 1D arrays.
    Displays students whose average is greater than or equal to threshold (default 75.0).
    """
    students_arr = np.array(students)
    
    # Create boolean mask
    mask = averages >= threshold
    
    # Apply boolean mask to slice array elements matching condition
    filtered_students = students_arr[mask]
    filtered_averages = averages[mask]
    
    # Return list of tuples (student_name, average)
    return list(zip(filtered_students.tolist(), filtered_averages.tolist()))


def calculate_class_std(marks_array: np.ndarray) -> float:
    """
    CONCEPT: np.std().
    Calculates standard deviation of all marks in the class to measure score dispersion.
    """
    return float(np.std(marks_array))


def rank_students(students: List[str], averages: np.ndarray) -> List[Tuple[int, str, float]]:
    """
    CONCEPT: np.argsort() and array indexing.
    Ranks students from highest average to lowest average.
    
    - np.argsort(-averages) returns indices that would sort the array in descending order.
    """
    # Negating averages gives descending order indices
    sorted_indices = np.argsort(-averages)
    
    rankings = []
    for rank, idx in enumerate(sorted_indices, start=1):
        rankings.append((rank, students[idx], float(averages[idx])))
        
    return rankings


def count_pass_fail(pass_fail_status: np.ndarray) -> Tuple[int, int]:
    """
    CONCEPT: Boolean comparison and np.sum().
    Counts how many students passed and how many failed.
    """
    # Compare array with string 'Pass' to get boolean mask, then np.sum() counts True values
    num_passed = int(np.sum(pass_fail_status == 'Pass'))
    num_failed = int(np.sum(pass_fail_status == 'Fail'))
    return num_passed, num_failed


def get_best_subject(subjects: List[str], subject_averages: np.ndarray) -> Tuple[str, float]:
    """
    CONCEPT: np.argmax() on subject averages.
    Displays the subject in which the class performed best.
    """
    best_idx = int(np.argmax(subject_averages))
    return subjects[best_idx], float(subject_averages[best_idx])


def get_worst_subject(subjects: List[str], subject_averages: np.ndarray) -> Tuple[str, float]:
    """
    CONCEPT: np.argmin() on subject averages.
    Displays the subject in which the class performed worst.
    """
    worst_idx = int(np.argmin(subject_averages))
    return subjects[worst_idx], float(subject_averages[worst_idx])


def calculate_subject_deviations(marks_array: np.ndarray, subject_averages: np.ndarray) -> np.ndarray:
    """
    CONCEPT: Array Broadcasting (2D array - 1D array).
    Calculates the deviation of each student's mark from the subject's class average.
    
    Broadcasting rule:
    - marks_array has shape (N_students, N_subjects) e.g. (5, 5).
    - subject_averages has shape (N_subjects,) e.g. (5,).
    - NumPy automatically broadcasts subject_averages across rows so each student row
      is subtracted element-wise by subject averages.
    """
    return marks_array - subject_averages


def predict_single_student(student_marks: np.ndarray) -> Dict[str, Any]:
    """
    CONCEPT: Simple Linear Regression using np.polyfit() & np.clip().
    Predicts a student's performance on the next assessment based on historical mark sequence.
    
    - np.polyfit(x, y, 1) fits a degree-1 polynomial (line: y = slope * x + intercept).
    - np.clip(val, min, max) ensures predicted marks stay within valid [0, 100] bounds.
    """
    num_assessments = len(student_marks)
    if num_assessments < 2:
        current_avg = float(np.mean(student_marks)) if num_assessments > 0 else 0.0
        risk = "Low Risk" if current_avg >= 75.0 else ("Moderate Risk" if current_avg >= 60.0 else "High Risk")
        return {
            "current_avg": current_avg,
            "slope": 0.0,
            "trend": "Stable",
            "predicted_score": current_avg,
            "risk_level": risk
        }

    # Generate sequence indices x = [0, 1, 2, ..., N-1]
    x = np.arange(num_assessments)
    
    # Fit line: y = slope * x + intercept
    # np.polyfit returns array [slope, intercept]
    coefficients = np.polyfit(x, student_marks, deg=1)
    slope = float(coefficients[0])
    intercept = float(coefficients[1])

    # Predict score for next assessment index (x_next = num_assessments)
    next_x = num_assessments
    predicted_val = slope * next_x + intercept
    # np.clip ensures output is within [0.0, 100.0]
    predicted_score = float(np.clip(predicted_val, 0.0, 100.0))

    current_avg = float(np.mean(student_marks))

    # Determine performance trend based on slope
    if slope > 0.5:
        trend = "Improving"
    elif slope < -0.5:
        trend = "Declining"
    else:
        trend = "Stable"

    # Determine risk level based on predicted score and trend
    if predicted_score < 60.0 or (trend == "Declining" and predicted_score < 70.0):
        risk_level = "High Risk"
    elif predicted_score < 75.0:
        risk_level = "Moderate Risk"
    else:
        risk_level = "Low Risk"

    return {
        "current_avg": current_avg,
        "slope": slope,
        "trend": trend,
        "predicted_score": predicted_score,
        "risk_level": risk_level
    }


def predict_all_students(students: List[str], marks_array: np.ndarray) -> List[Dict[str, Any]]:
    """
    Computes performance predictions for all students in the class.
    """
    predictions = []
    for idx, student in enumerate(students):
        student_marks = marks_array[idx]
        pred_dict = predict_single_student(student_marks)
        pred_dict["student"] = student
        predictions.append(pred_dict)
    return predictions

