"""
data.py

Provides default sample datasets and functions for user input validation
and custom dataset collection for the Student Performance Analyzer.
"""

from typing import Tuple, List

# Sample Dataset as specified in requirements
SAMPLE_STUDENTS: List[str] = ["Arun", "Divya", "Karthik", "Meena", "Rahul"]

SAMPLE_SUBJECTS: List[str] = [
    "Python",
    "Mathematics",
    "English",
    "Science",
    "Computer Networks"
]

SAMPLE_MARKS: List[List[float]] = [
    [85.0, 78.0, 92.0, 88.0, 95.0],  # Arun
    [72.0, 80.0, 75.0, 79.0, 84.0],  # Divya
    [90.0, 93.0, 89.0, 94.0, 91.0],  # Karthik
    [65.0, 70.0, 68.0, 72.0, 75.0],  # Meena
    [55.0, 60.0, 58.0, 62.0, 57.0]   # Rahul
]


def validate_mark(value_str: str) -> float:
    """
    Validates that the given input string represents a float between 0 and 100 inclusive.

    Raises:
        ValueError: If value cannot be converted to float or is out of range [0, 100].
    """
    try:
        val = float(value_str)
    except ValueError:
        raise ValueError("Invalid input! Please enter a numerical value.")

    if val < 0.0 or val > 100.0:
        raise ValueError(f"Mark ({val}) is out of bounds! Marks must be between 0 and 100.")

    return val


def get_sample_dataset() -> Tuple[List[str], List[str], List[List[float]]]:
    """
    Returns the pre-loaded sample students, subjects, and marks.
    """
    return SAMPLE_STUDENTS.copy(), SAMPLE_SUBJECTS.copy(), [row.copy() for row in SAMPLE_MARKS]


def get_custom_dataset() -> Tuple[List[str], List[str], List[List[float]]]:
    """
    Interactively collects custom student names, subject names, and validated marks from CLI.
    """
    print("\n--- Enter Custom Student Data ---")
    
    # Prompt for number of students
    while True:
        try:
            num_students = int(input("Enter number of students: ").strip())
            if num_students > 0:
                break
            print("Please enter a positive integer greater than 0.")
        except ValueError:
            print("Invalid input! Please enter a valid integer.")

    # Prompt for number of subjects
    while True:
        try:
            num_subjects = int(input("Enter number of subjects: ").strip())
            if num_subjects > 0:
                break
            print("Please enter a positive integer greater than 0.")
        except ValueError:
            print("Invalid input! Please enter a valid integer.")

    # Collect subject names
    subjects: List[str] = []
    print(f"\nEnter names for {num_subjects} subjects:")
    for i in range(num_subjects):
        name = input(f"  Subject {i + 1} name: ").strip()
        if not name:
            name = f"Subject_{i + 1}"
        subjects.append(name)

    # Collect student names and marks
    students: List[str] = []
    marks: List[List[float]] = []

    print("\nEnter student details and marks (0 - 100):")
    for s_idx in range(num_students):
        print(f"\n--- Student {s_idx + 1} ---")
        student_name = input("  Student name: ").strip()
        if not student_name:
            student_name = f"Student_{s_idx + 1}"
        students.append(student_name)

        student_marks: List[float] = []
        for subj in subjects:
            while True:
                user_val = input(f"    Mark for {subj} (0-100): ").strip()
                try:
                    valid_mark = validate_mark(user_val)
                    student_marks.append(valid_mark)
                    break
                except ValueError as err:
                    print(f"    Error: {err} Please try again.")

        marks.append(student_marks)

    return students, subjects, marks



def load_dataset_from_csv(file_path: str) -> Tuple[List[str], List[str], List[List[float]]]:
    """
    Reads student performance data from a CSV file.
    Expected CSV format: Student_ID, Name, Subject1, Subject2, ...
    
    Raises:
        FileNotFoundError: If the specified CSV file does not exist.
        ValueError: If file is empty, missing required columns, missing values, or invalid marks.
    """
    import os
    import csv

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found at path: '{file_path}'")

    with open(file_path, mode='r', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError("CSV file is empty or formatted incorrectly.")

        # Strip whitespace from fieldnames
        fieldnames = [field.strip() for field in reader.fieldnames if field]
        
        # Determine student identifier column (Name or Student_ID)
        name_col = None
        for candidate in ["Name", "Student_Name", "Student", "Student_ID", "ID"]:
            if candidate in fieldnames:
                name_col = candidate
                break

        if name_col is None:
            raise ValueError("Missing required column! CSV must contain a 'Name' or 'Student_ID' column.")

        # Identify subject columns (all non-identifier columns)
        non_subject_cols = {"Student_ID", "ID", "Name", "Student_Name", "Student"}
        subjects = [col for col in fieldnames if col not in non_subject_cols]

        if not subjects:
            raise ValueError("Missing required subject columns! CSV must contain at least one subject mark column.")

        students: List[str] = []
        marks: List[List[float]] = []

        row_num = 1
        for row in reader:
            row_num += 1
            # Clean keys in row
            clean_row = {k.strip(): v.strip() if v else "" for k, v in row.items() if k}
            
            student_name = clean_row.get(name_col, "").strip()
            if not student_name:
                raise ValueError(f"Missing student name/ID on row {row_num}.")

            students.append(student_name)
            student_marks: List[float] = []

            for subj in subjects:
                raw_mark = clean_row.get(subj, "").strip()
                if raw_mark == "":
                    raise ValueError(f"Missing value for student '{student_name}', subject '{subj}' on row {row_num}.")
                try:
                    val = validate_mark(raw_mark)
                    student_marks.append(val)
                except ValueError as err:
                    raise ValueError(f"Invalid data for student '{student_name}', subject '{subj}' on row {row_num}: {err}")

            marks.append(student_marks)

        if not students:
            raise ValueError("CSV file contains header but no student data rows.")

        return students, subjects, marks


def get_csv_dataset(default_path: str = "data/students.csv") -> Tuple[List[str], List[str], List[List[float]]]:
    """
    Prompts the user for a CSV file path and loads the dataset.
    Defaults to 'data/students.csv' if user presses Enter.
    """
    print("\n--- Load Dataset from CSV File ---")
    while True:
        user_path = input(f"Enter CSV file path [default: '{default_path}']: ").strip()
        if not user_path:
            user_path = default_path

        try:
            students, subjects, marks = load_dataset_from_csv(user_path)
            print(f"\nSuccessfully loaded {len(students)} students and {len(subjects)} subjects from '{user_path}'!")
            return students, subjects, marks
        except (FileNotFoundError, ValueError) as err:
            print(f"Error loading CSV file: {err}")
            retry = input("Would you like to try entering another path? (y/n): ").strip().lower()
            if retry != 'y':
                print("Falling back to default sample dataset.")
                return get_sample_dataset()

