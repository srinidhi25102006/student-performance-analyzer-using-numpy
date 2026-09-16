# Student Performance Analyzer using NumPy

A beginner-friendly, high-performance command-line application built with Python 3 and NumPy to analyze student academic performance. The application calculates individual totals, student averages, letter grades, pass/fail status, subject analytics, class statistics, student rankings, score deviations, and supports custom dataset entry via CLI.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [NumPy Concepts Demonstrated](#numpy-concepts-demonstrated)
- [Project Structure](#project-structure)
- [Installation Instructions](#installation-instructions)
- [How to Run the Project](#how-to-run-the-project)
- [How to Run Tests](#how-to-run-tests)
- [Sample Output](#sample-output)
- [Future Improvements](#future-improvements)
- [Author](#author)
- [License](#license)

---

## Project Overview

The **Student Performance Analyzer** processes student marks matrices using pure **NumPy** operations without relying on heavier data analysis libraries like Pandas. It provides clean tabular reporting, subject-level insights, class performance indicators, and index-based student rankings while demonstrating fundamental NumPy concepts in an accessible, educational codebase.

---

## Features

1. **Student Marks Matrix**: View all student marks across all subjects in a formatted ASCII table.
2. **Total & Average Calculations**: Compute total and average marks for every student using row-wise aggregations (`axis=1`).
3. **Automated Letter Grading**: Assign grades (`A+`, `A`, `B`, `C`, `D`, `F`) using vectorized grading rules based on average cutoffs.
4. **Pass/Fail Evaluation**: Determine pass/fail status based on strict multi-subject rules (requires >= 40 mark in all subjects).
5. **Subject-Wise Analytics**: Calculate class averages, highest marks, and lowest marks per subject (`axis=0`).
6. **Class Highlights**: Identify the top-performing student, lowest-performing student, class standard deviation, and best/worst performing subjects.
7. **High Performers Filter**: Extract students achieving an average mark >= 75.0 using boolean masking.
8. **Student Rankings**: Rank all students from highest to lowest average score using index sorting (`np.argsort`).
9. **Score Deviation Matrix**: Calculate score deviations relative to subject averages using 2D/1D **NumPy Broadcasting**.
10. **Performance Prediction**: Predict next assessment scores, detect score trajectory (`Improving`, `Declining`, `Stable`), and classify academic risk level (`Low Risk`, `Moderate Risk`, `High Risk`) using **NumPy Linear Regression**.
11. **Student Ranking System**: Compute overall averages, rank students from highest to lowest using **standard competition tie-ranking** (e.g. 1, 1, 3), and highlight top and lowest performing students.
12. **Subject-wise Performance Analysis**: Analyze each student's marks per subject, identify strongest/weakest subjects, calculate overall subject averages, and identify highest/lowest performing subjects with **graceful tie-handling**.
13. **CSV Dataset Support**: Load student performance records dynamically from `.csv` files with automated column detection and validation for missing values, invalid ranges, or header errors.
14. **Interactive CLI & Validation**: Switch between pre-loaded 5x5 sample dataset, custom interactive user entry, and external CSV files with validation ($0 - 100$).

---

## CSV Dataset Specifications

The application includes a sample dataset at [`data/students.csv`](file:///d:/student%20performance%20analyzer/data/students.csv).

### Required CSV Format

```csv
Student_ID,Name,Math,DBMS,OS,Python
101,Aarav,88,92,85,90
102,Bhavna,76,81,79,84
103,Chetan,95,89,94,92
104,Deepa,62,68,65,70
105,Esha,45,52,50,48
106,Farhan,82,78,85,80
107,Gita,91,95,88,93
108,Hari,55,60,58,62
109,Isha,70,75,72,74
110,Jatin,89,84,90,87
```

- **Identifier Columns**: Must contain `Name` or `Student_ID` (or both).
- **Subject Columns**: All additional numeric columns are automatically detected as subject marks.

---

## Technologies Used

- **Python 3**: Core application logic, `csv` standard library parser, and `unittest`.
- **NumPy**: Matrix creation, axis aggregations, boolean masking, index sorting (`np.argsort`), broadcasting, polynomial fitting, and tie resolution.

---

## NumPy Concepts Demonstrated

This project is tailored for learners mastering NumPy fundamentals:

| NumPy Concept | Usage in Project | Function / Operation |
| :--- | :--- | :--- |
| **Array Creation** | Converting 2D Python list to `ndarray` | `np.array(data, dtype=np.float64)` |
| **Metadata Inspection** | Reading array dimensions, size, & type | `arr.shape`, `arr.size`, `arr.ndim`, `arr.dtype` |
| **Row-wise Aggregation (`axis=1`)** | Calculating total and average marks per student | `np.sum(arr, axis=1)`, `np.mean(arr, axis=1)` |
| **Column-wise Aggregation (`axis=0`)** | Calculating class subject averages, max & min | `np.mean(arr, axis=0)`, `np.max(arr, axis=0)`, `np.min(arr, axis=0)` |
| **Argmax / Argmin** | Locating top student and best/worst subjects | `np.argmax(averages)`, `np.argmin(subject_averages)` |
| **Standard Deviation** | Measuring class score variance | `np.std(marks_array)` |
| **Boolean Masking & Ties** | Filtering high performers & resolving tied strongest/weakest subjects | `averages >= 75.0`, `np.where(row_marks == max_mark)` |
| **Conditional Selection** | Assigning grades and Pass/Fail status | `np.select(conditions, choices)`, `np.where(mask, 'Pass', 'Fail')` |
| **Index Sorting & Ranking** | Ranking students from highest to lowest score with competition tie resolution | `np.argsort(-averages)`, `compute_student_rankings()` |
| **Array Broadcasting** | Subtracting 1D subject averages array from 2D marks array | `marks_array - subject_averages` |
| **Linear Regression & Clipping** | Fitting trend slope to predict next assessment score | `np.polyfit(x, y, 1)`, `np.clip(val, 0, 100)` |

---

## Project Structure

```
student-performance-analyzer/
├── data/
│   └── students.csv      # Sample 10-student CSV performance dataset
├── src/
│   ├── main.py           # CLI menu, table formatting, and application workflow
│   ├── analyzer.py       # Modular NumPy computational functions with comments
│   └── data.py           # Pre-loaded sample dataset, CSV parser, & validation
├── tests/
│   └── test_analyzer.py  # Unit test suite built with Python's unittest module
├── README.md             # Project documentation and guide
├── requirements.txt      # Project dependencies (numpy)
├── .gitignore            # Git exclusion rules for Python
└── LICENSE               # MIT License
```

---

## Installation Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/student-performance-analyzer.git
   cd student-performance-analyzer
   ```

2. **Create a Virtual Environment (Optional but recommended)**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## How to Run the Project

Run the application main menu from the project root directory:

```bash
python src/main.py
```

### Interactive Menu Options

```
=== MAIN MENU ===
1. View Marks Table & Array Metadata
2. View Individual Student Performance (Totals, Averages, Grades, Status)
3. View Subject-wise Analytics & Best/Worst Subjects
4. View Detailed Per-Student Subject Analysis (Strongest/Weakest & Ties)
5. View Student Ranking Leaderboard (Competition Ranking & Ties)
6. View Class Insights, High Performers & Rankings
7. View Subject Deviations Matrix (NumPy Broadcasting Demo)
8. View Student Performance Predictions (NumPy Linear Regression)
9. Run Complete Full Analytical Report
10. Load & Analyze Student Data from CSV File
11. Switch to Custom Interactive Student Data Entry
12. Reset to Default Sample Dataset
13. Exit Application
```

### Running Student Ranking Leaderboard via CLI

Select Option `5` in the main menu to display the dedicated **Student Ranking Leaderboard**:

```text
================================================================================
                    STUDENT RANKING LEADERBOARD                 
================================================================================
Rank   | Student          |  Overall Average
------------------------------------------------
1      | Chetan           |            92.50
2      | Gita             |            91.75
3      | Aarav            |            88.75
4      | Jatin            |            87.50
------------------------------------------------
 Top Performing Student(s) : Chetan - 92.50
 Lowest Performing Student(s): Esha - 48.75
------------------------------------------------
```

---

## How to Run Tests

Execute the unit test suite using Python's built-in `unittest` discover tool:

```bash
python -m unittest discover -s tests
```

Expected Test Output:
```
.................................
----------------------------------------------------------------------
Ran 33 tests in 0.159s

OK
```

---

## Sample Output

```
================================================================================
                  STUDENT MARKS MATRIX                  
================================================================================
Student      |       Python |  Mathematics |      English |      Science | Computer Networks
--------------------------------------------------------------------------------------------
Arun         |        85.00 |        78.00 |        92.00 |        88.00 |        95.00
Divya        |        72.00 |        80.00 |        75.00 |        79.00 |        84.00
Karthik      |        90.00 |        93.00 |        89.00 |        94.00 |        91.00
Meena        |        65.00 |        70.00 |        68.00 |        72.00 |        75.00
Rahul        |        55.00 |        60.00 |        58.00 |        62.00 |        57.00
--------------------------------------------------------------------------------

--- NumPy Array Metadata ---
  Shape (Rows x Cols) : (5, 5)
  Total Size (Count)  : 25
  Dimensions (ndim)   : 2
  Data Type (dtype)   : float64

================================================================================
              INDIVIDUAL STUDENT PERFORMANCE REPORT             
================================================================================
Rank   | Student      |    Total |  Average |  Grade | Status
-------------------------------------------------------------
2      | Arun         |   438.00 |    87.60 |      A |   Pass
3      | Divya        |   390.00 |    78.00 |      B |   Pass
1      | Karthik      |   457.00 |    91.40 |     A+ |   Pass
4      | Meena        |   350.00 |    70.00 |      B |   Pass
5      | Rahul        |   292.00 |    58.40 |      D |   Pass
--------------------------------------------------------------------------------
```

---

## Future Improvements

- Add support for loading student datasets from CSV/JSON files without pandas.
- Export generated analytics reports to formatted plain text or Markdown files.
- Add weighted grade point average (GPA) calculations per subject credit hours.
- Implement percentile-based student grouping using `np.percentile()`.

---

## Author

**[Your Name Here]**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn Profile](https://linkedin.com/in/yourprofile)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
