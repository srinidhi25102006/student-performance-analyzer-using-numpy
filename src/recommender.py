"""
recommender.py

Personalized Student Improvement Recommendation Engine.
Generates targeted, actionable recommendations based on:
- Weakest subject(s)
- Performance trend (Improving, Stable, Declining)
- Risk level (High Risk, Moderate Risk, Low Risk)
- Overall average and subject scores

Reuses existing analyzer functions from analyzer.py to maintain modularity
and prevent duplication of analysis logic.
"""

from typing import List, Dict, Any, Optional
import numpy as np

from analyzer import (
    get_student_subject_breakdown,
    predict_single_student,
    detect_student_risk,
)


def normalize_risk_level(risk_level: str) -> str:
    """
    Normalizes risk level labels for display and rule checking.
    e.g. 'Moderate Risk' or 'MODERATE RISK' -> 'Moderate'
         'High Risk' or 'HIGH RISK' -> 'High'
         'Low Risk' or 'LOW RISK' -> 'Low'
    """
    risk_upper = str(risk_level).upper()
    if 'HIGH' in risk_upper:
        return 'High'
    elif 'MODERATE' in risk_upper:
        return 'Moderate'
    elif 'LOW' in risk_upper:
        return 'Low'
    return str(risk_level).replace(' Risk', '').replace(' RISK', '').strip()


def generate_recommendations(
    weakest_subject: str,
    trend: str,
    risk_level: str,
    overall_avg: Optional[float] = None,
    weakest_mark: Optional[float] = None
) -> List[str]:
    """
    Generates 2 to 4 simple, personalized recommendations based on student performance factors.

    Rules:
    1. Weakest Subject:
       - Recommends focusing on weakest subject fundamentals.
       - Recommends extra practice questions if score is low, risk is elevated, or trend is declining.
    2. Trend:
       - Declining: Recommends reviewing recent topics and increasing practice frequency.
       - Stable: Recommends setting targeted improvement goals.
       - Improving: Recommends maintaining positive study momentum.
    3. Risk Level:
       - High: Recommends creating a focused study plan for low-performing subjects.
       - Moderate: Recommends extra weekly revision time.
       - Low: Recommends advanced exercises or peer assistance.
    4. Assessment Follow-Up:
       - For Declining/Moderate/High risk: Recommends monitoring performance in next assessment.
       - For Improving/Low risk: Recommends setting higher benchmark goals.

    Guarantees between 2 and 4 prioritized, deduplicated recommendations.
    """
    recs: List[str] = []
    normalized_risk = normalize_risk_level(risk_level)
    trend_clean = trend.strip().capitalize()

    # Rule 1: Weakest Subject fundamentals
    if weakest_subject and weakest_subject != 'None':
        recs.append(f"Focus on {weakest_subject} fundamentals.")
        # Add practice recommendation if mark is low, risk is elevated, or trend is declining
        if weakest_mark is not None and weakest_mark < 65.0:
            recs.append(f"Practice more {weakest_subject} questions.")
        elif normalized_risk in ['High', 'Moderate'] or trend_clean == 'Declining':
            recs.append(f"Practice more {weakest_subject} questions.")

    # Rule 2: Performance Trend
    if trend_clean == 'Declining':
        recs.append("Review recent assessment topics and increase practice frequency.")
    elif trend_clean == 'Improving':
        recs.append("Maintain your positive learning momentum and continue consistent study habits.")
    elif trend_clean == 'Stable':
        recs.append("Set targeted goals to convert consistent performance into active improvement.")

    # Rule 3: Risk Level & Overall Average
    if normalized_risk == 'High' or (overall_avg is not None and overall_avg < 50.0):
        recs.append("Create a focused study plan for low-performing subjects.")
    elif normalized_risk == 'Moderate' or (overall_avg is not None and overall_avg < 70.0):
        if len(recs) < 3:
            recs.append("Dedicate extra weekly revision time to strengthen moderate-scoring subjects.")

    # Rule 4: Follow-up / Monitoring
    if trend_clean == 'Declining' or normalized_risk in ['High', 'Moderate']:
        recs.append("Monitor performance in the next assessment.")
    elif trend_clean == 'Improving' or normalized_risk == 'Low':
        if len(recs) < 3:
            recs.append("Set higher benchmark targets for the upcoming evaluation.")

    # Deduplicate while preserving order
    unique_recs: List[str] = []
    for item in recs:
        if item not in unique_recs:
            unique_recs.append(item)

    # Safeguards: enforce at least 2 recommendations
    if len(unique_recs) < 2:
        unique_recs.append("Review recent assessment topics and increase practice frequency.")
    if len(unique_recs) < 2:
        unique_recs.append("Monitor performance in the next assessment.")

    # Enforce maximum 4 recommendations
    return unique_recs[:4]


def generate_student_recommendations(
    student_name: str,
    subjects: List[str],
    student_marks: np.ndarray,
    prediction_info: Optional[Dict[str, Any]] = None,
    breakdown_info: Optional[Dict[str, Any]] = None,
    risk_info: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generates personalized recommendations for a single student.
    Reuses existing analyzer functions (get_student_subject_breakdown,
    predict_single_student, detect_student_risk) to avoid duplicating logic.
    """
    # Reuse existing breakdown if not provided
    if breakdown_info is None:
        breakdown_info = get_student_subject_breakdown(subjects, student_marks)

    # Reuse existing prediction if not provided
    if prediction_info is None:
        prediction_info = predict_single_student(student_marks)

    # Reuse existing risk analysis if not provided
    if risk_info is None:
        risk_info = detect_student_risk(student_marks, subjects)

    weakest_subject = breakdown_info.get('weakest', 'None')
    min_mark = breakdown_info.get('min_mark', None)
    trend = prediction_info.get('trend', 'Stable')

    # Use prediction risk_level if available, else risk_info risk_status
    raw_risk = prediction_info.get('risk_level') or risk_info.get('risk_status', 'Low Risk')
    normalized_risk = normalize_risk_level(raw_risk)

    overall_avg = float(prediction_info.get('current_avg', np.mean(student_marks)))

    recommendations = generate_recommendations(
        weakest_subject=weakest_subject,
        trend=trend,
        risk_level=normalized_risk,
        overall_avg=overall_avg,
        weakest_mark=min_mark
    )

    return {
        'student': student_name,
        'average': overall_avg,
        'weakest_subject': weakest_subject,
        'trend': trend,
        'risk_level': normalized_risk,
        'recommendations': recommendations
    }


def generate_all_recommendations(
    students: List[str],
    subjects: List[str],
    marks_array: np.ndarray
) -> List[Dict[str, Any]]:
    """
    Computes personalized recommendations for all students in the class.
    Handles empty arrays gracefully.
    """
    if len(students) == 0 or marks_array.size == 0:
        return []

    all_recommendations: List[Dict[str, Any]] = []
    for idx, student in enumerate(students):
        student_marks = marks_array[idx]
        rec = generate_student_recommendations(student, subjects, student_marks)
        all_recommendations.append(rec)

    return all_recommendations


def format_recommendations_report(student_rec: Dict[str, Any]) -> str:
    """
    Formats a student recommendation dictionary into the standardized text format.

    Example:
    Student: Rahul

    Overall Average: 68.5
    Weak Subject: DBMS
    Performance Trend: Declining
    Risk Level: Moderate

    Recommendations:
    1. Focus on DBMS fundamentals.
    2. Practice more DBMS questions.
    3. Review topics from recent assessments.
    4. Monitor performance in the next assessment.
    """
    avg = student_rec['average']
    if avg == int(avg):
        avg_str = f"{avg:.1f}"
    else:
        avg_str = f"{avg:.2f}".rstrip('0') if f"{avg:.2f}".endswith('0') else f"{avg:.2f}"

    lines = [
        f"Student: {student_rec['student']}",
        "",
        f"Overall Average: {avg_str}",
        f"Weak Subject: {student_rec['weakest_subject']}",
        f"Performance Trend: {student_rec['trend']}",
        f"Risk Level: {student_rec['risk_level']}",
        "",
        "Recommendations:"
    ]

    for idx, rec in enumerate(student_rec['recommendations'], 1):
        lines.append(f"{idx}. {rec}")

    return "\n".join(lines)
