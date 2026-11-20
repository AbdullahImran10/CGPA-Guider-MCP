"""Tool functions for the PUCIT GPA/CGPA agent.

The agent never does arithmetic itself — every number in its answers must
come from one of these functions.
"""

from langchain_core.tools import tool

# marks lower bound -> grade points (checked from highest down)
GRADE_BANDS = [
    (85, 4.0),
    (80, 3.7),
    (75, 3.3),
    (70, 3.0),
    (65, 2.7),
    (61, 2.3),
    (58, 2.0),
    (55, 1.7),
    (50, 1.0),
    (0, 0.0),
]

# semester -> list of (code, name, credit_hours)
# MD-001 / MD-002 (Math Deficiency) are non-credit pass/fail and deliberately excluded.
COURSES = {
    1: [
        ("MS-251", "Probability & Statistics", 3.0),
        ("GE-160", "Applications of ICT", 3.0),
        ("GE-169", "Applied Physics", 3.0),
        ("GE-167", "Discrete Structures", 3.0),
        ("HQ-001", "Quran Translation - I", 0.5),
        ("GE-190", "Functional English", 3.0),
    ],
    2: [
        ("CC-112", "Programming Fundamentals", 3.0),
        ("CC-112-L", "Programming Fundamentals Lab", 1.0),
        ("CC-110", "Digital Logic Design", 2.0),
        ("CC-110-L", "Digital Logic Design Lab", 1.0),
        ("MS-252", "Linear Algebra", 3.0),
        ("GE-191", "Expository Writing", 3.0),
        ("GE-163", "Islamic Studies", 2.0),
        ("HQ-002", "Quran Translation - II", 0.5),
    ],
    3: [
        ("CC-211", "Object Oriented Programming", 3.0),
        ("CC-215", "Database Systems", 3.0),
        ("CC-215-L", "Database Systems Lab", 1.0),
        ("CC-210", "Computer Organization & Assembly Language", 3.0),
        ("GE-162", "Calculus & Analytical Geometry", 3.0),
        ("GE-192", "Introduction to Management", 2.0),
        ("HQ-003", "Quran Translation - III", 0.5),
        ("CC-211-L", "Object Oriented Programming Lab", 1.0),
    ],
    4: [
        ("CC-213", "Data Structures", 3.0),
        ("CC-213-L", "Data Structures Lab", 1.0),
        ("CC-312", "Information Security", 3.0),
        ("CC-214", "Computer Networks", 3.0),
        ("CC-212", "Software Engineering", 3.0),
        ("DC-220", "Advanced Database Management Systems", 3.0),
        ("HQ-004", "Quran Translation - IV", 0.5),
    ],
    5: [
        ("CC-313", "Analysis of Algorithms", 3.0),
        ("CC-310", "Artificial Intelligence", 3.0),
        ("DC-320", "Theory of Automata and Formal Languages", 3.0),
        ("DC-321", "Human Computer Interaction", 3.0),
        ("DC-322", "Computer Architecture", 3.0),
        ("EC-330", "Web Technologies / Elective", 3.0),
        ("HQ-005", "Quran Translation - V", 0.5),
    ],
    6: [
        ("CC-311", "Operating Systems", 3.0),
        ("EC-333", "Mobile Application Development / Elective", 3.0),
        ("EC-324", "Software Construction & Development / Elective", 3.0),
        ("EC-335", "Machine Learning / Elective", 3.0),
        ("EC-334", "Game Design and Development / Elective", 3.0),
        ("MS-253", "Multivariable Calculus", 3.0),
        ("HQ-006", "Quran Translation - VI", 0.5),
    ],
    7: [
        ("CC-411", "Final Year Project - I", 3.0),
        ("DC-328", "Parallel & Distributed Computing", 3.0),
        ("EC-345", "Computer Vision / Elective", 3.0),
        ("EC-425", "Software Quality Engineering / Elective", 3.0),
        ("MS-254", "Technical and Business Writing", 3.0),
        ("GE-263", "Entrepreneurship", 2.0),
        ("GE-262", "Professional Practices", 1.0),
        ("HQ-007", "Quran Translation - VII", 0.5),
    ],
    8: [
        ("CC-412", "Final Year Project - II", 4.0),
        ("DC-421", "Compiler Construction", 3.0),
        ("UE-272", "Introduction to Marketing", 3.0),
        ("GE-168", "Ideology and Constitution of Pakistan", 2.0),
        ("GE-363", "Civics and Community Engagement", 2.0),
        ("HQ-008", "Quran Translation - VIII", 0.5),
    ],
}

@tool
def get_semester_courses(semester: int) -> list:
    """Return the courses and credit hours for a specific semester.

    Use this when the student asks about the courses in a semester
    or when course credit hours are needed.
    """

    if semester not in COURSES:
        return "Error: semester must be between 1 and 8"

    return [
        {
            "code": code,
            "name": name,
            "credit_hours": credit_hours
        }
        for code, name, credit_hours in COURSES[semester]
    ]


@tool
def marks_to_grade_points(marks: int) -> float:
    """Convert marks (0-100) on a single course to PUCIT grade points.

    Use this whenever the student gives you raw marks for a course (e.g.
    "I got 78 in AI") and you need the grade point before computing a GPA.
    Never work out the grade band yourself — always call this tool.
    """
    if not isinstance(marks, (int, float)) or marks < 0 or marks > 100:
        return "Error: marks must be between 0 and 100"
    for threshold, points in GRADE_BANDS:
        if marks >= threshold:
            return points
    return 0.0


@tool
def calculate_semester_gpa(grade_points: list[float], credit_hours: list[float]) -> float:
    """Calculate one semester's credit-hour-weighted GPA.

    Use this once you have a grade point and a credit-hour value for every
    course the student took in a semester. grade_points and credit_hours
    must be the same length and in matching order (grade_points[i] pairs
    with credit_hours[i]).
    """
    if len(grade_points) != len(credit_hours):
        return "Error: grade_points and credit_hours must be the same length"
    if not grade_points:
        return "Error: at least one course is required"
    if any(ch <= 0 for ch in credit_hours):
        return "Error: credit hours must be greater than 0"
    total_points = sum(gp * ch for gp, ch in zip(grade_points, credit_hours))
    total_hours = sum(credit_hours)
    return round(total_points / total_hours, 3)


@tool
def calculate_new_cgpa(
    current_cgpa: float,
    completed_credit_hours: float,
    semester_gpa: float,
    semester_credit_hours: float,
) -> float:
    """Project the new CGPA after folding one semester's GPA into a student's record.

    Use this when the student gives you a current CGPA, how many credit
    hours they've completed so far, and an actual or hypothetical GPA for
    one more semester (with that semester's credit hours).
    """
    if completed_credit_hours < 0 or semester_credit_hours <= 0:
        return "Error: credit hours must be positive"
    if not (0 <= current_cgpa <= 4.0) or not (0 <= semester_gpa <= 4.0):
        return "Error: CGPA and GPA must be between 0.0 and 4.0"
    total_points = current_cgpa * completed_credit_hours + semester_gpa * semester_credit_hours
    total_hours = completed_credit_hours + semester_credit_hours
    return round(total_points / total_hours, 3)


@tool
def required_gpa_for_target(
    target_cgpa: float,
    current_cgpa: float,
    completed_credit_hours: float,
    remaining_credit_hours: float,
) -> float:
    """Calculate the GPA needed over given remaining credit hours to hit a target CGPA.

    Use this when the student names a target CGPA (e.g. "I want to graduate
    with a 3.5") and you know their current CGPA, completed credit hours,
    and the credit hours remaining over whichever horizon you're checking
    (one semester, several semesters, or all the way to graduation). A
    result above 4.0 means the target is impossible over that horizon —
    report that plainly rather than as an achievable GPA.
    """
    if remaining_credit_hours <= 0:
        return "Error: remaining credit hours must be greater than 0"
    if not (0 <= target_cgpa <= 4.0) or not (0 <= current_cgpa <= 4.0):
        return "Error: CGPA values must be between 0.0 and 4.0"
    total_ch = completed_credit_hours + remaining_credit_hours
    required = (target_cgpa * total_ch - current_cgpa * completed_credit_hours) / remaining_credit_hours
    return round(required, 3)

#get semester courses
@tool
def get_credit_hours_after_semester(after_semester: int) -> float:
    """Total the graded credit hours in all semesters AFTER after_semester (1-8).

    Use it to find how many credit hours are left after a given semester.
    For a shorter horizon, call it for two semester numbers and subtract:
    hours in semester 5 alone =
    get_credit_hours_after_semester(4) - get_credit_hours_after_semester(5).
    """
    if after_semester not in COURSES:
        return "Error: semester must be between 1 and 8"
    remaining = sum(
        ch for sem, courses in COURSES.items() if sem > after_semester for _, _, ch in courses
    )
    return round(remaining, 3)


@tool
def get_remaining_credit_hours(current_semester: int) -> float:
    """Total the graded credit hours remaining after a given semester (1-8).

    Returns the sum of credit hours for every semester AFTER current_semester,
    all the way to semester 8. Use it to find out how many credit hours are
    left until graduation. To find the credit hours for just a shorter
    horizon (e.g. only the next one or two semesters), call this tool for
    two different semester numbers and take the difference — e.g. the
    credit hours in semester 5 alone are
    get_remaining_credit_hours(4) - get_remaining_credit_hours(5).
    """
    if current_semester not in COURSES:
        return "Error: semester must be between 1 and 8"
    remaining = sum(
        ch for sem, courses in COURSES.items() if sem > current_semester for _, _, ch in courses
    )
    return round(remaining, 3)


@tool
def save_report(filename: str, content: str) -> str:
    """Save a GPA/CGPA summary or plan to a text file and confirm it was saved.

    Only call this after the student has explicitly asked you to save
    something, and only when there is already a real result to save (a
    semester GPA, a projected CGPA, or a plan for reaching a target).
    """
    if not filename.strip():
        return "Error: filename must not be empty"
    if not filename.endswith(".txt"):
        filename = filename + ".txt"
    try:
        with open(filename, "w") as f:
            f.write(content)
    except OSError as e:
        return f"Error: could not save file ({e})"
    return f"Saved report to {filename}"
