"""
Student Grading System
Day 1 Task - MLB Internship

Takes student name, class, subjects, and marks as input,
calculates the average, and assigns a grade.
"""


def get_student_info():
    """Collects basic student info from user input."""
    name = input("Enter student name: ")
    student_class = input("Enter student class: ")
    return name, student_class


def get_marks():
    """Collects subject names and marks from the user.
    Returns a dictionary: {subject_name: marks}
    """
    marks = {}  # dictionary -> subject: marks
    num_subjects = int(input("How many subjects? "))

    for i in range(num_subjects):
        subject = input(f"Enter name of subject {i + 1}: ")
        mark = float(input(f"Enter marks for {subject} (out of 100): "))
        marks[subject] = mark

    return marks


def calculate_average(marks):
    """Calculates average marks from a dictionary of subject:marks."""
    total = sum(marks.values())
    average = total / len(marks)
    return average


def assign_grade(average):
    """Assigns a grade based on the average marks."""
    if average >= 90:
        return "A"
    elif average >= 75:
        return "B"
    elif average >= 50:
        return "C"
    else:
        return "F"


def display_report(name, student_class, marks, average, grade):
    """Prints a clean report card."""
    print("\n" + "=" * 35)
    print("        STUDENT REPORT CARD")
    print("=" * 35)
    print(f"Name       : {name}")
    print(f"Class      : {student_class}")
    print("-" * 35)
    print("Subject-wise Marks:")
    for subject, mark in marks.items():
        print(f"  {subject:<15}: {mark}")
    print("-" * 35)
    print(f"Average    : {average:.2f}")
    print(f"Grade      : {grade}")
    print("=" * 35)


def main():
    name, student_class = get_student_info()
    marks = get_marks()
    average = calculate_average(marks)
    grade = assign_grade(average)
    display_report(name, student_class, marks, average, grade)


if __name__ == "__main__":
    main()