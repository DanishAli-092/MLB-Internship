# Day 1 — Python & Git Fundamentals

## Topics Covered
### Python Fundamentals
- Virtual Environments — isolating project dependencies
- Data Types — int, float, str, bool
- Variables — dynamic typing in Python
- Data Structures — List, Tuple, Set, Dictionary
- Functions — writing reusable, modular code
- Conditional Operators — if/elif/else, comparison & logical operators

### Git & GitHub
- Git vs GitHub — local version control vs remote hosting
- Creating a repository
- Cloning a repository locally
- Branching workflow
- Commit & push workflow

## Task: Student Grading System
`grading_system.py` takes student name, class, subjects, and marks as input,
calculates the average, and assigns a grade using the following criteria:

| Average Range | Grade |
|----------------|-------|
| 90 - 100       | A     |
| 75 - 89        | B     |
| 50 - 74        | C     |
| Below 50       | F     |

### How to Run
```bash
python grading_system.py
```

## Key Learnings
- Dictionaries are ideal for subject:marks pairs (key-value lookup).
- Breaking the program into small functions (get_marks, calculate_average,
  assign_grade, display_report) keeps the code readable and easy to debug.
- Used a virtual environment to keep this project's dependencies isolated.

