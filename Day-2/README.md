# Day 2 - Python Data Structures & Problem Solving

## Overview
Today I practiced Python's core data structures (Lists, Tuples, Sets, Dictionaries) and used them to solve small problems. I also built a console-based Student Record Management System using Python.

## Topics Covered
- Lists
- Tuples
- Sets
- Dictionaries
- Basic Problem Solving

## Practice Problems

**Lists**
- Find largest number in a list
- Find second largest number
- Remove duplicate values
- Reverse a list (without using `reverse()`)
- Find common elements between two lists

**Tuples**
- Count occurrences of an element
- Convert tuple into list and back

**Sets**
- Find unique values from a list
- Union and intersection of two sets

**Dictionaries**
- Create a student record dictionary
- Calculate average marks
- Count word frequency in a sentence

## What I learned
- Tuples vs lists: tuples for data that shouldn't change, lists when I need to keep adding/removing items.
- Sets are the easiest way to get unique values or compare two collections without writing loops.
- Dictionaries map naturally to real records like a student's info (name, roll no, age, course).

## Challenges Faced
- Second largest number broke on lists with repeated max values (e.g. `[90, 90, 78]`) — first attempt kept returning 90 as the "second largest" too.
- Deciding what should happen if a user leaves a field blank while updating a student record.
- Preventing the program from crashing when age is entered as text instead of a number.

## Solutions Implemented
- Tracked `first` and `second` separately, only updating `second` when the value is different from `first`.
- Update function skips a field if input is left empty, instead of overwriting it with blank data.
- Used `.isdigit()` for age and empty-string checks for other fields before saving a record.
- Checked roll number against existing records before adding, so duplicates aren't allowed.

## Project Structure
```
Day-2/
│
├── practice/
│   ├── lists.py
│   ├── tuples.py
│   ├── sets.py
│   └── dictionaries.py
│
├── student_management.py
├── README.md
└── screenshots/
```

## Student Record Management System (Version 1)
Console based app, data stored in memory (no file storage yet), using a list of dictionaries.

Features implemented:
- Add Student
- View All Students
- Search Student (by Name or Roll Number)
- Update Student Information
- Delete Student
- Display Total Number of Students (bonus)
- Menu Driven Interface (bonus)
- Basic Input Validation (bonus)

Run it with:
```
python student_management.py
```

## Screenshots
Screenshots of program execution are added in the `screenshots/` folder.