# Day 4 - File Handling & JSON in Python

## Overview

Up until yesterday, my Student Record System had a fatal flaw: the second you closed the program, every record vanished. All that data entry, gone. Today was about fixing that for good — learning how Python talks to files, how JSON keeps that data structured, and turning my Day-2 project into something that actually remembers.

## Folder Structure

```
Day-4/
├── practice_programs/
│   ├── 01_file_write_read.py
│   ├── 02_file_append.py
│   ├── 03_count_lines.py
│   ├── 04_json_store_students.py
│   ├── 05_json_read.py
│   ├── 06_json_update.py
│   └── 07_json_add_student.py
├── student_management/
│   ├── student_management_system.py
│   └── students.json
└── README.md
```

## Topics Covered

**File Handling:**
- Opening, reading, and writing files
- Appending data without wiping what's already there
- File modes — `r`, `w`, `a` — and when each one actually makes sense
- Using `with` so files close properly on their own

**JSON:**
- What JSON is and why it's the go-to format for structured data
- Reading and writing JSON files
- Converting Python dictionaries into JSON
- Loading JSON data back into real Python objects

**Exception Handling** — because none of this matters if one bad input crashes the whole program.

## Practice Programs

**File Handling:**
- Created a text file and wrote data into it
- Read and displayed file contents
- Appended new data to an existing file
- Counted the number of lines in a file

**JSON:**
- Stored student information in a JSON file
- Read data back from a JSON file
- Updated an existing student's information
- Added a new student to the JSON file

## Student Record Management System (Persistent Version)

This is the real deal now. Features:

- Add Student
- View All Students
- Search Student (by name or roll number)
- Update Student Information
- Delete Student
- Auto-loads existing records from `students.json` when the program starts
- Auto-saves every change back to the file, immediately
- Input validation + exception handling throughout

The difference from the Day-2 version is what happens behind the scenes: the moment the program starts, it pulls in whatever records already exist. And the moment you add, update, or delete something, it saves right then — not when you exit. Close the program mid-session, crash your laptop, whatever — the data's already on disk.

Each record holds a **name, roll number, age, and course**. Nothing fancy, but enough to actually be useful.

## What I Learned

- A Python list is temporary by default  it only exists while the program is running. Files are what make things actually stick.
- JSON is the bridge between memory and disk  it's what lets a list of dictionaries become readable text, and turn back into that same list later without losing structure.
- `try/except` isn't some abstract safety net  it's the actual thing standing between "user typed a letter instead of a number" and "program crashes and loses everything."
- How to build something that behaves like real software  not just a script that works while it happens to be open.

## How File Handling and JSON Work Together

Opening a file with `open()` just gets you access — it doesn't care what's inside. JSON is what gives that content meaning. `json.dump()` takes my list of student dictionaries and turns it into properly formatted text that a file can actually hold. `json.load()` does the reverse — reads that text and rebuilds it into the exact Python objects I started with. So file handling is the door, JSON is what makes sense of what's on the other side of it.

## Challenges Faced

- **First-run crash:** running the program for the very first time, `students.json` didn't exist yet, so trying to load it blew up immediately. Fixed by checking if the file exists first and starting with an empty list if it doesn't.
- **Changes not actually saving:** my updates looked fine while the program was running but disappeared after closing it. I was only saving once at the very end instead of after every change. Moved the save call to right after every add, update, and delete.
- **Wrong working directory:** I was running the script one folder above `student_management`, so Python kept reading and writing a `students.json` that wasn't the one I thought it was. This eventually threw a confusing error (`'int' object has no attribute 'lower'`) that only made sense once I realized it was pulling data from an entirely different file. Lesson: always check where you actually are before hitting run.
- **Bad input crashing the program:** typing a letter where a number was expected (for age) crashed things until I wrapped that input in `try/except`  now it just asks again instead of taking the whole program down with it.

## Files Included

- Practice Programs
- Updated Student Record Management System
- Sample JSON File
- README.md

## Conclusion

Day 4 was less about learning new syntax and more about understanding *why* persistence matters — the difference between a program that "works" while it's open and one that actually holds onto information the way real software has to. Files and JSON together are what make that possible, and honestly this feels like the first project that's starting to look like something more than a classroom exercise.