
#07 - Add a new student to the JSON file.

import json

def load_students(filename):
    
    with open(filename, "r") as f:
        
        return json.load(f)


def save_students(filename, data):
    
    with open(filename, "w") as f:
        
        json.dump(data, f, indent=4)


def add_student(filename, new_student):
    
    students = load_students(filename)

    # prevent duplicate roll numbers
    
    if any(s["roll_no"] == new_student["roll_no"] for s in students):
        
        print(f"Roll No {new_student['roll_no']} already exists.")
        return

    students.append(new_student)
    
    save_students(filename, students)
    
    print(f"Added new student: {new_student['name']}")


if __name__ == "__main__":
    
    new_student = {"roll_no": 4, "name": "Hina Malik", "department": "AI", "gpa": 3.8}
    
    add_student("students.json", new_student)