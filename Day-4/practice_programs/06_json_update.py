
#06 - Update an existing student's information in the JSON file.

import json

def load_students(filename):
    
    with open(filename, "r") as f:
        
        return json.load(f)


def save_students(filename, data):
    
    with open(filename, "w") as f:
        
        json.dump(data, f, indent=4)


def update_student(filename, roll_no, new_data):
    
    students = load_students(filename)
    
    updated = False

    for s in students:
        if s["roll_no"] == roll_no:
            s.update(new_data) 
            updated = True
            break

    if updated:
        
        save_students(filename, students)
        
        
        print(f"Student with roll_no {roll_no} updated.")
        
    else:
        
        print(f"No student found with roll_no {roll_no}.")


if __name__ == "__main__":
    
    # Example: update Sara Khan's GPA
    
    update_student("students.json", roll_no=2, new_data={"gpa": 4.0})