
#05 - Read data from a JSON file.

import json

def load_students(filename):
    
    with open(filename, "r") as f:
        
        return json.load(f)  


if __name__ == "__main__":
    try:
        students = load_students("students.json")
        
        print("--- student records -------------")
        for s in students:
            print(f"Roll No: {s['roll_no']} | Name: {s['name']} | "
                  
                  f"Dept: {s['department']} | GPA: {s['gpa']}")
    except FileNotFoundError:
        
        
        print("students.json not found. Run 04_json_store_students.py first.")