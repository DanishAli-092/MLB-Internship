"""
04 ----<>>>>>>> Store student information in a JSON file.
JSON = JavaScript Object Notation. It's a lightweight, human-readable
data format that maps very naturally to Python dicts/lists which is

why it is the standard way to save structured data to disk.
"""
import json

students = [
    {"roll_no": 1, "name": "Danish Ali", "department": "CS", "gpa": 3.6},
    
    {"roll_no": 2, "name": "Sara Khan", "department": "SE", "gpa": 3.9},
    
    {"roll_no": 3, "name": "Tania Ahmed", "department": "AI", "gpa": 3.4},
]

def save_students(filename, data):
    
    with open(filename, "w") as f:
        
        json.dump(data, f, indent=4)  
        
        
        
    print(f"Saved {len(data)} students to {filename}")


if __name__ == "__main__":
    
    save_students("students.json", students)