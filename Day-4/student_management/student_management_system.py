# Day 4 - Student Record Management System 
# MLB Internship
# Upgraded from Day-2 version: now saves data to a JSON file
# so records are not lost when the program closes.
# Invalid inputs (like age) are handled using try/except (exception handling).

import json
import os

DATA_FILE = "students.json"

students = []


# ------- Load and Save ----------------

def load_students():
    #Load students from the JSON file when the program starts.
    
    global students

    if not os.path.exists(DATA_FILE):
        
        students = []
        
        return

    try:
        
        file = open(DATA_FILE, "r")
        
        content = file.read()
        
        file.close()

        if content.strip() == "":
            students = []
        else:
            students = json.loads(content)

    except Exception as e:
        print("Could not read the file propely. startng with empty list.")
        
        print("error detail:", e)
        students = []


def save_students():
    
    #Save the current students list to the JSON file.
    
    try:
        
        file = open(DATA_FILE, "w")
        
        json.dump(students, file, indent=4)
        
        file.close()
        
    except Exception as e:
        
        print("could not save data to file.")
        
        print("Error details:", e)


# ---------------- main operations ---------------------------------

def add_student():
    
    print("--- ad new student ---")
    
    
    name = input("Enter name: ").strip()
    if name == "":
        print("name canot be empty.")
        return

    roll_no = input("enter roll number: ").strip()
    if roll_no == "":
        print("Roll number cannot be empty.")
        return

    # check roll number is not already used
    for s in students:
        
        if s["roll_no"].lower() == roll_no.lower():
            
            
            print("A student with this roll number already exists.")
            return

    # exception handling for invalid age input
    try:
        age = int(input("Enter age: ").strip())
        
    except ValueError:
        
        
        
        print("Invalid input. Age must be a number.")
        return

    course = input("enter course: ").strip()
    if course == "":
        print("course cannot b empty.")
        
        return

    new_student = {
        
        "name": name,
        
        "roll_no": roll_no,
        
        "age": age,
        
        "course": course
        
    }
    students.append(new_student)
    save_students()
    
    
    print(f"Student '{name}' added successfully.")


def view_students():
    print("--- all students ---")
    if len(students) == 0:
        print("no recods found.")
        return

    for i, s in enumerate(students, start=1):
        
        print(f"{i}. Name: {s['name']} | Roll No: {s['roll_no']} | Age: {s['age']} | Course: {s['course']}")


def search_student():
    
    
    print("\n--- search student ------------------")
    print("1. Search by Name")
    
    print("2. serch by roll number")
    
    choice = input("Enter choice: ").strip()

    if choice == "1":
        
        
        name = input("Enter name to search: ").strip().lower()
        found = False
        
        
        for s in students:
            
            
            if s["name"].lower() == name:
                print(f"Found -> Name: {s['name']} | Roll No: {s['roll_no']} | Age: {s['age']} | Course: {s['course']}")
                found = True
        if not found:
            
            print("No student found with that name.")

    elif choice == "2":
        
        roll_no = input("Enter roll number to search: ").strip().lower()
        found = False
        
        for s in students:
            
            
            if s["roll_no"].lower() == roll_no:
                print(f"Found -> Name: {s['name']} | Roll No: {s['roll_no']} | Age: {s['age']} | Course: {s['course']}")
                found = True
                
                
                break
            
        if not found:
            print("no student found with that roll number.")
    else:
        print("Invalid choice.")


def update_student():
    print("\n--- Update Student Information ----------//-")
    
    roll_no = input("Enter roll number of student to update: ").strip().lower()

    for s in students:
        
        
        if s["roll_no"].lower() == roll_no:
            
            
            print("Leave field empty if you don't want to change it.")

            new_name = input(f"New name ({s['name']}): ").strip()
            if new_name != "":
                s["name"] = new_name
                

            new_age = input(f"New age ({s['age']}): ").strip()
            if new_age != "":
                
                
                try:
                    s["age"] = int(new_age)
                    
                except ValueError:
                    
                    
                    print("Invalid age entered. Keeping old value.")

            new_course = input(f"New course ({s['course']}): ").strip()
            if new_course != "":
                
                s["course"] = new_course

            save_students()
            
            
            print("student record updated.")
            return

    print("No student found with that roll number.")


def delete_student():
    
    
    
    print("--- delete student ---------------")
    
    
    roll_no = input("Enter roll number of student to delete: ").strip().lower()

    for s in students:
        if s["roll_no"].lower() == roll_no:
            
            students.remove(s)
            
            save_students()
            
            print("student record deleted.")
            return

    print("No student found with that roll number.")


def total_students():
    
    
    print(f"\nTotal number of students: {len(students)}")


def show_menu():
    
    print("\n=====-- Student Record Management System ---=====")
    print("1. Add Student")
    
    print("2. View All Students")
    
    print("3. Search Student")
    
    print("4. Update Student Information")
    
    print("5. Delete Student")
    
    print("6. Display totl number of Students")
    
    
    print("7. Exit")


def main():
    
    
    load_students()  

    while True:
        show_menu()
        
        choice = input("Enter your choice (1-7): ").strip()

        if choice == "1":
            
            add_student()
            
        elif choice == "2":
            
            view_students()
            
        elif choice == "3":
            
            search_student()
            
        elif choice == "4":
            
            update_student()
            
        elif choice == "5":
            
            delete_student()
            
        elif choice == "6":
            total_students()
        elif choice == "7":
            
            print("All data is saved. Exiting program. Goodbye!")
            break
        else:
            
            print("Invalid choice, please enter a number between 1 and 7.")


if __name__ == "__main__":
    main()