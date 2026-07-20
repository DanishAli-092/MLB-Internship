# Day 2 - Mini Challenge
# Student Record Management System (Version 1)
# MLB Internship
# Data is stored in memory using a list of dictionaries (no file storage yet)

students = []


def add_student():
    print("\n--- Add New Student ---")
    name = input("Enter name: ").strip()
    if name == "":
        print("Name cannot be empty.")
        return

    roll_no = input("Enter roll number: ").strip()
    if roll_no == "":
        print("Roll number cannot be empty.")
        return

    # check roll number is not already used
    for s in students:
        if s["roll_no"].lower() == roll_no.lower():
            print("A student with this roll number already exists.")
            return

    age_input = input("Enter age: ").strip()
    if not age_input.isdigit():
        print("Age must be a number.")
        return
    age = int(age_input)

    course = input("Enter course: ").strip()
    if course == "":
        print("Course cannot be empty.")
        return

    new_student = {
        "name": name,
        "roll_no": roll_no,
        "age": age,
        "course": course
    }
    students.append(new_student)
    print(f"Student '{name}' added successfully.")


def view_students():
    print("\n--- All Students ---")
    if len(students) == 0:
        print("No records found.")
        return

    for i, s in enumerate(students, start=1):
        print(f"{i}. Name: {s['name']} | Roll No: {s['roll_no']} | Age: {s['age']} | Course: {s['course']}")


def search_student():
    print("\n--- Search Student ---")
    print("1. Search by Name")
    print("2. Search by Roll Number")
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
            print("No student found with that roll number.")
    else:
        print("Invalid choice.")


def update_student():
    print("\n--- Update Student Information ---")
    roll_no = input("Enter roll number of student to update: ").strip().lower()

    for s in students:
        if s["roll_no"].lower() == roll_no:
            print("Leave field empty if you don't want to change it.")

            new_name = input(f"New name ({s['name']}): ").strip()
            if new_name != "":
                s["name"] = new_name

            new_age = input(f"New age ({s['age']}): ").strip()
            if new_age != "":
                if new_age.isdigit():
                    s["age"] = int(new_age)
                else:
                    print("Invalid age, keeping old value.")

            new_course = input(f"New course ({s['course']}): ").strip()
            if new_course != "":
                s["course"] = new_course

            print("Student record updated.")
            return

    print("No student found with that roll number.")


def delete_student():
    print("\n--- Delete Student ---")
    roll_no = input("Enter roll number of student to delete: ").strip().lower()

    for s in students:
        if s["roll_no"].lower() == roll_no:
            students.remove(s)
            print("Student record deleted.")
            return

    print("No student found with that roll number.")


def total_students():
    print(f"\nTotal number of students: {len(students)}")


def show_menu():
    print("\n===== Student Record Management System =====")
    print("1. Add Student")
    print("2. View All Students")
    print("3. Search Student")
    print("4. Update Student Information")
    print("5. Delete Student")
    print("6. Display Total Number of Students")
    print("7. Exit")


def main():
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
            print("Exiting program. Goodbye!")
            break
        else:
            print("Invalid choice, please enter a number between 1 and 7.")


if __name__ == "__main__":
    main()