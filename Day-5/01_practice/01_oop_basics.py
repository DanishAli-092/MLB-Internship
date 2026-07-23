#Day-5
#MLB-Internship 
#oop

#Student class

# ============ Student Class ============
class Student:
    def __init__(self, name, roll_no, marks):
        self.name = name
        self.roll_no = roll_no
        self.marks = marks

    def show_details(self):
        print(f"Name: {self.name}, Roll No: {self.roll_no}, Marks: {self.marks}")

    def result(self):
        if self.marks >= 50:
            print(f"{self.name} passed the exam.")
        else:
            print(f"{self.name} failed the exam.")


# ============ Employee Class ============
class Employee:
    def __init__(self, name, employee_id, salary):
        self.name = name
        self.employee_id = employee_id
        self.salary = salary

    def show_details(self):
        print(f"Employee: {self.name} ID: {self.employee_id}  Salary: {self.salary}")

    def give_raise(self, amount):
        self.salary += amount
        print(f"{self.name}'s new salary: {self.salary}")


# ============ Car Class ============
class Car:
    def __init__(self, brand, model, price):
        self.brand = brand
        self.model = model
        self.price = price

    def show_details(self):
        print(f"{self.brand} {self.model} costs {self.price}")


# ============ Main Program ============
if __name__ == "__main__":
    print("----- Student Objects -----")
    s1 = Student("Danish Ali", "L1F22BSCS0121", 85)
    
    s2 = Student("Sara", "L1F22BSCS0011", 40)
    s1.show_details()
    s1.result()
    s2.show_details()
    s2.result()

    print("\n----- Employee Objects -----")
    
    e1 = Employee("Danish", "EMP001", 50000)
    
    e2 = Employee("Hamza", "EMP002", 60000)
    
    e1.show_details()
    
    e1.give_raise(5000)
    
    e2.show_details()

    print("\n----- Car Objects (Multiple Objects Demo) -----")
    
    car1 = Car("Toyota", "Corolla", 5000000)
    
    car2 = Car("Honda",  "Civic", 6500000)
    
    car3 = Car("Suzuki", "Alto", 2500000)

    for car in [car1, car2, car3]:
        
        car.show_details()
    
    
    
        
        
         