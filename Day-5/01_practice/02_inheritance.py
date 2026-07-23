# ============ Parent Class ============
class Person:
    
    
    def __init__(self, name, age):
        
        
        self.name = name
        
        
        self.age = age

    def show_details(self):
        
        
        print(f"Name: {self.name}, Age: {self.age}")

    def introduce(self):
        
        
        print(f"Hi, I am {self.name}.")


# ============ Child Class: Student ===-=============
class Student(Person):
    
    
    def __init__(self, name, age, roll_no):
        
        
        super().__init__(name, age)   
        
        
        self.roll_no = roll_no

    def introduce(self):            
        
        
        print(f"Hi, I am {self.name}, a student with roll number {self.roll_no}.")
        


# ============ Child Class: Teacher ============
class Teacher(Person):
    
    
    def __init__(self, name, age, subject):
        
        
        super().__init__(name, age)
        
        self.subject = subject

    def introduce(self):              
        
        
        print(f"Hi, I am {self.name}, I teach {self.subject}.")


# ============ Main Program ============

if __name__ == "__main__":
    
    s1 = Student("Ali", 21, "L1F22BSCS0244")
    
    
    t1 = Teacher("Dr. Hafiz", 40, "Information Security")

    print("----- student ----------------")
    
    s1.show_details()   
    
    s1.introduce()       

    print("\n-------------- teacher -----")
    
    
    t1.show_details()   
    
    t1.introduce()       