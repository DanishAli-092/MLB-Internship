#Day-3
#MLB-internship
#Danish Ali

while True:
    print("-" * 30)
    print("1. check positive, negative or zero")
    print("2. check even or odd")
    print("3. grade calculator")
    print("4. largest of three numbers")
    print("5. leap year check")
    print("6. leap year check (using logical operators)")
    print("7. exit")
    print("-" * 30)

    choice = input("enter choice: ")
    print()

    if choice == "1":
        num_check = int(input("Enter a number to check: "))
        if num_check == 0:
            print(f"{num_check} is zero.")
        elif num_check > 0:
            print(f"{num_check} is a positive number.")
        else:
            print(f"{num_check} is a negative number.")

    elif choice == "2":
        even_odd_check = int(input("Enter a number to check even or odd: "))
        if even_odd_check % 2 == 0:
            print(f"{even_odd_check} is an even number.")
        else:
            print(f"{even_odd_check} is an odd number.")

    elif choice == "3":
        marks = int(input("Enter marks: "))
        if marks >= 90:
            print(f"Marks: {marks} -> Grade: A")
        elif marks >= 80:
            print(f"Marks: {marks} -> Grade: B")
        elif marks >= 70:
            print(f"Marks: {marks} -> Grade: C")
        elif marks >= 60:
            print(f"Marks: {marks} -> Grade: D")
        else:
            print(f"Marks: {marks} -> Grade: F")

    elif choice == "4":
        num1 = int(input("Enter 1st number: "))
        num2 = int(input("Enter 2nd number: "))
        num3 = int(input("Enter 3rd number: "))
        if num1 >= num2 and num1 >= num3:
            print(f"The largest number is {num1}.")
        elif num2 >= num1 and num2 >= num3:
            print(f"The largest number is {num2}.")
        else:
            print(f"The largest number is {num3}.")

    elif choice == "5":
        check_leap_year = int(input("Enter a year to check if it's a leap year: "))
        if check_leap_year % 4 != 0:
            print(f"{check_leap_year} is not a leap year.")
        else:
            if check_leap_year % 100 != 0:
                print(f"{check_leap_year} is a leap year.")
            else:
                if check_leap_year % 400 == 0:
                    print(f"{check_leap_year} is a leap year.")
                else:
                    print(f"{check_leap_year} is not a leap year.")

    elif choice == "6":
        check_leap_year = int(input("Enter a year to check if it's a leap year: "))
        if (check_leap_year % 4 == 0 and check_leap_year % 100 != 0) or (check_leap_year % 400 == 0):
            print(f"{check_leap_year} is a leap year.")
        else:
            print(f"{check_leap_year} is not a leap year.")

    elif choice == "7":
        print("exiting...")
        break

    else:
        print("invalid choice, try again")