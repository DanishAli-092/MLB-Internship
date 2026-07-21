# Day-3
# Danish Ali
# MLB-INTERNSHIP

while True:
    print("-" * 30)
    print("1. print numbers 1 to 100")
    print("2. print even numbers 1 to 100")
    print("3. sum of numbers 1 to N")
    print("4. multiplication table")
    print("5. count digits in a number")
    print("6. exit")
    print("-" * 30)

    choice = input("enter choice: ")
    print()

    if choice == "1":
        print("numbers 1 to 100:")
        for i in range(1, 101):
            print(i, end=" ")
        print()

    elif choice == "2":
        print("even numbers 1 to 100:")
        for i in range(1, 101):
            if i % 2 == 0:
                print(i, end=" ")
        print()

    elif choice == "3":
        n = int(input("enter N: "))
        total_sum = 0
        for i in range(1, n + 1):
            total_sum += i
        print(f"sum of 1 to {n} is {total_sum}")

    elif choice == "4":
        table_num = int(input("enter number for table: "))
        print(f"multiplication table of {table_num}:")
        i = 1
        while i <= 10:
            print(f"{table_num} x {i} = {table_num * i}")
            i += 1

    elif choice == "5":
        user_input = int(input("enter a number: "))
        digit = abs(user_input)
        count_digit = 0
        if digit == 0:
            count_digit = 1
        else:
            while digit > 0:
                count_digit += 1
                digit = digit // 10
        print(f"total digits: {count_digit}")

    elif choice == "6":
        print("exiting...")
        break

    else:
        print("invalid choice, try again")