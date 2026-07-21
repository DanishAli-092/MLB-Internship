#Day-3
#MLB-internship
#Danish Ali
#Bonus Challenge - Menu Driven App

while True:
    print("-" * 30)
    print("1. check prime number")
    print("2. generate fibonacci series")
    print("3. check palindrome")
    print("4. multiplication table")
    print("5. exit")
    print("-" * 30)

    choice = input("enter choice: ")
    print()

    if choice == "1":
        num = int(input("enter a number: "))
        is_prime = True
        if num < 2:
            is_prime = False
        else:
            for i in range(2, num):
                if num % i == 0:
                    is_prime = False
                    break

        if is_prime:
            print(f"{num} is a prime number")
        else:
            print(f"{num} is not a prime number")

    elif choice == "2":
        n = int(input("how many terms: "))
        first = 0
        second = 1
        count = 0
        while count < n:
            print(first, end=" ")
            new_num = first + second
            first = second
            second = new_num
            count += 1
        print()

    elif choice == "3":
        num = int(input("enter a number: "))
        temp = num
        reversed_num = 0
        while temp > 0:
            last_digit = temp % 10
            reversed_num = reversed_num * 10 + last_digit
            temp = temp // 10

        if num == reversed_num:
            print(f"{num} is a palindrome")
        else:
            print(f"{num} is not a palindrome")

    elif choice == "4":
        table_num = int(input("enter number for table: "))
        i = 1
        while i <= 10:
            print(f"{table_num} x {i} = {table_num * i}")
            i += 1

    elif choice == "5":
        print("exiting...")
        break

    else:
        print("invalid choice, try again")
        