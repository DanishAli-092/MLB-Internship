#Day-3
#MLB-internship
#Danish Ali

while True:
    print("-" * 30)
    print("1. reverse a number")
    print("2. check palindrome")
    print("3. fibonacci sequence")
    print("4. check prime number")
    print("5. primes between 1 and 100")
    print("6. exit")
    print("-" * 30)

    choice = input("enter choice: ")
    print()

    if choice == "1":
        num = int(input("enter a number: "))
        original = num
        reversed_num = 0
        while num > 0:
            last_digit = num % 10
            reversed_num = reversed_num * 10 + last_digit
            num = num // 10
        print(f"reverse of {original} is {reversed_num}")

    elif choice == "2":
        num = int(input("enter a number: "))
        original = num
        temp = num
        reversed_num = 0
        while temp > 0:
            last_digit = temp % 10
            reversed_num = reversed_num * 10 + last_digit
            temp = temp // 10

        if original == reversed_num:
            print(f"{original} is a palindrome")
        else:
            print(f"{original} is not a palindrome")

    elif choice == "3":
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

    elif choice == "4":
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

    elif choice == "5":
        print("prime numbers between 1 and 100:")
        for num in range(2, 101):
            is_prime = True
            for i in range(2, num):
                if num % i == 0:
                    is_prime = False
                    break
            if is_prime:
                print(num, end=" ")
        print()

    elif choice == "6":
        print("exiting...")
        break

    else:
        print("invalid choice, try again")