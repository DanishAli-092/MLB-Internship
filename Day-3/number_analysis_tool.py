#Day-3
#MLB-internship
#Danish Ali
#Mini Challenge - Number Analysis Tool

num = int(input("enter a number to analyze: "))
print()

original = num
print(f"Analysis Report for: {original}")
print("-" * 30)

# even or odd
if num % 2 == 0:
    print("Even or Odd  : Even")
else:
    print("Even or Odd  : Odd")

# prime check
is_prime = True
if num < 2:
    is_prime = False
else:
    for i in range(2, num):
        if num % i == 0:
            is_prime = False
            break

if is_prime:
    print("Prime Number : Yes")
else:
    print("Prime Number : No")

# digit count
digit = abs(num)
count_digit = 0
if digit == 0:
    count_digit = 1
else:
    while digit > 0:
        count_digit += 1
        digit = digit // 10

print(f"Digit Count  : {count_digit}")

# reverse number
temp = abs(num)
reversed_num = 0
while temp > 0:
    last_digit = temp % 10
    reversed_num = reversed_num * 10 + last_digit
    temp = temp // 10

print(f"Reversed     : {reversed_num}")

# palindrome check
if abs(num) == reversed_num:
    print("Palindrome   : Yes")
else:
    print("Palindrome   : No")

print("-" * 30)