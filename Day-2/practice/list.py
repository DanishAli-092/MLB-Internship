# Day 2 - Lists Practice
# MLB Internship

numbers = [12, 45, 3, 78, 90, 90, 21, 5]

# 1. Find the largest number in a list
def find_largest(lst):
    largest = lst[0]
    for n in lst:
        if n > largest:
            largest = n
    return largest

# 2. Find the second largest number
def find_second_largest(lst):
    first = second = None
    for n in lst:
        if first is None or n > first:
            second = first
            first = n
        elif n != first and (second is None or n > second):
            second = n
    return second

# 3. Remove duplicate values from a list
def remove_duplicates(lst):
    new_list = []
    for n in lst:
        if n not in new_list:
            new_list.append(n)
    return new_list

# 4. Reverse a list without using built-in reverse()
def reverse_list(lst):
    reversed_list = []
    for i in range(len(lst) - 1, -1, -1):
        reversed_list.append(lst[i])
    return reversed_list

# 5. Find common elements between two lists
def common_elements(lst1, lst2):
    common = []
    for item in lst1:
        if item in lst2 and item not in common:
            common.append(item)
    return common


if __name__ == "__main__":
    print("Original list:", numbers)
    print("Largest number:", find_largest(numbers))
    print("Second largest number:", find_second_largest(numbers))
    print("List without duplicates:", remove_duplicates(numbers))
    print("Reversed list:", reverse_list(numbers))

    list_a = [1, 2, 3, 4, 5]
    list_b = [4, 5, 6, 7, 8]
    print("Common elements of", list_a, "and", list_b, "->", common_elements(list_a, list_b)) 