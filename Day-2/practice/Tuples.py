# Day 2 - Tuples Practice
# MLB Internship

my_tuple = (1, 2, 3, 2, 4, 2, 5)

# 1. Count occurrences of an element
def count_occurrence(tup, value):
    count = 0
    for item in tup:
        if item == value:
            count += 1
    return count

# 2. Convert tuple to list and back to tuple
def tuple_to_list(tup):
    return list(tup)

def list_to_tuple(lst):
    return tuple(lst)


if __name__ == "__main__":
    print("Tuple:", my_tuple)
    print("Occurrences of 2:", count_occurrence(my_tuple, 2))

    converted_list = tuple_to_list(my_tuple)
    print("Converted to list:", converted_list)

    converted_back = list_to_tuple(converted_list)
    print("Converted back to tuple:", converted_back)

    # Quick note for myself:
    # Tuples are used when the data should not change (like coordinates, days of week etc.)
    # Lists are used when we need to add/remove/update items often.