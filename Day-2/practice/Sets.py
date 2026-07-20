# Day 2 - Sets Practice
# MLB Internship

marks_list = [55, 67, 55, 89, 90, 67, 45, 89]

# 1. Find unique values from a list using set
unique_marks = set(marks_list)
print("Original list:", marks_list)
print("Unique values (set):", unique_marks)

# 2. Union and Intersection
set_a = {1, 2, 3, 4, 5}
set_b = {4, 5, 6, 7, 8}

union_result = set_a.union(set_b)
intersection_result = set_a.intersection(set_b)
difference_result = set_a.difference(set_b)

print("\nSet A:", set_a)
print("Set B:", set_b)
print("Union:", union_result)
print("Intersection:", intersection_result)
print("Difference (A - B):", difference_result)