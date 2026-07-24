import numpy as np

# 1. Create 1D and 2D arrays
print("----- 1D Array -----")
array_1d = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(array_1d)

print("----- 2D Array -----")
array_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(array_2d)

print("Array 1 shape:", array_1d.shape)


print("Array 1 data type:", array_1d.dtype)

print("Array 1 dimensions:", array_1d.ndim)

print("Array 2 shape:", array_2d.shape)

print("Array 2 data type:", array_2d.dtype)

print("Array 2 dimensions:", array_2d.ndim)


# 2. Perform arithmetic operations on arrays
print("\n----- Arithmetic Operations -----")
a1 = np.array([1, 2, 3, 4])

a2 = np.array([5, 6, 7, 8])

print("Addition:", a1 + a2)

print("Subtraction:", a1 - a2)

print("Multiplication:", a1 * a2)

print("Division:", a1 / a2)

print("Dot product:", a1 @ a2)


# 3. Find the maximum, minimum, mean, and sum of an array
print("\n----- Max, Min, Mean, Sum -----")

numbers = np.array([23, 45, 12, 67, 34, 89, 21])


print("Array:", numbers)

max_value = numbers.max()

min_value = numbers.min()

mean_value = numbers.mean()

sum_value = numbers.sum()

print("Max:", max_value)

print("Min:", min_value)

print("Mean:", mean_value)

print("Sum:", sum_value)

print("Index of max value (argmax):", numbers.argmax())

print("Index of min value (argmin):", numbers.argmin())


# 4. Reshape arrays into different dimensions
print("\n----- Reshaping Arrays -----")

sequence = np.arange(1, 13)

print("Original array:", sequence)

print("Reshaped into 4x3:\n", sequence.reshape(4, 3))

print("Reshaped into 3x4:\n", sequence.reshape(3, 4))

print("Reshaped into 2x6:\n", sequence.reshape(2, 6))

print("Reshaped using -1 (3 rows, columns auto-calculated):\n", sequence.reshape(3, -1))


# 5. Slice and index a 1D array
print("\n----- Slicing and Indexing (1D) -----")

values = np.array([10, 20, 30, 40, 50, 60, 70, 80])

first_three = values[0:3]

print("First 3 elements:", first_three)

last_three = values[5:]

print("Last 3 elements:", last_three)

every_second = values[::2]

print("Every 2nd element:", every_second)

reversed_array = values[::-1]

print("Reversed array:", reversed_array)

middle_slice = values[2:6]
print("Elements from index 2 to 5:", middle_slice)


# 6, 7, 8. Slice and index a 2D array
print("\n----- Slicing and Indexing (2D) -----")
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("Matrix:\n", matrix)

middle_row = matrix[1, :]
print("Middle row:", middle_row)

last_column = matrix[:, -1]
print("Last column:", last_column)

greater_than_five = matrix[matrix > 5]
print("Elements greater than 5:", greater_than_five)