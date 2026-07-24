import pandas as pd

#load dataste

df = pd.read_csv('Day-6/03_student_performance_project/students_original.csv')
print("First 5 rows from dataset:")
print(df.head())

print("\nLast 5 rows from dataset:")
print(df.tail())


# Task 2: Display Dataset Information

# 1. df.info() -> structure: columns, dtypes, non-null counts, memory
print("----- Dataset Info -----")

print(df.info())

# 2. df.describe() -> statistical summary (mean, std, min, max, quartiles)
print("\n----- Statistical Summary ----------")

print(df.describe())

# 3. df.shape -> (rows, columns)
print("\n----- Dataset Shape ------------")

print(df.shape)

# 4. df.columns -> column names
print("\n----- Column Names ----------------")

print(df.columns)



print(df.isnull())


#Task 3: Find Missing Values
print(df.isnull().sum())

total_missing=df.isnull().sum().sum()

print(f'total missing values:{total_missing}')

missing_rows=df[df.isnull().any(axis=1)]

print(missing_rows)

#Task 4: filter data based on condition
python_filter=df[df['Python']>80]

print(python_filter)

ai_students=df[df['Program']=='AI']

print(ai_students)

#Task 5: summary statistics
print(df.describe())

avg_python=df['Python'].mean()

avg_ml=df['Machine_Learning'].mean()

max_attendance=df['Attendance'].max()

print(f'avg python marks:{avg_python}')


print(f'avg ML marks:{avg_ml}')

print(f'highest attendance:{max_attendance}')


