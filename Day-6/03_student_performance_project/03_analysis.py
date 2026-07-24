import pandas as pd

#Task 1: load dataset and show basic info

df = pd.read_csv('Day-6/03_student_performance_project/students_original.csv')
df.to_csv('Day-6/03_student_performance_project/students_processed.csv', index=False)

print("----- First 5 rows -----")


print(df.head())

print("\n----- Dataset Info -----")


print(df.info())

print("\n----- Dataset Shape -----")


print(df.shape)


#Task 2: average marks for each subject

avg_python = df['Python'].mean()
avg_maths = df['Mathematics'].mean()
avg_stats = df['Statistics'].mean()
avg_ml = df['Machine_Learning'].mean()

print("\n----- Average Marks per Subject -----")

print(f'Average Python marks: {avg_python}')

print(f'Average Mathematics marks: {avg_maths}')

print(f'Average Statistics marks: {avg_stats}')

print(f'Average Machine Learning marks: {avg_ml}')


#Task 3: top 5 performing students




df['Overall_Average'] = df[['Python', 'Mathematics', 'Statistics', 'Machine_Learning']].mean(axis=1)

top_5_students = df.sort_values(by='Overall_Average', ascending=False).head(5)

print("\n----- Top 5 Performing Students -----")

print(top_5_students[['Student_ID', 'Name', 'Overall_Average']])


#Task 4: students scoring below average

class_average = df['Overall_Average'].mean()

below_average_students = df[df['Overall_Average'] < class_average]

print(f"\n----- Students Below Class Average ({class_average:.2f}) -----")


print(below_average_students[['Student_ID', 'Name', 'Overall_Average']])


#Task 5: total number of students

total_students = len(df)


print(f"\nTotal number of students: {total_students}")


#Task 6: save the processed dataset as a new csv file

df.to_csv('students_processed.csv', index=False)


print("\nProcessed dataset saved as students_processed.csv")