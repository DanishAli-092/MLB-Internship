import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load cleaned data
df = pd.read_csv("03_cleaned_student_performance.csv")

# Make sure charts folder exists
os.makedirs("05_charts", exist_ok=True)

sns.set_style("whitegrid")

# 1. Line Chart - Average score trend across students (sorted order)
plt.figure(figsize=(10, 5))
plt.plot(df["Name"], df["Average_Score"], marker="o", color="teal")
plt.xticks(rotation=75)
plt.xlabel("Student")
plt.ylabel("Average Score")
plt.title("Average Score Trend Across Students")
plt.tight_layout()
plt.savefig("05_charts/01_line_chart.png")
plt.close()

# 2. Bar Chart - Average score per student
plt.figure(figsize=(10, 5))
sns.barplot(data=df, x="Name", y="Average_Score", hue="Name", palette="viridis", legend=False)
plt.xticks(rotation=75)
plt.xlabel("Student")
plt.ylabel("Average Score")
plt.title("Average Score per Student")
plt.tight_layout()
plt.savefig("05_charts/02_bar_avg_score.png")
plt.close()

# 3. Histogram - Distribution of Average Score
plt.figure(figsize=(8, 5))
plt.hist(df["Average_Score"], bins=8, color="skyblue", edgecolor="black")
plt.xlabel("Average Score")
plt.ylabel("Number of Students")
plt.title("Distribution of Average Score")
plt.tight_layout()
plt.savefig("05_charts/03_histogram_avg_score.png")
plt.close()

# 4. Scatter Plot - Python vs Machine Learning marks
plt.figure(figsize=(7, 6))
sns.scatterplot(data=df, x="Python", y="Machine_Learning", hue="Program", s=80)
plt.xlabel("Python Marks")
plt.ylabel("Machine Learning Marks")
plt.title("Python vs Machine Learning Marks")
plt.tight_layout()
plt.savefig("05_charts/04_scatter_python_ml.png")
plt.close()

# 5. Pie Chart - Performance categories
performance_counts = df["Performance"].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(performance_counts, labels=performance_counts.index, autopct="%1.1f%%",
        colors=sns.color_palette("pastel"))
plt.title("Performance Category Distribution")
plt.tight_layout()
plt.savefig("05_charts/05_pie_performance.png")
plt.close()

# 6. Box Plot - Marks across all subjects
subject_cols = ["Python", "Mathematics", "Statistics", "Machine_Learning"]
plt.figure(figsize=(8, 5))
sns.boxplot(data=df[subject_cols])
plt.ylabel("Marks")
plt.title("Marks Distribution Across Subjects")
plt.tight_layout()
plt.savefig("05_charts/06_box_subjects.png")
plt.close()

print("All 6 charts saved in 05_charts/ folder")