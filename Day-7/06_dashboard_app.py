import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Student Performance Dashboard", layout="wide")

# Load cleaned data
df = pd.read_csv("03_cleaned_student_performance.csv")
subject_cols = ["Python", "Mathematics", "Statistics", "Machine_Learning"]

st.title("Student Performance Dashboard")
st.write("Day 7 Mini Project - MLB Internship")
st.write("By Danish")

# ---- Filters (defined first, everything below uses filtered_df) ----
st.sidebar.header("Filters")
program_filter = st.sidebar.multiselect(
    "Filter by Program", options=df["Program"].unique(), default=df["Program"].unique()
)
filtered_df = df[df["Program"].isin(program_filter)]

# ---- Key Metrics ----
total_students = filtered_df.shape[0]
subject_avg = filtered_df[subject_cols].mean().round(2)
highest_avg_subject = subject_avg.idxmax()
needs_improvement = filtered_df[filtered_df["Performance"] == "Needs Improvement"]

col1, col2, col3 = st.columns(3)
col1.metric("Total Students", total_students)
col2.metric("Highest Avg Subject", highest_avg_subject, f"{subject_avg.max()}")
col3.metric("Needs Improvement", needs_improvement.shape[0])

st.subheader("Average Score per Subject")
st.dataframe(subject_avg.rename("Average Marks"))

st.subheader("Top 5 Students")
top5 = filtered_df.sort_values(by="Average_Score", ascending=False).head(5)
st.dataframe(top5[["Name", "Program", "Average_Score", "Performance"]])

st.subheader("Students Needing Improvement")
if needs_improvement.empty:
    st.write("No students in this category.")
else:
    st.dataframe(needs_improvement[["Name", "Program", "Average_Score"]])

st.subheader("Filtered Data")
st.dataframe(filtered_df)

# ---- Visualizations ----
st.subheader("Visualizations")

sns.set_style("whitegrid")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**1. Line Chart — Average Score Trend**")
    sorted_df = filtered_df.sort_values(by="Average_Score", ascending=False)
    fig, ax = plt.subplots()
    ax.plot(sorted_df["Name"], sorted_df["Average_Score"], marker="o", color="teal")
    plt.xticks(rotation=75)
    ax.set_xlabel("Student")
    ax.set_ylabel("Average Score")
    st.pyplot(fig)

with col2:
    st.markdown("**2. Bar Chart — Average Score per Student**")
    fig, ax = plt.subplots()
    sns.barplot(data=filtered_df, x="Name", y="Average_Score", hue="Name",
                palette="viridis", legend=False, ax=ax)
    plt.xticks(rotation=75)
    st.pyplot(fig)

col3, col4 = st.columns(2)

with col3:
    st.markdown("**3. Histogram — Average Score Distribution**")
    fig, ax = plt.subplots()
    ax.hist(filtered_df["Average_Score"], bins=8, color="skyblue", edgecolor="black")
    ax.set_xlabel("Average Score")
    ax.set_ylabel("Number of Students")
    st.pyplot(fig)

with col4:
    st.markdown("**4. Pie Chart — Performance Distribution**")
    fig, ax = plt.subplots()
    performance_counts = filtered_df["Performance"].value_counts()
    ax.pie(performance_counts, labels=performance_counts.index, autopct="%1.1f%%",
           colors=sns.color_palette("pastel"))
    st.pyplot(fig)

col5, col6 = st.columns(2)

with col5:
    st.markdown("**5. Scatter Plot — Python vs Machine Learning Marks**")
    fig, ax = plt.subplots()
    sns.scatterplot(data=filtered_df, x="Python", y="Machine_Learning", hue="Program", s=80, ax=ax)
    st.pyplot(fig)

with col6:
    st.markdown("**6. Box Plot — Marks Distribution Across Subjects**")
    fig, ax = plt.subplots()
    sns.boxplot(data=filtered_df[subject_cols], ax=ax)
    st.pyplot(fig)