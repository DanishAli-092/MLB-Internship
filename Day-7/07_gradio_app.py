import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned data
df = pd.read_csv("03_cleaned_student_performance.csv")
subject_cols = ["Python", "Mathematics", "Statistics", "Machine_Learning"]


def get_summary(programs):
    filtered = filter_data(programs)
    total_students = filtered.shape[0]
    subject_avg = filtered[subject_cols].mean().round(2)
    highest_avg_subject = subject_avg.idxmax()
    needs_improvement = filtered[filtered["Performance"] == "Needs Improvement"]

    # Build a proper markdown table for subject averages
    table_rows = "\n".join([f"| {subject} | {avg} |" for subject, avg in subject_avg.items()])

    summary_text = f"""### Key Metrics
- **Total Students:** {total_students}
- **Subject with Highest Average:** {highest_avg_subject} ({subject_avg.max()})
- **Students Needing Improvement:** {needs_improvement.shape[0]}

### Average Score per Subject
| Subject | Average Marks |
|---------|---------------|
{table_rows}
"""
    return summary_text


def filter_data(programs):
    if not programs:
        filtered = df
    else:
        filtered = df[df["Program"].isin(programs)]
    return filtered


def get_top5(programs):
    filtered = filter_data(programs)
    top5 = filtered.sort_values(by="Average_Score", ascending=False).head(5)
    return top5[["Name", "Program", "Average_Score", "Performance"]]


def get_needs_improvement(programs):
    filtered = filter_data(programs)
    needs_improvement = filtered[filtered["Performance"] == "Needs Improvement"]
    if needs_improvement.empty:
        return pd.DataFrame({"Message": ["No students in this category."]})
    return needs_improvement[["Name", "Program", "Average_Score"]]


def plot_line(programs):
    filtered = filter_data(programs)
    filtered = filtered.sort_values(by="Average_Score", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(filtered["Name"], filtered["Average_Score"], marker="o", color="teal")
    plt.xticks(rotation=75)
    ax.set_xlabel("Student")
    ax.set_ylabel("Average Score")
    ax.set_title("Average Score Trend Across Students")
    plt.tight_layout()
    return fig


def plot_histogram(programs):
    filtered = filter_data(programs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hist(filtered["Average_Score"], bins=8, color="skyblue", edgecolor="black")
    ax.set_xlabel("Average Score")
    ax.set_ylabel("Number of Students")
    ax.set_title("Distribution of Average Score")
    plt.tight_layout()
    return fig


def plot_bar(programs):
    filtered = filter_data(programs)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=filtered, x="Name", y="Average_Score", hue="Name",
                palette="viridis", legend=False, ax=ax)
    plt.xticks(rotation=75)
    ax.set_title("Average Score per Student")
    plt.tight_layout()
    return fig


def plot_pie(programs):
    filtered = filter_data(programs)
    fig, ax = plt.subplots(figsize=(5, 5))
    counts = filtered["Performance"].value_counts()
    ax.pie(counts, labels=counts.index, autopct="%1.1f%%", colors=sns.color_palette("pastel"))
    ax.set_title("Performance Distribution")
    return fig


def plot_scatter(programs):
    filtered = filter_data(programs)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.scatterplot(data=filtered, x="Python", y="Machine_Learning", hue="Program", s=80, ax=ax)
    ax.set_title("Python vs Machine Learning Marks")
    return fig


def plot_box(programs):
    filtered = filter_data(programs)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.boxplot(data=filtered[subject_cols], ax=ax)
    ax.set_title("Marks Distribution Across Subjects")
    return fig


def update_all(programs):
    return (
        get_summary(programs),
        get_top5(programs),
        get_needs_improvement(programs),
        plot_line(programs),
        plot_bar(programs),
        plot_histogram(programs),
        plot_pie(programs),
        plot_scatter(programs),
        plot_box(programs),
    )


with gr.Blocks(title="Student Performance Dashboard") as demo:
    gr.Markdown("# Student Performance Dashboard")
    gr.Markdown("Day 7 Mini Project - MLB Internship (Gradio version) by Danish")

    program_filter = gr.CheckboxGroup(
        choices=list(df["Program"].unique()),
        value=list(df["Program"].unique()),
        label="Filter by Program"
    )

    summary_box = gr.Markdown()

    with gr.Row():
        top5_table = gr.Dataframe(label="Top 5 Students")
        improvement_table = gr.Dataframe(label="Students Needing Improvement")

    gr.Markdown("## Visualizations")

    with gr.Row():
        with gr.Column():
            gr.Markdown("**1. Line Chart — Average Score Trend**")
            line_plot = gr.Plot(show_label=False)
        with gr.Column():
            gr.Markdown("**2. Bar Chart — Average Score per Student**")
            bar_plot = gr.Plot(show_label=False)

    with gr.Row():
        with gr.Column():
            gr.Markdown("**3. Histogram — Average Score Distribution**")
            histogram_plot = gr.Plot(show_label=False)
        with gr.Column():
            gr.Markdown("**4. Pie Chart — Performance Distribution**")
            pie_plot = gr.Plot(show_label=False)

    with gr.Row():
        with gr.Column():
            gr.Markdown("**5. Scatter Plot — Python vs Machine Learning Marks**")
            scatter_plot = gr.Plot(show_label=False)
        with gr.Column():
            gr.Markdown("**6. Box Plot — Marks Distribution Across Subjects**")
            box_plot = gr.Plot(show_label=False)

    # Load initial values and refresh on filter change
    demo.load(
        fn=update_all,
        inputs=program_filter,
        outputs=[summary_box, top5_table, improvement_table, line_plot, bar_plot,
                 histogram_plot, pie_plot, scatter_plot, box_plot],
    )
    program_filter.change(
        fn=update_all,
        inputs=program_filter,
        outputs=[summary_box, top5_table, improvement_table, line_plot, bar_plot,
                 histogram_plot, pie_plot, scatter_plot, box_plot],
    )

if __name__ == "__main__":
    demo.launch()