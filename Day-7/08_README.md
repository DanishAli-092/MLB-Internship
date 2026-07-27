# Day 7 – Data Cleaning & Visualization
**MLB Internship | Student Performance Analysis**

## 📌 Overview
This project involves cleaning a raw student performance dataset, engineering new features, visualizing key patterns using Matplotlib & Seaborn, and building a mini analytical dashboard to extract actionable insights.

---

## 🧹 Data Cleaning Steps

The raw dataset (`01_student_performance.csv`) was cleaned using `02_data_cleaning.py` with the following steps:

1. **Checked for missing values** across all columns using `df.isnull().sum()`.
2. **Handled missing values** — numeric columns were filled with their column mean, and text/object columns were filled with `"Unknown"`.
3. **Removed duplicate rows** using `df.drop_duplicates()` to ensure each student record is unique.
4. **Cleaned column names** by stripping any leading/trailing whitespace from column headers.
5. **Fixed data types** — `Age` converted to `int`; `Python`, `Mathematics`, `Statistics`, `Machine_Learning`, and `Attendance` converted to `float`.
6. **Created `Average_Score`** column as the row-wise mean of all 4 subject scores, rounded to 2 decimals.
7. **Created `Performance`** category column based on `Average_Score`:
   | Performance Category | Score Range |
   |---|---|
   | Excellent | ≥ 90 |
   | Good | 80 – 89 |
   | Average | 70 – 79 |
   | Needs Improvement | < 70 |
8. **Sorted** the data by `Average_Score` in descending order and reset the index.
9. Final cleaned dataset saved as **`03_cleaned_student_performance.csv`**.

---

## 📊 Visualizations Created

All charts generated via `04_data_visualization.py`, saved inside `05_charts/`:

| Chart | File | Purpose |
|---|---|---|
| Line Chart | `01_line_chart.png` | Score trend across students/subjects |
| Bar Chart | `02_bar_avg_score.png` | Average score per student |
| Histogram | `03_histogram_avg_score.png` | Distribution of average scores |
| Scatter Plot | `04_scatter_python_ml.png` | Correlation between Python and Machine Learning marks |
| Pie Chart | `05_pie_performance.png` | Share of students in each performance category |
| Box Plot | `06_box_subjects.png` | Score spread & outliers across all subjects |

---

## 💡 Key Insights

1. **Overall performance is strong, but a fifth of the class needs support.**
   Out of 20 students, 4 (20%) are **Excellent**, 6 (30%) are **Good**, 6 (30%) are **Average**, and 4 (20%) fall into **Needs Improvement**. This means half the class (50%) is performing at a "Good" level or above, while 1 in 5 students requires targeted academic support.

2. **Machine Learning has the highest class average; Python has the lowest.**
   Subject-wise averages came out as: **Machine_Learning – 82.6** (highest), Statistics – 80.6, Mathematics – 79.5, and **Python – 78.9** (lowest). Since ML performance typically builds on programming fundamentals, the relatively lower Python average suggests foundational coding skills may need reinforcement even though applied ML scores are strong.

3. **Python and Machine Learning scores are strongly positively correlated.**
   The scatter plot shows top performers (e.g. Laiba Khan: 97 Python / 99 ML, Ayesha Malik: 95/97) score high in both subjects, while lower performers (e.g. Hassan Tariq: 55/62) score low in both — confirming that strong programming ability directly supports Machine Learning performance.

---

## 🖥️ Mini Project – Student Performance Dashboard

Built using `06_dashboard_app.py` (Streamlit) and `07_gradio_app.py` (Gradio), the dashboard displays:

- **Total Students:** 20
- **Average Score per Subject:**

  | Subject | Average Marks |
  |---|---|
  | Machine_Learning | 82.6 |
  | Statistics | 80.6 |
  | Mathematics | 79.5 |
  | Python | 78.9 |

- **Top 5 Students:**

  | Rank | Name | Program | Average Score | Performance |
  |---|---|---|---|---|
  | 1 | Laiba Khan | SE | 97.25 | Excellent |
  | 2 | Ayesha Malik | SE | 95.5 | Excellent |
  | 3 | Noor Fatima | SE | 92.5 | Excellent |
  | 4 | Ahmed Raza | SE | 90.5 | Excellent |
  | 5 | Hira Shah | DS | 89.5 | Good |

- **Students Needing Improvement (4 total):** Abdullah (69.0), Fatima Noor (68.75), Danish Ali (64.0), Hassan Tariq (58.75)
- **Subject with Highest Average:** Machine_Learning (82.6)
- The dashboard also includes a **Program filter** (SE / DS / AI) in the sidebar, allowing metrics and charts to be explored per program
- Full visual breakdown of all charts listed above (line, bar, histogram, pie, scatter, box)

---

## 🌐 Deployment (Ngrok)

The app was tunneled locally using ngrok for evaluation access:

- **Gradio App:** `10_ngrok_gradio.py` → `"https://margin-twisted-kindly.ngrok-free.dev"` forwards to `"http://localhost:7860"`
- **Streamlit App:** `09_ngrok_streamlit.py` → `"https://margin-twisted-kindly.ngrok-free.dev"` forwards to `"http://localhost:8501"`

> Note: Since this is a free-tier ngrok tunnel, the link is only live while the local server + ngrok process are running. A screen recording (`recording.mp4`) is included in this folder as permanent proof of the working application and all deliverables.

## 🎥 Screen Recording

A short screen recording (`recording.mp4`) is included in the `Day-7` folder, demonstrating:
- The data cleaning script running successfully
- The dashboard app (Streamlit/Gradio) running locally
- The live ngrok URL working in the browser
- Key charts and dashboard metrics in action

---

## 📂 Folder Structure

```
Day-7/
├── 01_student_performance.csv          # Raw dataset
├── 02_data_cleaning.py                 # Data cleaning script
├── 03_cleaned_student_performance.csv  # Cleaned dataset
├── 04_data_visualization.py            # Visualization script
├── 05_charts/                          # All generated charts (PNG)
├── 06_dashboard_app.py                 # Streamlit dashboard
├── 07_gradio_app.py                    # Gradio app
├── 08_README.md                        # This file
├── 09_ngrok_streamlit.py               # Ngrok tunnel for Streamlit
├── 10_ngrok_gradio.py                  # Ngrok tunnel for Gradio
└── recording.mp4                       # Screen recording demo
```

---

## ✅ Outcome

By completing this task, the following skills were demonstrated:
- Cleaning and preprocessing real-world datasets using Pandas
- Feature engineering (Average_Score, Performance categories)
- Data visualization using Matplotlib & Seaborn
- Extracting and presenting actionable insights
- Deploying a local app publicly using Ngrok for remote evaluation

---
