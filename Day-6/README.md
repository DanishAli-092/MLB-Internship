# Day 6 — NumPy & Pandas

Day 6 was about moving from plain Python to the two libraries every AI/ML project starts with: NumPy for fast array math, and Pandas for working with tabular data. The day ends with a mini project — analyzing a class of 20 students' marks across four subjects.

## What's in this folder

```
Day-6/
├── 01_numpy_practice/
│   └── 01_numpy_basics.py          # NumPy exercises I solved
├── 02_pandas_practice/
│   └── 02_pandas_basics.py         # Pandas exercises I solved
├── 03_student_performance_project/
│   ├── 03_analysis.py              # the mini project
│   ├── students_original.csv       # the dataset (20 students)
│   └── students_processed.csv      # output: dataset + Overall_Average column
└── README.md                       # this file
```

Run order: `01_numpy_basics.py` and `02_pandas_basics.py` can run independently. `03_analysis.py` needs `students_original.csv` in the same folder — it reads it, does the analysis, and writes `students_processed.csv` back into the same folder.

## What I learned about NumPy

NumPy is a library for fast numerical computation using arrays. The main difference from a normal Python list is that a NumPy array holds elements of the same type stored together in memory, which is what makes it fast and lets it support math operations directly on the whole array instead of writing loops.

What I practiced:
- Creating 1D and 2D arrays with `np.array()`, and checking `.shape`, `.dtype`, `.ndim` on them
- Arithmetic on arrays — addition, subtraction, multiplication, division — happens element-wise, not like list concatenation
- `*` is element-wise multiplication, `@` is actual dot/matrix product — these are not the same thing and I mixed them up at first
- Finding `max()`, `min()`, `mean()`, `sum()` of an array, plus `argmax()`/`argmin()` to get the position of the max/min value, not just the value
- Reshaping with `.reshape(rows, cols)` — the total element count has to match, and using `-1` for one dimension lets NumPy calculate it automatically instead of me doing the math
- Indexing and slicing — `start:stop:step` for 1D, and for 2D it's `arr[row, col]`, where `arr[:, col]` grabs a whole column and `arr[row, :]` grabs a whole row
- Reverse slicing `arr[::-1]` — a negative step moves through the array backward instead of forward
- Boolean indexing, e.g. `arr[arr > 5]` — this pulls out only the elements matching a condition, no loop needed

The thing I'll remember: slicing returns a **view**, not a copy — if I change a slice, the original array changes too, unless I explicitly use `.copy()`.

## What I learned about Pandas

Pandas is basically NumPy with labels on top of it — that's what makes it useful for real datasets. There are two core objects:
- **Series** — one column of data with an index attached
- **DataFrame** — a full table, which is really just several Series sharing the same index

What I practiced:
- Loading data with `pd.read_csv()`
- The standard first-look routine: `.head()` / `.tail()` to see the data, `.info()` to check column types and non-null counts, `.describe()` to see the statistical summary, `.shape` and `.columns` for structure
- Checking for missing data with `.isnull().sum()`
- `df['col']` returns a Series, `df[['col']]` (double brackets) returns a DataFrame — small syntax difference, different result
- Filtering rows with conditions, e.g. `df[df['Python'] > 80]` — and combining conditions with `&` / `|`, not Python's normal `and`/`or`
- Calculating averages per column with `.mean()`
- Creating a new column from existing ones, e.g. `df[['Python','Mathematics','Statistics','Machine_Learning']].mean(axis=1)` to get each student's overall average
- Sorting with `.sort_values()` to pull out the top performers
- Saving the updated DataFrame back to a new CSV with `.to_csv(..., index=False)`

## Key insights from the dataset

- The class average across all four subjects is 80.40.
- Machine Learning had the highest subject average (82.6), Python had the lowest (78.9).
- Laiba Khan came out on top with an overall average of 97.25.
- 10 out of 20 students are scoring below the class average — a handful of high scorers are pulling the average up, so "below average" doesn't necessarily mean failing, it means below the group's center.
- This particular dataset had zero missing values, so no cleaning was actually needed — but I still ran `.isnull().sum()` since that's the first thing to check on any new dataset, clean or not.

## Challenges I faced

**1. Relative path errors when reading the CSV.**
I ran the script from my project's root folder in the terminal, but the CSV was inside a subfolder, and I got `FileNotFoundError`. The reason: a relative path like `'students_original.csv'` is resolved from wherever the terminal's current directory is, not from where the `.py` file itself sits. Fix: give the path relative to where I'm actually running the script from, e.g. `Day-6/03_student_performance_project/students_original.csv`.

**2. Naming a variable `max`, `min`, `sum`.**
While practicing NumPy stats, I named my variables `max`, `min`, `sum` — same names as Python's built-in functions. This overwrites the built-ins for the rest of the script, so if I tried to use the real `sum()` function later, it would break because `sum` was now just a number. Fix: renamed to `max_value`, `min_value`, `sum_value`.

**3. `reshape(-1, n)` didn't make sense at first.**
I understood `reshape(3, 4)` fine, but `-1` confused me. It's not a real dimension — it's telling NumPy "figure this dimension out yourself based on the total number of elements and the size I did give you."

**4. Reverse slicing `arr[::-1]`.**
Same confusion — I didn't get how leaving `start` and `stop` empty but setting `step = -1` reverses the array. The logic is that a negative step just means "walk through the indexes backward, from the end toward the start," instead of the normal forward direction.

## Requirements met

- NumPy: arrays, arithmetic, min/max/mean/sum, reshaping, indexing & slicing
- Pandas: Series/DataFrames, reading CSVs, exploring, selecting, filtering, statistics
- Mini project: loaded the dataset, showed basic info, averaged each subject
- Found the top 5 students and everyone scoring below the class average
- Displayed the total number of students
- Saved the processed dataset as `students_processed.csv`