# Day 3 - Conditional Statements, Loops & Problem Solving

## Overview
Today I practiced conditional statements (if / elif / else, nested conditions, logical operators) and loops (for, while) to solve problem-solving and logic-building exercises. I also built a Number Analysis Tool and a bonus menu-driven application.

## Concepts Learned
- `if-elif-else` chains for handling multiple ranges of conditions (used in the grade calculator).
- Nested conditions for multi-step rules where a later check only matters if an earlier one passes (leap year).
- Logical operators (`and`, `or`) as a way to combine multiple conditions into a single expression instead of nesting.
- `for` loops with `range()` for a known number of repetitions, and `while` loops for repeating until a condition changes (like counting digits or building a number in reverse).
- The "accumulator pattern" using a variable that keeps updating inside a loop (used for sum of N numbers, digit count, and reversing a number).
- `while True` with `break` to build a menu-driven program that keeps running until the user chooses to exit.

## Problems Solved
**Conditional Statements**
- Positive / negative / zero check
- Even / odd check
- Grade calculator based on marks
- Largest among three numbers
- Leap year check (both a nested if-else version and a one-line version using logical operators)

**Loops**
- Print numbers 1 to 100
- Print even numbers 1 to 100
- Sum of numbers from 1 to N
- Multiplication table of a given number
- Count digits in a number

**Logic Building**
- Reverse a number
- Palindrome check
- Fibonacci sequence
- Prime number check
- All prime numbers between 1 and 100

**Mini Challenge**
- Number Analysis Tool — takes one number and reports even/odd, prime check, digit count, reversed number, and palindrome check together.

**Bonus**
- Menu-driven app with prime check, Fibonacci series, palindrome check, and multiplication table, running in a loop until the user exits.

## Challenges Faced
- In "largest of three numbers", using strict `>` missed the case where two numbers tie for the largest (e.g. 10, 10, 5) — it would print the wrong result. Switching to `>=` fixed it.
- The leap year rule was confusing at first because it has an exception (divisible by 100 → not leap) and then an exception to that exception (divisible by 400 → leap again). I worked through it step by step before writing any code.
- Initially forgot to convert `input()` values to `int` in one of the problems, which caused numbers to be compared as text instead of actual numbers, giving wrong results for two-digit vs one-digit comparisons.
- Reversing a number without converting it to a string took a bit of thinking  had to build the reversed number digit by digit using `% 10` and `// 10`.

## Lessons Learned
- Writing the logic out in plain English before coding makes nested conditions much easier to get right.
- Always test edge cases (ties, boundary years like 1900/2000, single-digit numbers) instead of just the obvious cases.
- Menu-driven loops (`while True` + `break`) make testing multiple problems a lot faster since the program doesn't need to be rerun each time.

## Project Structure
```
Day-3/
├── practice/
│   ├── conditionals.py
│   ├── loops.py
│   └── logic_building.py
├── number_analysis_tool.py
├── menu_app.py
├── README.md
└── video-recording/
    ├── practice_problems_demo.mp4
    └── menu_and_number_analysis_demo.mp4
```

## Video Recording
Program execution is recorded in the `video-recording/` folder  one video covers the practice problems (conditionals, loops, logic building), and the other covers the Number Analysis Tool and the bonus menu-driven app.