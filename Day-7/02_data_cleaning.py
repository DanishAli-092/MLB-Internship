import pandas as pd

# Load the dataset
df = pd.read_csv("01_student_performance.csv")

print("Original shape:", df.shape)

# 1. Check for missing values
print("\nMissing values in each column:")

print(df.isnull().sum())

# Fill missing values if any (numeric columns with mean, text with "Unknown")

for col in df.columns:
    
    if df[col].isnull().sum() > 0:
        
        if df[col].dtype == "object":
            
            df[col] = df[col].fillna("Unknown")
            
        else:
            df[col] = df[col].fillna(df[col].mean())

# 2. Remove duplicate rows
before = df.shape[0]

df = df.drop_duplicates()

print(f"\nDuplicates removed: {before - df.shape[0]}")

# 3. Rename columns (just cleaning up spaces/casing if any)

df.columns = [col.strip() for col in df.columns]

# 4. Fix data types

df["Age"] = df["Age"].astype(int)

score_cols = ["Python", "Mathematics", "Statistics", "Machine_Learning"]

for col in score_cols:
    
    df[col] = df[col].astype(float)
    
df["Attendance"] = df["Attendance"].astype(float)

# 5. Create Average_Score column

df["Average_Score"] = df[score_cols].mean(axis=1).round(2)

# 6. Create Performance column based on Average_Score

def get_performance(score):
    
    if score >= 90:
        
        return "Excellent"
    
    elif score >= 80:
        
        return "Good"
    
    elif score >= 70:
        
        return "Average"
    else:
        
        return "Needs Improvement"

df["Performance"] = df["Average_Score"].apply(get_performance)

# 7. Sort by Average_Score (highest first)
df = df.sort_values(by="Average_Score", ascending=False).reset_index(drop=True)

print("\nPerformance breakdown:")

print(df["Performance"].value_counts())

print("\nTop 5 students:")

print(df[["Name", "Average_Score", "Performance"]].head())


# Save cleaned file
df.to_csv("03_cleaned_student_performance.csv", index=False)

print("\nSaved cleaned data to 03_cleaned_student_performance.csv")