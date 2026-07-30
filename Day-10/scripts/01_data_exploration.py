"""
Day 10 - Step 1: Dataset Exploration

Loading the Breast Cancer Wisconsin dataset from sklearn, converting it
to a DataFrame, and exploring it before building any model.
"""

from sklearn.datasets import load_breast_cancer
import pandas as pd


def load_data():
    # sklearn gives us this dataset as a Bunch object, not a DataFrame,
    # so we need to convert it ourselves first.
    try:
        bunch = load_breast_cancer()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None

    df = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    df["target"] = bunch.target  # 0 = malignant, 1 = benign
    
    return df, bunch.target_names




def explore_data(df, target_names):
    print("First 5 rows:")
    print(df.head())

    print("\nDataset info:")
    df.info()

    print("\nStatistical summary:")
    print(df.describe())

    # checking how many samples belong to each class
    print("\nTarget class distribution:")
    counts = df["target"].value_counts().sort_index()
    print(counts)

    for class_value, count in counts.items():
        percent = (count / len(df)) * 100
        print(f"{target_names[class_value]}: {count} samples ({percent:.1f}%)")

    # if one class has way more samples than the other, accuracy alone
    # won't be a reliable metric later, so it's good to check this early
    ratio = counts.max() / counts.min()
    print(f"\nImbalance ratio: {ratio:.2f} : 1")


if __name__ == "__main__":
    df, target_names = load_data()

    if df is None:
        print("Could not load dataset, stopping.")
    else:
        pass
        #explore_data(df, target_names)

        try:
            df.to_csv("../outputs/breast_cancer_data.csv", index=False)
            print("\nDataset saved to outputs/breast_cancer_data.csv")
        except Exception as e:
            print(f"Could not save CSV: {e}")