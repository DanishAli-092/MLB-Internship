"""
Day 11 - Script 1: Iris Dataset Exploration
--------------------------------------------
Before applying any unsupervised learning technique, it's important to
actually look at the data first - shape, feature ranges, correlations,
and species distribution (even though we won't use the labels for
clustering, they're useful later to check how good our clusters are).
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

import style_config  # (applies the shared theme on import)

OUTPUT_DIR = "../outputs"
SPECIES_PALETTE = {
    "setosa": style_config.PALETTE[0],
    "versicolor": style_config.PALETTE[1],
    "virginica": style_config.PALETTE[2],
}


def load_iris_as_dataframe():
    """Loads the sklearn Iris dataset and returns it as a clean DataFrame."""
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)
    return df


def explore_dataset(df):
    print("=" * 60)
    print("IRIS DATASET - BASIC EXPLORATION")
    print("=" * 60)

    print(f"\nShape: {df.shape[0]} rows, {df.shape[1]} columns")

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values per column:")
    print(df.isnull().sum())

    print("\nStatistical summary:")
    print(df.describe())

    print("\nSpecies distribution:")
    print(df["species"].value_counts())

    return df


def plot_feature_distributions(df):
    """Pairplot to see how the 4 features relate to each other, colored by species.
    This is just for our own understanding - K-Means later won't see the species column."""
    grid = sns.pairplot(
        df,
        hue="species",
        diag_kind="kde",
        palette=SPECIES_PALETTE,
        plot_kws={"alpha": 0.75, "s": 45, "edgecolor": "white", "linewidth": 0.4},
        diag_kws={"fill": True, "alpha": 0.55, "linewidth": 1.2},
        height=2.1,
    )
    style_config.add_figure_title(
        grid.figure,
        "Iris Feature Relationships by Species",
        "Pairwise comparison of all 4 measurements",
        y=1.04,
    )
    for ax in grid.axes.flatten():
        if ax is not None:
            style_config.style_axes_spines(ax)
    grid.savefig(f"{OUTPUT_DIR}/01_feature_pairplot.png", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"\nSaved: {OUTPUT_DIR}/01_feature_pairplot.png")


def plot_correlation_heatmap(df):
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    corr = df.drop(columns="species").corr()
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap=style_config.HEATMAP_CMAP,
        vmin=-1, vmax=1,
        linewidths=0.8,
        linecolor="white",
        square=True,
        cbar_kws={"label": "Correlation", "shrink": 0.85},
        ax=ax,
    )
    style_config.add_title(ax, "Feature Correlation Heatmap", "How strongly each pair of measurements moves together")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/02_correlation_heatmap.png", dpi=150)
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/02_correlation_heatmap.png")


if __name__ == "__main__":
    iris_df = load_iris_as_dataframe()
    iris_df = explore_dataset(iris_df)
    plot_feature_distributions(iris_df)
    plot_correlation_heatmap(iris_df)

    # Save a cleaned copy so the other scripts don't have to reload from sklearn
    iris_df.to_csv("../data/iris_clean.csv", index=False)
    print("\nSaved cleaned dataset to ../data/iris_clean.csv")