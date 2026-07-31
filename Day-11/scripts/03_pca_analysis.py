"""
Day 11 - Script 3: PCA on the Iris Dataset
----------------------------------------------
The Iris dataset has 4 features, which we can't plot directly in 2D.
PCA compresses those 4 features into 2 new "principal component" axes
that capture as much of the original variance as possible, so we can
actually see the structure of the data on a normal scatter plot.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import style_config  #(applies the shared theme on import)

OUTPUT_DIR = "../outputs"


def load_features():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["species"] = iris.target_names[iris.target]
    return df


def apply_pca(df, n_components=2):
    features = df.drop(columns="species")

    # Standardize first - PCA is sensitive to feature scale, same reasoning as K-Means
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    pca = PCA(n_components=n_components, random_state=42)
    transformed = pca.fit_transform(scaled)

    print(f"Original shape: {features.shape}")
    print(f"Shape after PCA: {transformed.shape}")
    print(f"Explained variance ratio per component: {pca.explained_variance_ratio_}")
    print(f"Total variance retained: {pca.explained_variance_ratio_.sum() * 100:.2f}%")

    return transformed, pca


def plot_pca_scatter(transformed, species_labels):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    species_names = species_labels.unique()

    for name, color in zip(species_names, style_config.PALETTE):
        mask = species_labels == name
        ax.scatter(
            transformed[mask, 0], transformed[mask, 1],
            label=name, color=color, s=65, alpha=0.8,
            edgecolor="white", linewidth=0.6,
        )

    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.axhline(0, color="#CCCCCC", linewidth=0.8, zorder=0)
    ax.axvline(0, color="#CCCCCC", linewidth=0.8, zorder=0)
    style_config.style_axes_spines(ax)
    style_config.add_title(ax, "Iris Data Projected onto 2 Principal Components", "Same 150 flowers, compressed from 4 dimensions to 2")
    ax.legend(title="Species")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/05_pca_scatter.png", dpi=150)
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/05_pca_scatter.png")


def plot_explained_variance(pca):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    components = [f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))]
    values = pca.explained_variance_ratio_ * 100
    bars = ax.bar(components, values, color=style_config.PALETTE[1], edgecolor="white", linewidth=1, width=0.55)

    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1.5, f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold", color="#333333")

    ax.set_ylabel("Explained Variance (%)")
    ax.set_ylim(0, max(values) * 1.25)
    style_config.style_axes_spines(ax)
    style_config.add_title(ax, "Variance Explained by Each Principal Component", f"Together they retain {values.sum():.1f}% of the original information")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/06_explained_variance.png", dpi=150)
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/06_explained_variance.png")


if __name__ == "__main__":
    iris_df = load_features()
    pca_result, pca_model = apply_pca(iris_df)
    plot_pca_scatter(pca_result, iris_df["species"])
    plot_explained_variance(pca_model)
