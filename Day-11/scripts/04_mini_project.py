"""
Day 11 - Mini Project: Iris Flower Clustering & Visualization
------------------------------------------------------------------
Pulls together everything from the earlier scripts into one flow:

  1. Load + explore the Iris dataset
  2. Cluster it with K-Means (K chosen via Elbow Method)
  3. Reduce it to 2D with PCA
  4. Compare: original features vs K-Means clusters vs PCA projection,
     all in one figure so it's easy to see how they relate.

Run this file directly: python 04_mini_project.py
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import style_config  # (applies the shared theme on import)

OUTPUT_DIR = "../outputs"
RANDOM_STATE = 42


def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["species"] = iris.target_names[iris.target]
    return df


def get_optimal_k(scaled_data, max_k=10):
    """Same elbow logic as script 2 - kept here too so this file can run
    standalone as a full mini project without depending on the others."""
    wcss = []
    for k in range(1, max_k + 1):
        model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        model.fit(scaled_data)
        wcss.append(model.inertia_)
    return wcss


def _scatter_by_group(ax, x, y, group_codes, group_names=None):
    """Helper: scatters x vs y, coloring each unique group with the shared palette
    and giving every panel the same clean, consistent look."""
    unique_groups = sorted(set(group_codes))
    for i, g in enumerate(unique_groups):
        mask = group_codes == g
        label = group_names[g] if group_names is not None else f"Cluster {g}"
        ax.scatter(
            x[mask], y[mask],
            color=style_config.PALETTE[i % len(style_config.PALETTE)],
            s=55, alpha=0.8, edgecolor="white", linewidth=0.6, label=label,
        )
    style_config.style_axes_spines(ax)


def build_comparison_figure(df, scaled_data, cluster_labels, pca_2d, chosen_k, wcss):
    """One figure, 4 panels: elbow curve, raw features, K-Means clusters, PCA projection."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10.5))
    fig.suptitle("Iris Clustering & PCA — Mini Project Summary", fontsize=16, fontweight="bold", y=0.985)

    # Panel 1: Elbow curve
    ax = axes[0, 0]
    ax.plot(range(1, len(wcss) + 1), wcss, marker="o", markersize=6, linewidth=2.2,
            color=style_config.PALETTE[0], markerfacecolor="white", markeredgewidth=1.8,
            markeredgecolor=style_config.PALETTE[0])
    ax.axvline(chosen_k, color=style_config.PALETTE[3], linestyle="--", linewidth=1.5, label=f"chosen K = {chosen_k}")
    ax.set_title("1. Elbow Method", fontweight="bold")
    ax.set_xlabel("K")
    ax.set_ylabel("WCSS")
    ax.legend()
    style_config.style_axes_spines(ax)

    # Panel 2: Original data using two real features, colored by TRUE species
    species_codes = df["species"].astype("category").cat.codes
    species_names = df["species"].astype("category").cat.categories
    _scatter_by_group(axes[0, 1], df["petal length (cm)"], df["petal width (cm)"], species_codes, species_names)
    axes[0, 1].set_title("2. Original Data (true species)", fontweight="bold")
    axes[0, 1].set_xlabel("petal length (cm)")
    axes[0, 1].set_ylabel("petal width (cm)")
    axes[0, 1].legend(fontsize=8.5)

    # Panel 3: Same features, colored by K-Means cluster (no labels were used to get these)
    _scatter_by_group(axes[1, 0], df["petal length (cm)"], df["petal width (cm)"], cluster_labels)
    axes[1, 0].set_title(f"3. K-Means Clusters (K={chosen_k})", fontweight="bold")
    axes[1, 0].set_xlabel("petal length (cm)")
    axes[1, 0].set_ylabel("petal width (cm)")
    axes[1, 0].legend(fontsize=8.5)

    # Panel 4: PCA projection, colored by K-Means cluster
    _scatter_by_group(axes[1, 1], pca_2d[:, 0], pca_2d[:, 1], cluster_labels)
    axes[1, 1].set_title("4. PCA Projection (colored by cluster)", fontweight="bold")
    axes[1, 1].set_xlabel("Principal Component 1")
    axes[1, 1].set_ylabel("Principal Component 2")
    axes[1, 1].legend(fontsize=8.5)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(f"{OUTPUT_DIR}/07_mini_project_summary.png", dpi=150)
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/07_mini_project_summary.png")


def print_observations(df, cluster_labels, pca_model):
    crosstab = pd.crosstab(df["species"], cluster_labels, rownames=["Species"], colnames=["Cluster"])
    accuracy_like = crosstab.max(axis=1).sum() / len(df) * 100

    print("\n" + "=" * 60)
    print("OBSERVATIONS")
    print("=" * 60)
    print(f"\nCluster vs Species crosstab:\n{crosstab}")
    print(f"\n~{accuracy_like:.1f}% of points fall into the cluster that best matches their true species.")
    print(f"PCA retained {pca_model.explained_variance_ratio_.sum()*100:.1f}% of the original variance in just 2 dimensions.")
    print("\nKey takeaway: setosa separates cleanly into its own cluster in both the")
    print("raw feature plot and the PCA plot. versicolor and virginica overlap a bit")
    print("more - which matches biology, since those two species are more similar")
    print("to each other than either is to setosa.")


if __name__ == "__main__":
    iris_df = load_data()
    features = iris_df.drop(columns="species")

    scaler = StandardScaler()
    scaled_X = scaler.fit_transform(features)

    wcss_values = get_optimal_k(scaled_X)
    K = 3  # confirmed by elbow curve, and matches the known 3 species

    kmeans = KMeans(n_clusters=K, n_init=10, random_state=RANDOM_STATE)
    clusters = kmeans.fit_predict(scaled_X)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pca_coords = pca.fit_transform(scaled_X)

    build_comparison_figure(iris_df, scaled_X, clusters, pca_coords, K, wcss_values)
    print_observations(iris_df, clusters, pca)
