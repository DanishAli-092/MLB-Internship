"""
Day 11 - Script 2: K-Means Clustering on the Iris Dataset
------------------------------------------------------------
Goal: group the 150 flowers into clusters based on their 4 measurements,
without telling the algorithm which species they actually belong to.
We use the Elbow Method to pick a sensible number of clusters (K)
instead of just guessing.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import style_config  #(applies the shared theme on import)

OUTPUT_DIR = "../outputs"
RANDOM_STATE = 42  # fixed so results are reproducible every run


def load_features():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["species"] = iris.target_names[iris.target]
    return df


def scale_features(df):
    """K-Means uses Euclidean distance, so features on bigger scales
    (like petal length in cm) would dominate features on smaller scales
    if we don't standardize first."""
    features = df.drop(columns="species")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    return scaled, scaler


def find_optimal_k(scaled_data, k_range=range(1, 11)):
    """Runs K-Means for each K and records the inertia (WCSS) so we can
    plot the elbow curve and pick K visually."""
    wcss = []
    for k in k_range:
        model = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=RANDOM_STATE)
        model.fit(scaled_data)
        wcss.append(model.inertia_)
    return list(k_range), wcss


def plot_elbow(k_values, wcss, chosen_k=3):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(k_values, wcss, marker="o", markersize=7, linewidth=2.2,
            color=style_config.PALETTE[0], markerfacecolor="white",
            markeredgewidth=2, markeredgecolor=style_config.PALETTE[0])
    ax.axvline(chosen_k, color=style_config.PALETTE[3], linestyle="--", linewidth=1.5, alpha=0.8)
    ax.annotate(
        f"chosen K = {chosen_k}",
        xy=(chosen_k, wcss[chosen_k - 1]),
        xytext=(chosen_k + 1.1, wcss[chosen_k - 1] + (max(wcss) - min(wcss)) * 0.12),
        fontsize=10, color=style_config.PALETTE[3], fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=style_config.PALETTE[3], lw=1.3),
    )
    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("WCSS (Inertia)")
    ax.set_xticks(list(k_values))
    style_config.style_axes_spines(ax)
    style_config.add_title(ax, "Elbow Method for Optimal K", "Where WCSS stops dropping sharply is our best K")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/03_elbow_method.png", dpi=150)
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/03_elbow_method.png")


def run_kmeans(scaled_data, k):
    model = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=RANDOM_STATE)
    labels = model.fit_predict(scaled_data)
    return model, labels


def plot_clusters(df, labels, feature_x, feature_y):
    """Scatter plot of clusters using two of the original (unscaled) features,
    just so the axes stay in real-world units (cm)."""
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    cluster_colors = [style_config.PALETTE[i % len(style_config.PALETTE)] for i in sorted(set(labels))]

    for cluster_id, color in zip(sorted(set(labels)), cluster_colors):
        mask = labels == cluster_id
        ax.scatter(
            df.loc[mask, feature_x], df.loc[mask, feature_y],
            s=65, alpha=0.8, color=color, edgecolor="white", linewidth=0.6,
            label=f"Cluster {cluster_id}",
        )

    ax.set_xlabel(feature_x)
    ax.set_ylabel(feature_y)
    style_config.style_axes_spines(ax)
    style_config.add_title(ax, "K-Means Clusters", f"{feature_x}  vs.  {feature_y}")
    ax.legend(title="Cluster", loc="best")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/04_kmeans_clusters.png", dpi=150)
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/04_kmeans_clusters.png")


def compare_clusters_to_species(df, labels):
    """Since we actually have the true species labels for Iris, we can
    sanity-check how well the unsupervised clusters line up with reality.
    This is only possible because it's a teaching dataset - in real
    unsupervised problems you usually don't have ground truth."""
    comparison = pd.crosstab(df["species"], labels, rownames=["Actual Species"], colnames=["Cluster"])
    print("\nCluster vs Actual Species crosstab:")
    print(comparison)
    return comparison


if __name__ == "__main__":
    iris_df = load_features()
    scaled_X, _ = scale_features(iris_df)

    k_values, wcss_values = find_optimal_k(scaled_X)
    plot_elbow(k_values, wcss_values)

    # From the elbow plot, K=3 is the natural choice for Iris
    # (matches the fact there really are 3 species).
    chosen_k = 3
    kmeans_model, cluster_labels = run_kmeans(scaled_X, chosen_k)

    iris_df["cluster"] = cluster_labels
    plot_clusters(iris_df, cluster_labels, "petal length (cm)", "petal width (cm)")
    compare_clusters_to_species(iris_df, cluster_labels)

    iris_df.to_csv("../data/iris_with_clusters.csv", index=False)
    print("\nSaved clustered dataset to ../data/iris_with_clusters.csv")
