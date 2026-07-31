"""
Day 11 - Interactive App: Clustering & Dimensionality Reduction Explorer
==========================================================================
Lets a user upload their own dataset (or use the built-in Iris dataset),
then:
  - Shows the Elbow Method curve with a live marker on the currently
    selected K, so the user can see exactly where their choice sits
    before committing to it
  - Runs K-Means clustering on demand (Apply button) with Silhouette
    Score and centroids plotted alongside the clusters
  - Reduces the data to 2D/3D with PCA on demand (Apply button), and
    shows which original features drive each principal component
  - Reports the data's shape before and after PCA
  - Lets the user download the clustered results as a CSV

Run locally with:  streamlit run app.py
"""

import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="DataLens", page_icon="👁️", layout="wide")

# A diverging pair (pink -> neutral -> blue) used for anything correlation-like,
# kept consistent with the palette used across the other Day 11 graphs.
DIVERGING_SCALE = [[0, "#DB2777"], [0.5, "#F5F5F5"], [1, "#2563EB"]]
CATEGORICAL_SCALE = ["#2563EB", "#F59E0B", "#059669", "#DB2777", "#7C3AED"]


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

@st.cache_data
def get_iris_dataframe():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["species"] = iris.target_names[iris.target]
    return df


def load_uploaded_file(uploaded_file):
    """Reads a CSV or Excel file the user uploads. Returns None on failure
    instead of crashing the whole app."""
    try:
        if uploaded_file.name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith((".xls", ".xlsx")):
            return pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file type. Please upload a .csv or .xlsx file.")
            return None
    except Exception as e:
        st.error(f"Could not read the file: {e}")
        return None


# ---------------------------------------------------------------------------
# Core ML logic
# ---------------------------------------------------------------------------

def get_numeric_columns(df):
    return df.select_dtypes(include=[np.number]).columns.tolist()


def compute_elbow_curve(scaled_data, max_k):
    wcss = []
    k_options = list(range(1, max_k + 1))
    for k in k_options:
        model = KMeans(n_clusters=k, n_init=10, random_state=42)
        model.fit(scaled_data)
        wcss.append(model.inertia_)
    return k_options, wcss


def run_kmeans(scaled_data, k):
    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = model.fit_predict(scaled_data)
    return labels, model


def run_pca(scaled_data, n_components):
    pca = PCA(n_components=n_components, random_state=42)
    transformed = pca.fit_transform(scaled_data)
    return transformed, pca


def get_cluster_quality(scaled_data, labels):
    """Silhouette Score ranges from -1 to 1 and measures how well-separated
    the clusters are: values near 1 mean points sit comfortably inside their
    own cluster and far from neighboring ones, values near 0 mean clusters
    overlap, and negative values mean points were probably assigned to the
    wrong cluster. Needs at least 2 clusters and fewer clusters than samples,
    so we guard against the edge case instead of letting it crash the app."""
    n_labels = len(set(labels))
    if 1 < n_labels < len(scaled_data):
        return silhouette_score(scaled_data, labels)
    return None


def build_scatter_with_centroids(plot_df, x, y, z, color_col, centroids_df, dims):
    """Builds a 2D or 3D scatter of the data points plus black 'X' markers
    for the K-Means centroids, so the viewer can see exactly what each
    cluster is centered on - not just the color grouping. Pass
    centroids_df=None to skip drawing centroids (e.g. before K-Means has
    been applied)."""
    categories = sorted(plot_df[color_col].astype(str).unique())
    fig = go.Figure()

    for i, category in enumerate(categories):
        subset = plot_df[plot_df[color_col].astype(str) == category]
        color = CATEGORICAL_SCALE[i % len(CATEGORICAL_SCALE)]
        marker_kwargs = dict(size=6 if dims == 3 else 9, color=color, opacity=0.8,
                              line=dict(width=0.5, color="white"))
        if dims == 3:
            fig.add_trace(go.Scatter3d(x=subset[x], y=subset[y], z=subset[z],
                                        mode="markers", name=str(category), marker=marker_kwargs))
        else:
            fig.add_trace(go.Scatter(x=subset[x], y=subset[y],
                                      mode="markers", name=str(category), marker=marker_kwargs))

    if centroids_df is not None:
        centroid_kwargs = dict(size=12 if dims == 3 else 15, color="black", symbol="x",
                                line=dict(width=2, color="black"))
        if dims == 3:
            fig.add_trace(go.Scatter3d(x=centroids_df[x], y=centroids_df[y], z=centroids_df[z],
                                        mode="markers", name="Centroid", marker=centroid_kwargs))
        else:
            fig.add_trace(go.Scatter(x=centroids_df[x], y=centroids_df[y],
                                      mode="markers", name="Centroid", marker=centroid_kwargs))

    if dims == 3:
        fig.update_layout(scene=dict(xaxis_title=x, yaxis_title=y, zaxis_title=z), height=550)
    else:
        fig.update_layout(xaxis_title=x, yaxis_title=y, height=500)
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
    return fig


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------

st.title("👁️ DataLens: Clustering & PCA Explorer")
st.caption("Day 11 · Unsupervised Learning · K-Means + PCA — upload your own dataset or explore Iris")

with st.sidebar:
    st.header("1. Choose your data")
    data_source = st.radio("Data source", ["Use Iris (built-in)", "Upload my own CSV/Excel"])

    if data_source == "Upload my own CSV/Excel":
        uploaded = st.file_uploader("Upload a file", type=["csv", "xlsx", "xls"])
        raw_df = load_uploaded_file(uploaded) if uploaded is not None else None
    else:
        raw_df = get_iris_dataframe()

if raw_df is None:
    st.info("Upload a CSV or Excel file from the sidebar to get started, or switch to the Iris dataset.")
    st.stop()

numeric_cols = get_numeric_columns(raw_df)

if len(numeric_cols) < 2:
    st.error("This dataset needs at least 2 numeric columns for clustering/PCA to make sense. "
              "Please upload a different file.")
    st.stop()

with st.sidebar:
    st.header("2. Select features")
    selected_features = st.multiselect(
        "Numeric columns to use",
        options=numeric_cols,
        default=numeric_cols,
        help="Non-numeric columns (like species/labels) are ignored for clustering but can still color the plots."
    )

    non_numeric_cols = [c for c in raw_df.columns if c not in numeric_cols]
    color_by_option = st.selectbox(
        "Optional: color points by a label column",
        options=["(none)"] + non_numeric_cols,
    ) if non_numeric_cols else "(none)"

if len(selected_features) < 2:
    st.warning("Select at least 2 numeric features from the sidebar.")
    st.stop()

feature_df = raw_df[selected_features].dropna()

if feature_df.shape[0] < 3:
    st.error("Not enough clean (non-missing) rows left after selecting these features to run clustering.")
    st.stop()

scaler = StandardScaler()
scaled_features = scaler.fit_transform(feature_df)
feature_selection_id = (tuple(selected_features), feature_df.shape[0])  # used to detect stale results below

# The K slider lives in the sidebar (not the main area) specifically so its
# value is already known by the time we draw the elbow chart in the main
# area just below - that's what lets the chart mark the chosen K live.
with st.sidebar:
    st.header("3. Clustering settings")
    max_k = min(10, feature_df.shape[0] - 1) if feature_df.shape[0] > 3 else 3
    chosen_k = st.slider("Number of clusters (K)", min_value=2, max_value=max_k, value=min(3, max_k))

# --- Dataset overview -------------------------------------------------------
st.subheader("📋 Dataset Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Rows", raw_df.shape[0])
col2.metric("Original Dimensions (features used)", len(selected_features))
col3.metric("Missing rows dropped", raw_df.shape[0] - feature_df.shape[0])

with st.expander("Preview data"):
    st.dataframe(raw_df.head(10))

# --- Elbow method (live marker on the currently selected K) -----------------
st.subheader("📈 Elbow Method — choosing K")
k_options, wcss_values = compute_elbow_curve(scaled_features, max_k)

elbow_fig = go.Figure()
elbow_fig.add_trace(go.Scatter(
    x=k_options, y=wcss_values, mode="lines+markers",
    line=dict(color=CATEGORICAL_SCALE[0], width=2.5),
    marker=dict(size=8), name="WCSS",
))
elbow_fig.add_vline(
    x=chosen_k, line_dash="dash", line_color=CATEGORICAL_SCALE[3],
    annotation_text=f"K = {chosen_k}", annotation_position="top",
)
elbow_fig.update_layout(
    title="WCSS vs Number of Clusters",
    xaxis_title="Number of clusters (K)", yaxis_title="WCSS (Inertia)",
    height=400, margin=dict(l=10, r=10, t=40, b=10),
)
st.plotly_chart(elbow_fig, width="stretch")
st.caption("The dashed line follows the K slider in the sidebar - drag it to see where your choice sits on the curve.")

apply_kmeans = st.button("▶ Apply K-Means", type="primary")

if apply_kmeans:
    labels, model = run_kmeans(scaled_features, chosen_k)
    st.session_state["kmeans_result"] = {
        "labels": labels,
        "model": model,
        "silhouette": get_cluster_quality(scaled_features, labels),
        "k": chosen_k,
        "features": feature_selection_id,
    }

kmeans_result = st.session_state.get("kmeans_result")
kmeans_is_stale = kmeans_result is not None and (
    kmeans_result["k"] != chosen_k or kmeans_result["features"] != feature_selection_id
)
if kmeans_is_stale:
    st.info("K or feature selection changed since the last run - click **Apply K-Means** again to refresh the results below.")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

explore_tab, cluster_tab, pca_tab, insights_tab = st.tabs(
    ["🔎 Data Exploration", "🧩 K-Means Clustering", "🧬 PCA", "💡 Insights & Download"]
)

with explore_tab:
    st.markdown("#### Feature correlations")
    st.caption("How strongly each pair of selected features moves together, before any clustering happens.")
    corr = feature_df.corr()
    corr_fig = px.imshow(
        corr, text_auto=".2f", aspect="auto",
        color_continuous_scale=DIVERGING_SCALE, zmin=-1, zmax=1,
    )
    corr_fig.update_layout(height=450, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(corr_fig, width="stretch")

    st.markdown("#### Feature distributions")
    dist_feature = st.selectbox("Feature to inspect", selected_features)
    hist_fig = px.histogram(
        raw_df, x=dist_feature, color=color_by_option if color_by_option != "(none)" else None,
        nbins=20, color_discrete_sequence=CATEGORICAL_SCALE,
    )
    hist_fig.update_layout(height=400, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(hist_fig, width="stretch")

with cluster_tab:
    if kmeans_result is None:
        st.info("Click **▶ Apply K-Means** above to run clustering and see the results here.")
    elif kmeans_is_stale:
        st.warning("Your data or feature selection changed since K-Means was last run - click **▶ Apply K-Means** again above to refresh the results below.")
    else:
        cluster_labels = kmeans_result["labels"]
        kmeans_model = kmeans_result["model"]
        silhouette = kmeans_result["silhouette"]

        result_df = feature_df.copy()
        result_df["cluster"] = cluster_labels.astype(str)
        if color_by_option != "(none)":
            result_df[color_by_option] = raw_df.loc[feature_df.index, color_by_option].values
        cluster_color = color_by_option if color_by_option != "(none)" else "cluster"

        # Centroids come out of KMeans in scaled space - inverse_transform puts
        # them back into the original feature units so they sit on the same
        # axes as the raw data points instead of floating in the wrong scale.
        centroids_original = pd.DataFrame(
            scaler.inverse_transform(kmeans_model.cluster_centers_), columns=selected_features,
        )

        quality_col1, quality_col2 = st.columns(2)
        quality_col1.metric("Silhouette Score", f"{silhouette:.3f}" if silhouette is not None else "N/A")
        quality_col2.metric("WCSS (Inertia)", f"{kmeans_model.inertia_:.1f}")
        st.caption(
            "Silhouette Score ranges from -1 to 1. Closer to 1 means clusters are tight and well "
            "separated; closer to 0 means they overlap."
        )

        st.markdown("#### Cluster visualization")
        if len(selected_features) >= 3:
            fig = build_scatter_with_centroids(
                result_df, selected_features[0], selected_features[1], selected_features[2],
                cluster_color, centroids_original, dims=3,
            )
        else:
            fig = build_scatter_with_centroids(
                result_df, selected_features[0], selected_features[1], None,
                cluster_color, centroids_original, dims=2,
            )
        st.plotly_chart(fig, width="stretch")
        st.caption("Black X markers show each cluster's centroid - the point KMeans considers the 'center' of that group.")

with pca_tab:
    max_pca_components = min(3, len(selected_features))
    apply_pca = st.button("▶ Apply PCA", type="primary")

    if apply_pca:
        pca_coords, pca_model = run_pca(scaled_features, max_pca_components)
        st.session_state["pca_result"] = {
            "coords": pca_coords,
            "model": pca_model,
            "features": feature_selection_id,
        }

    pca_result = st.session_state.get("pca_result")

    # Guard against a stale PCA model: if the feature selection changed since
    # PCA was last applied, pca_model was fit on a different number/set of
    # columns than the CURRENT kmeans_result's cluster centers. Transforming
    # one against the other raises a hard sklearn ValueError ("X has N
    # features, but PCA is expecting M features"), so we refuse to render
    # anything from the stale result instead of letting that crash happen -
    # the user just needs to click Apply PCA again.
    pca_is_stale = pca_result is not None and pca_result["features"] != feature_selection_id

    if pca_result is None:
        st.info("Click **▶ Apply PCA** above to reduce the data's dimensions and see the results here.")
    elif pca_is_stale:
        st.warning("Feature selection changed since PCA was last run - click **▶ Apply PCA** again to refresh the results below.")
    else:
        pca_model = pca_result["model"]
        pc_cols = [f"PC{i+1}" for i in range(pca_model.n_components_)]
        pca_df = pd.DataFrame(pca_result["coords"], columns=pc_cols)

        # Color PCA points by cluster if K-Means has already been applied on
        # the same features, otherwise fall back to the chosen label column
        # (or a single color if neither is available).
        if kmeans_result is not None and not kmeans_is_stale:
            pca_df["cluster"] = kmeans_result["labels"].astype(str)
            pca_color_col = "cluster"
            centroids_pca = pd.DataFrame(
                pca_model.transform(kmeans_result["model"].cluster_centers_), columns=pc_cols,
            )
        elif color_by_option != "(none)":
            pca_df[color_by_option] = raw_df.loc[feature_df.index, color_by_option].values
            pca_color_col = color_by_option
            centroids_pca = None
        else:
            pca_df["group"] = "all points"
            pca_color_col = "group"
            centroids_pca = None

        dim_col1, dim_col2, dim_col3 = st.columns(3)
        dim_col1.metric("Dimensions before PCA", feature_df.shape[1])
        dim_col2.metric("Dimensions after PCA", pca_model.n_components_)
        dim_col3.metric("Variance retained", f"{pca_model.explained_variance_ratio_.sum()*100:.1f}%")

        variance_fig = px.bar(
            x=pc_cols, y=pca_model.explained_variance_ratio_ * 100,
            labels={"x": "Principal Component", "y": "Explained Variance (%)"},
            title="Explained Variance per Principal Component",
            color_discrete_sequence=[CATEGORICAL_SCALE[1]],
        )
        variance_fig.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(variance_fig, width="stretch")

        st.markdown("#### PCA projection" + (" (with centroids)" if centroids_pca is not None else ""))
        if not (kmeans_result is not None and not kmeans_is_stale):
            st.caption("Run **Apply K-Means** on the K-Means Clustering tab to also see cluster centroids projected here.")
        if pca_model.n_components_ >= 3:
            pca_fig = build_scatter_with_centroids(
                pca_df, "PC1", "PC2", "PC3", pca_color_col, centroids_pca, dims=3,
            )
        else:
            pca_fig = build_scatter_with_centroids(
                pca_df, "PC1", "PC2", None, pca_color_col, centroids_pca, dims=2,
            )
        st.plotly_chart(pca_fig, width="stretch")

        st.markdown("#### Feature loadings")
        st.caption("How much each original feature contributes to each principal component. "
                   "Larger magnitude (either direction) means that feature drives that component more.")
        loadings = pd.DataFrame(
            pca_model.components_.T, columns=pc_cols, index=selected_features,
        )
        loadings_fig = px.imshow(
            loadings, text_auto=".2f", aspect="auto",
            color_continuous_scale=DIVERGING_SCALE, zmin=-1, zmax=1,
        )
        loadings_fig.update_layout(height=350, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(loadings_fig, width="stretch")

with insights_tab:
    if kmeans_result is None or pca_result is None:
        st.info("Apply K-Means and PCA on their respective tabs to unlock the full summary here.")
    elif kmeans_is_stale or pca_is_stale:
        st.warning(
            "Your data or feature selection changed since K-Means / PCA were last run - "
            "revisit the **K-Means Clustering** and **PCA** tabs and click Apply again to refresh this summary."
        )
    else:
        st.markdown("#### Summary")
        summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
        summary_col1.metric("Clusters found", kmeans_result["k"])
        summary_col2.metric("Silhouette Score", f"{kmeans_result['silhouette']:.3f}" if kmeans_result["silhouette"] is not None else "N/A")
        summary_col3.metric("Dimensions reduced", f"{feature_df.shape[1]} → {pca_result['model'].n_components_}")
        summary_col4.metric("Variance retained", f"{pca_result['model'].explained_variance_ratio_.sum()*100:.1f}%")

        result_df = feature_df.copy()
        result_df["cluster"] = kmeans_result["labels"].astype(str)
        if color_by_option != "(none)":
            result_df[color_by_option] = raw_df.loc[feature_df.index, color_by_option].values
            st.markdown("#### Cluster vs. label crosstab")
            st.caption(f"How well the unsupervised clusters line up with '{color_by_option}' (only meaningful if that column is a true category label).")
            crosstab = pd.crosstab(result_df[color_by_option], result_df["cluster"])
            st.dataframe(crosstab, width="stretch")

        st.markdown("#### 📥 Download results")
        csv_buffer = io.StringIO()
        result_df.to_csv(csv_buffer, index=False)
        st.download_button(
            "Download clustered data as CSV",
            data=csv_buffer.getvalue(),
            file_name="clustered_output.csv",
            mime="text/csv",
        )

st.markdown("---")
st.caption("Day 11 — MLB Internship · K-Means Clustering & PCA...\nDeveloped By Danish")