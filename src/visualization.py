import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os


def setup_style():
    """Set up matplotlib style for consistent plots."""
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "figure.dpi": 100,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
    })


def plot_data_overview(df: pd.DataFrame, product_df: pd.DataFrame, save_dir: str = "output"):
    """Generate data overview visualizations."""
    os.makedirs(save_dir, exist_ok=True)
    setup_style()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Data Overview", fontsize=16, fontweight="bold")

    # 1. Top 15 best-selling products
    top_products = product_df.nlargest(15, "TotalQuantitySold")
    axes[0, 0].barh(
        top_products["ProductName"].str.title(), top_products["TotalQuantitySold"],
        color=sns.color_palette("viridis", 15)
    )
    axes[0, 0].set_title("Top 15 Best-Selling Products")
    axes[0, 0].set_xlabel("Total Quantity Sold")
    axes[0, 0].invert_yaxis()

    # 2. Price distribution
    axes[0, 1].hist(product_df["AvgPrice"], bins=50, color="#4C72B0", edgecolor="white", alpha=0.8)
    axes[0, 1].set_title("Product Price Distribution")
    axes[0, 1].set_xlabel("Average Price")
    axes[0, 1].set_ylabel("Number of Products")
    axes[0, 1].axvline(product_df["AvgPrice"].median(), color="red", linestyle="--",
                        label=f'Median: {product_df["AvgPrice"].median():.2f}')
    axes[0, 1].legend()

    # 3. Top 10 countries by transaction count
    country_counts = df["Country"].value_counts().head(10)
    axes[1, 0].bar(range(len(country_counts)), country_counts.values,
                   color=sns.color_palette("Set2", 10))
    axes[1, 0].set_title("Top 10 Countries by Transactions")
    axes[1, 0].set_xlabel("Country")
    axes[1, 0].set_ylabel("Number of Transactions")
    axes[1, 0].set_xticks(range(len(country_counts)))
    axes[1, 0].set_xticklabels(country_counts.index, rotation=45, ha="right")

    # 4. Products per customer distribution
    products_per_customer = df.groupby("CustomerNo")["ProductNo"].nunique()
    axes[1, 1].hist(products_per_customer, bins=50, color="#DD8452", edgecolor="white", alpha=0.8)
    axes[1, 1].set_title("Products per Customer Distribution")
    axes[1, 1].set_xlabel("Number of Unique Products")
    axes[1, 1].set_ylabel("Number of Customers")
    axes[1, 1].axvline(products_per_customer.median(), color="red", linestyle="--",
                        label=f"Median: {products_per_customer.median():.0f}")
    axes[1, 1].legend()

    plt.tight_layout()
    filepath = os.path.join(save_dir, "01_data_overview.png")
    plt.savefig(filepath, bbox_inches="tight")
    plt.close()
    print(f"[Visualization] Saved: {filepath}")


def plot_similarity_heatmap(model, save_dir: str = "output", sample_size: int = 30):
    """Plot a heatmap of cosine similarity for a sample of products."""
    os.makedirs(save_dir, exist_ok=True)
    setup_style()

    # Sample products for readable heatmap
    np.random.seed(42)
    n = min(sample_size, len(model.product_df))
    indices = np.random.choice(len(model.product_df), n, replace=False)
    indices = sorted(indices)

    sub_matrix = model.similarity_matrix[np.ix_(indices, indices)]
    labels = model.product_df.iloc[indices]["ProductName"].str.title().str[:30].values

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        sub_matrix, xticklabels=labels, yticklabels=labels,
        cmap="YlOrRd", vmin=0, vmax=1, annot=False,
        linewidths=0.5, ax=ax
    )
    ax.set_title(f"Cosine Similarity Heatmap (Sample of {n} Products)", fontsize=14, fontweight="bold")
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(fontsize=7)

    plt.tight_layout()
    filepath = os.path.join(save_dir, "02_similarity_heatmap.png")
    plt.savefig(filepath, bbox_inches="tight")
    plt.close()
    print(f"[Visualization] Saved: {filepath}")


def plot_similarity_distribution(model, save_dir: str = "output"):
    """Plot distribution of similarity scores."""
    os.makedirs(save_dir, exist_ok=True)
    setup_style()

    # Get upper triangle of similarity matrix (exclude diagonal)
    upper_tri = model.similarity_matrix[np.triu_indices_from(model.similarity_matrix, k=1)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Cosine Similarity Score Distribution", fontsize=14, fontweight="bold")

    # Histogram of all similarity scores
    axes[0].hist(upper_tri, bins=100, color="#4C72B0", edgecolor="white", alpha=0.8)
    axes[0].set_title("All Pairwise Similarity Scores")
    axes[0].set_xlabel("Cosine Similarity")
    axes[0].set_ylabel("Frequency")
    axes[0].axvline(np.mean(upper_tri), color="red", linestyle="--",
                     label=f"Mean: {np.mean(upper_tri):.4f}")
    axes[0].legend()

    # Histogram of non-zero similarity scores
    nonzero = upper_tri[upper_tri > 0]
    axes[1].hist(nonzero, bins=100, color="#55A868", edgecolor="white", alpha=0.8)
    axes[1].set_title("Non-Zero Similarity Scores")
    axes[1].set_xlabel("Cosine Similarity")
    axes[1].set_ylabel("Frequency")
    axes[1].axvline(np.mean(nonzero), color="red", linestyle="--",
                     label=f"Mean: {np.mean(nonzero):.4f}")
    axes[1].legend()

    plt.tight_layout()
    filepath = os.path.join(save_dir, "03_similarity_distribution.png")
    plt.savefig(filepath, bbox_inches="tight")
    plt.close()
    print(f"[Visualization] Saved: {filepath}")


def plot_recommendation_example(model, product_name: str, top_n: int = 10, save_dir: str = "output"):
    """Visualize recommendation results for a specific product."""
    os.makedirs(save_dir, exist_ok=True)
    setup_style()

    recs = model.get_recommendations(product_name, top_n=top_n)
    if recs.empty:
        print(f"[Visualization] No recommendations found for '{product_name}'")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Recommendations for: "{product_name.title()}"', fontsize=14, fontweight="bold")

    # Bar chart of similarity scores
    names = recs["ProductName"].str.title().str[:35].values
    scores = recs["SimilarityScore"].values

    colors = plt.cm.RdYlGn(scores / max(scores))
    axes[0].barh(range(len(names)), scores, color=colors)
    axes[0].set_yticks(range(len(names)))
    axes[0].set_yticklabels(names, fontsize=9)
    axes[0].set_xlabel("Cosine Similarity Score")
    axes[0].set_title("Similarity Scores")
    axes[0].invert_yaxis()
    for i, v in enumerate(scores):
        axes[0].text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)

    # Price comparison
    query_price = model.product_df[
        model.product_df["ProductName"] == product_name.strip().lower()
    ]["AvgPrice"].values
    query_price = query_price[0] if len(query_price) > 0 else 0

    prices = recs["AvgPrice"].values
    bar_colors = ["#4C72B0"] * len(prices)
    axes[1].barh(range(len(names)), prices, color=bar_colors)
    axes[1].set_yticks(range(len(names)))
    axes[1].set_yticklabels(names, fontsize=9)
    axes[1].set_xlabel("Average Price")
    axes[1].set_title("Price Comparison")
    axes[1].axvline(query_price, color="red", linestyle="--",
                     label=f"Query Price: {query_price:.2f}")
    axes[1].legend()
    axes[1].invert_yaxis()

    plt.tight_layout()
    safe_name = product_name.replace(" ", "_")[:30]
    filepath = os.path.join(save_dir, f"04_recommendations_{safe_name}.png")
    plt.savefig(filepath, bbox_inches="tight")
    plt.close()
    print(f"[Visualization] Saved: {filepath}")


def plot_evaluation_metrics(results: dict, save_dir: str = "output"):
    """Visualize evaluation metric results."""
    os.makedirs(save_dir, exist_ok=True)
    setup_style()

    metrics = ["Precision@K", "Recall@K", "F1-Score@K", "NDCG@K"]
    values = [results[m] for m in metrics]
    labels = [f"Precision@{results['K']}", f"Recall@{results['K']}",
              f"F1-Score@{results['K']}", f"NDCG@{results['K']}"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Evaluation Metrics (K={results['K']}, N={results['Evaluated Customers']} customers)",
                 fontsize=14, fontweight="bold")

    # Bar chart
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    bars = axes[0].bar(labels, values, color=colors, edgecolor="white", linewidth=1.5)
    axes[0].set_ylabel("Score")
    axes[0].set_title("Metric Comparison")
    axes[0].set_ylim(0, max(values) * 1.3)
    for bar, val in zip(bars, values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f"{val:.4f}", ha="center", va="bottom", fontweight="bold")

    # Radar chart
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    values_radar = values + [values[0]]
    angles += angles[:1]

    ax_radar = fig.add_subplot(122, polar=True)
    ax_radar.fill(angles, values_radar, color="#4C72B0", alpha=0.25)
    ax_radar.plot(angles, values_radar, color="#4C72B0", linewidth=2, marker="o", markersize=6)
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(labels, fontsize=10)
    ax_radar.set_title("Metric Radar Chart", pad=20)

    # Remove the duplicate axes[1] since we replaced it with polar
    axes[1].set_visible(False)

    plt.tight_layout()
    filepath = os.path.join(save_dir, "05_evaluation_metrics.png")
    plt.savefig(filepath, bbox_inches="tight")
    plt.close()
    print(f"[Visualization] Saved: {filepath}")


def generate_all_visualizations(df, product_df, model, results, save_dir="output"):
    """Generate all visualizations."""
    print("\n[Visualization] Generating all plots...")

    plot_data_overview(df, product_df, save_dir)
    plot_similarity_heatmap(model, save_dir)
    plot_similarity_distribution(model, save_dir)

    # Sample 3 products for recommendation visualization
    sample_products = product_df.sample(3, random_state=42)
    for _, row in sample_products.iterrows():
        plot_recommendation_example(model, row["ProductName"], top_n=10, save_dir=save_dir)

    plot_evaluation_metrics(results, save_dir)

    print(f"[Visualization] All plots saved to '{save_dir}/' folder.")
