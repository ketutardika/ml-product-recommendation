import numpy as np
import pandas as pd


def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    """Precision@K: proportion of relevant items in the top-K recommendations."""
    recommended_k = recommended[:k]
    if len(recommended_k) == 0:
        return 0.0
    hits = len(set(recommended_k) & relevant)
    return hits / len(recommended_k)


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    """Recall@K: proportion of relevant items successfully recommended in top-K."""
    recommended_k = recommended[:k]
    if len(relevant) == 0:
        return 0.0
    hits = len(set(recommended_k) & relevant)
    return hits / len(relevant)


def f1_at_k(precision: float, recall: float) -> float:
    """F1-Score@K: harmonic mean of Precision and Recall."""
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    """NDCG@K: Normalized Discounted Cumulative Gain."""
    recommended_k = recommended[:k]
    dcg = 0.0
    for i, item in enumerate(recommended_k):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 2)  # position starts from 1, log2(1+1)

    # Ideal DCG: all relevant items placed at the top positions
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))

    if idcg == 0:
        return 0.0
    return dcg / idcg


def evaluate_model(model, customer_products: dict, k: int = 10,
                   sample_size: int = 500, random_state: int = 42) -> dict:
    """
    Evaluate Content-Based model using customer purchase history as ground truth.

    Logic:
    - For each customer, pick one product as the "query"
    - The remaining purchased products = ground truth (relevant items)
    - Generate recommendations from the model based on the query product
    - Compute metrics: Precision@K, Recall@K, F1@K, NDCG@K
    """
    np.random.seed(random_state)

    # Filter customers with >= 3 products (enough ground truth)
    eligible_customers = {
        cid: prods for cid, prods in customer_products.items()
        if len(prods) >= 3
    }
    print(f"[Evaluation] Eligible customers (>= 3 products): {len(eligible_customers)}")

    customer_ids = list(eligible_customers.keys())
    if len(customer_ids) > sample_size:
        customer_ids = list(np.random.choice(customer_ids, sample_size, replace=False))

    print(f"[Evaluation] Evaluating {len(customer_ids)} customers with K={k}...")

    precisions, recalls, f1s, ndcgs = [], [], [], []
    valid_count = 0

    product_no_set = set(model.product_df["ProductNo"].values)

    for cid in customer_ids:
        products = list(eligible_customers[cid])
        # Randomly pick one product as query
        query_product = np.random.choice(products)
        # The rest = ground truth
        relevant = set(products) - {query_product}

        # Filter relevant items that exist in the model
        relevant = relevant & product_no_set
        if len(relevant) == 0:
            continue

        # Generate recommendations
        recs = model.get_recommendations_by_id(query_product, top_n=k)
        if recs.empty:
            continue

        recommended = recs["ProductNo"].tolist()

        p = precision_at_k(recommended, relevant, k)
        r = recall_at_k(recommended, relevant, k)
        f1 = f1_at_k(p, r)
        n = ndcg_at_k(recommended, relevant, k)

        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        ndcgs.append(n)
        valid_count += 1

    results = {
        "Precision@K": np.mean(precisions) if precisions else 0.0,
        "Recall@K": np.mean(recalls) if recalls else 0.0,
        "F1-Score@K": np.mean(f1s) if f1s else 0.0,
        "NDCG@K": np.mean(ndcgs) if ndcgs else 0.0,
        "K": k,
        "Evaluated Customers": valid_count,
    }

    return results


def print_evaluation_results(results: dict):
    """Print evaluation results in a formatted table."""
    print("\n" + "=" * 55)
    print("     CONTENT-BASED FILTERING EVALUATION RESULTS")
    print("=" * 55)
    print(f"  K (Top-N)              : {results['K']}")
    print(f"  Evaluated Customers    : {results['Evaluated Customers']}")
    print("-" * 55)
    print(f"  Precision@{results['K']:<3}           : {results['Precision@K']:.4f}")
    print(f"  Recall@{results['K']:<3}              : {results['Recall@K']:.4f}")
    print(f"  F1-Score@{results['K']:<3}            : {results['F1-Score@K']:.4f}")
    print(f"  NDCG@{results['K']:<3}                : {results['NDCG@K']:.4f}")
    print("=" * 55)
