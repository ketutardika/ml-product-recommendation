import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import joblib
import os


class ContentBasedRecommender:
    """Content-Based Filtering using TF-IDF on ProductName + Price."""

    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        self.scaler = MinMaxScaler()
        self.similarity_matrix = None
        self.product_df = None
        self.product_indices = None

    def fit(self, product_df: pd.DataFrame):
        """Build feature matrix and compute cosine similarity."""
        self.product_df = product_df.reset_index(drop=True)

        # Index mapping: ProductName -> index
        self.product_indices = pd.Series(
            self.product_df.index, index=self.product_df["ProductName"]
        )

        print("[Model] Building TF-IDF matrix from ProductName...")
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(
            self.product_df["ProductName"]
        )
        print(f"[Model] TF-IDF matrix shape: {tfidf_matrix.shape}")

        # Normalize price feature
        price_scaled = self.scaler.fit_transform(
            self.product_df[["AvgPrice"]].values
        )

        # Combine TF-IDF + Price (weight 0.3 for price)
        price_sparse = sparse.csr_matrix(price_scaled * 0.3)
        feature_matrix = sparse.hstack([tfidf_matrix, price_sparse])

        print("[Model] Computing Cosine Similarity matrix...")
        self.similarity_matrix = cosine_similarity(feature_matrix, feature_matrix)
        print(f"[Model] Similarity matrix shape: {self.similarity_matrix.shape}")

        return self

    def get_recommendations(self, product_name: str, top_n: int = 10) -> pd.DataFrame:
        """Get top-N similar products for a given product name."""
        product_name = product_name.strip().lower()

        if product_name not in self.product_indices.index:
            # Try partial match
            matches = self.product_df[
                self.product_df["ProductName"].str.contains(product_name, na=False)
            ]
            if matches.empty:
                print(f"Product '{product_name}' not found.")
                return pd.DataFrame()
            # Use the first match
            idx = matches.index[0]
            product_name = self.product_df.loc[idx, "ProductName"]
            print(f"[Model] Using partial match: '{product_name}'")
        else:
            idx = self.product_indices[product_name]
            # If duplicate names exist, take the first one
            if isinstance(idx, pd.Series):
                idx = idx.iloc[0]

        # Get similarity scores
        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get top_n+1 (skip the product itself at position 0)
        sim_scores = sim_scores[1: top_n + 1]

        product_indices = [i[0] for i in sim_scores]
        scores = [i[1] for i in sim_scores]

        recommendations = self.product_df.iloc[product_indices].copy()
        recommendations["SimilarityScore"] = scores

        return recommendations[
            ["ProductNo", "ProductName", "AvgPrice", "SimilarityScore"]
        ]

    def get_recommendations_by_id(self, product_no: str, top_n: int = 10) -> pd.DataFrame:
        """Get top-N similar products for a given ProductNo."""
        match = self.product_df[self.product_df["ProductNo"] == product_no]
        if match.empty:
            print(f"ProductNo '{product_no}' not found.")
            return pd.DataFrame()

        idx = match.index[0]
        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1: top_n + 1]

        product_indices = [i[0] for i in sim_scores]
        scores = [i[1] for i in sim_scores]

        recommendations = self.product_df.iloc[product_indices].copy()
        recommendations["SimilarityScore"] = scores

        return recommendations[
            ["ProductNo", "ProductName", "AvgPrice", "SimilarityScore"]
        ]

    def save_model(self, model_dir: str = "models"):
        """Save similarity matrix and model artifacts."""
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(self.similarity_matrix, os.path.join(model_dir, "similarity_matrix.pkl"))
        joblib.dump(self.tfidf_vectorizer, os.path.join(model_dir, "tfidf_vectorizer.pkl"))
        joblib.dump(self.scaler, os.path.join(model_dir, "price_scaler.pkl"))
        self.product_df.to_csv(os.path.join(model_dir, "product_df.csv"), index=False)
        print(f"[Model] Model saved to '{model_dir}/'")
