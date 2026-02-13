import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src.data_preprocessing import preprocess_pipeline
from src.content_based_model import ContentBasedRecommender
from src.evaluation import evaluate_model, print_evaluation_results
from src.visualization import generate_all_visualizations


def main():
    DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "sales_transaction_v4a.csv")
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
    K = 10

    # === 1. Preprocessing ===
    print("=" * 55)
    print("  STEP 1: DATA PREPROCESSING")
    print("=" * 55)
    df, product_df, customer_products = preprocess_pipeline(DATA_PATH)

    # === 2. Build Model ===
    print("\n" + "=" * 55)
    print("  STEP 2: BUILD CONTENT-BASED MODEL")
    print("=" * 55)
    model = ContentBasedRecommender()
    model.fit(product_df)

    # === 3. Sample Recommendations ===
    print("\n" + "=" * 55)
    print("  STEP 3: SAMPLE RECOMMENDATIONS")
    print("=" * 55)

    # Pick 3 sample products for demo
    sample_products = product_df.sample(3, random_state=42)
    for _, row in sample_products.iterrows():
        product_name = row["ProductName"]
        print(f"\n>> Recommendations for: '{product_name}' (Price: {row['AvgPrice']:.2f})")
        print("-" * 55)
        recs = model.get_recommendations(product_name, top_n=5)
        if not recs.empty:
            for i, (_, rec) in enumerate(recs.iterrows(), 1):
                print(f"   {i}. {rec['ProductName']:<40s} "
                      f"Price: {rec['AvgPrice']:>7.2f}  "
                      f"Similarity: {rec['SimilarityScore']:.4f}")
        else:
            print("   No recommendations found.")

    # === 4. Evaluation ===
    print("\n" + "=" * 55)
    print("  STEP 4: MODEL EVALUATION")
    print("=" * 55)
    results = evaluate_model(model, customer_products, k=K, sample_size=500)
    print_evaluation_results(results)

    # === 5. Visualization ===
    print("\n" + "=" * 55)
    print("  STEP 5: DATA VISUALIZATION")
    print("=" * 55)
    generate_all_visualizations(df, product_df, model, results, save_dir=OUTPUT_DIR)

    # === 6. Save Model ===
    print("\n" + "=" * 55)
    print("  STEP 6: SAVE MODEL")
    print("=" * 55)
    model.save_model("models")

    print("\nDone!")


if __name__ == "__main__":
    main()
