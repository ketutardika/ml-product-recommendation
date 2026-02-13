# ML Product Recommendation

A product recommendation system using **Content-Based Filtering** built with scikit-learn. The model uses TF-IDF vectorization on product names combined with price features, then computes cosine similarity to recommend similar products.

## Project Structure

```
ml-product-recommendation/
├── main.py                        # Main pipeline (run this)
├── requirements.txt               # Python dependencies
├── data/
│   └── sales_transaction_v4a.csv  # Transaction dataset (~536K rows)
├── src/
│   ├── data_preprocessing.py      # Data cleaning & preparation
│   ├── content_based_model.py     # TF-IDF + Cosine Similarity model
│   ├── evaluation.py              # Precision@K, Recall@K, F1@K, NDCG@K
│   └── visualization.py           # Data & result visualizations
├── models/                        # Saved model artifacts
├── output/                        # Generated visualization plots
└── notebooks/                     # Jupyter notebooks (optional)
```

## Dataset

Source: [An Online Shop Business - Kaggle](https://www.kaggle.com/datasets/gabrielramos87/an-online-shop-business)

The dataset contains **536,350** e-commerce transaction records with the following columns:

| Column | Description |
|---|---|
| TransactionNo | Transaction ID (prefix 'C' = cancelled) |
| Date | Transaction date |
| ProductNo | Product ID |
| ProductName | Product name |
| Price | Unit price |
| Quantity | Quantity purchased |
| CustomerNo | Customer ID |
| Country | Customer country |

## How It Works

### 1. Data Preprocessing
- Remove cancelled transactions (TransactionNo starting with 'C')
- Remove rows with negative/zero quantity or price
- Drop missing values and duplicates
- Build unique product dataframe with aggregated features

### 2. Content-Based Filtering Model
- **TF-IDF Vectorizer** extracts text features from product names
- **MinMaxScaler** normalizes price as a numerical feature
- Features are combined into a single matrix (TF-IDF + weighted price)
- **Cosine Similarity** computes pairwise similarity between all products

### 3. Evaluation Metrics
The model is evaluated using customer purchase history as ground truth:

| Metric | Description |
|---|---|
| **Precision@K** | Proportion of recommended items that are relevant |
| **Recall@K** | Proportion of relevant items that were recommended |
| **F1-Score@K** | Harmonic mean of Precision and Recall |
| **NDCG@K** | Normalized Discounted Cumulative Gain (ranking quality) |

### 4. Visualizations
The system generates the following plots in the `output/` folder:

- **Data Overview** - Top products, price distribution, country breakdown, products per customer
- **Similarity Heatmap** - Cosine similarity matrix sample
- **Similarity Distribution** - Distribution of pairwise similarity scores
- **Recommendation Examples** - Similarity scores and price comparison charts
- **Evaluation Metrics** - Bar chart and radar chart of all metrics

## Getting Started

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
pip install -r requirements.txt
```

### Run

```bash
python main.py
```

### Output

The pipeline will:
1. Preprocess and clean the transaction data
2. Build the content-based filtering model
3. Display sample recommendations in the console
4. Evaluate the model and print metrics
5. Generate visualization plots in `output/`
6. Save model artifacts to `models/`

## Sample Results

```
>> Recommendations for: 'white wicker star' (Price: 11.31)
   1. wicker star                    Price: 12.12  Similarity: 0.8852
   2. small white heart of wicker    Price: 11.29  Similarity: 0.6425
   3. large white heart of wicker    Price: 12.57  Similarity: 0.6384
```

## Evaluation Results

| Metric | Score |
|---|---|
| Precision@10 | 0.1420 |
| Recall@10 | 0.0505 |
| F1-Score@10 | 0.0525 |
| NDCG@10 | 0.1685 |

## Tech Stack

- **pandas** - Data manipulation
- **scikit-learn** - TF-IDF, MinMaxScaler, Cosine Similarity
- **matplotlib / seaborn** - Visualization
- **scipy** - Sparse matrix operations
- **numpy** - Numerical computing
