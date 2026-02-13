import pandas as pd
import numpy as np


def load_data(filepath: str) -> pd.DataFrame:
    """Load CSV dataset."""
    df = pd.read_csv(filepath, encoding="utf-8")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean transaction data: remove cancellations, duplicates, missing values."""
    print(f"[Preprocessing] Initial data: {len(df)} rows")

    # Convert TransactionNo to string to check for 'C' prefix
    df["TransactionNo"] = df["TransactionNo"].astype(str)

    # Remove cancelled transactions (TransactionNo starting with 'C')
    df = df[~df["TransactionNo"].str.startswith("C")].copy()
    print(f"[Preprocessing] After removing cancellations: {len(df)} rows")

    # Remove rows with Quantity <= 0
    df = df[df["Quantity"] > 0].copy()
    print(f"[Preprocessing] After removing Quantity <= 0: {len(df)} rows")

    # Remove rows with Price <= 0
    df = df[df["Price"] > 0].copy()
    print(f"[Preprocessing] After removing Price <= 0: {len(df)} rows")

    # Drop missing values in key columns
    key_cols = ["ProductNo", "ProductName", "CustomerNo"]
    df = df.dropna(subset=key_cols).copy()
    print(f"[Preprocessing] After removing missing values: {len(df)} rows")

    # Remove duplicates
    df = df.drop_duplicates().copy()
    print(f"[Preprocessing] After removing duplicates: {len(df)} rows")

    # Clean ProductName
    df["ProductName"] = df["ProductName"].str.strip().str.lower()

    # Convert CustomerNo to string
    df["CustomerNo"] = df["CustomerNo"].astype(int).astype(str)

    return df


def build_product_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Create unique product dataframe with aggregated features."""
    product_df = df.groupby("ProductNo").agg(
        ProductName=("ProductName", "first"),
        AvgPrice=("Price", "mean"),
        TotalQuantitySold=("Quantity", "sum"),
        TransactionCount=("TransactionNo", "nunique"),
    ).reset_index()

    print(f"[Preprocessing] Total unique products: {len(product_df)}")
    return product_df


def build_customer_product_matrix(df: pd.DataFrame) -> dict:
    """Build dictionary of customer -> set of purchased ProductNo."""
    customer_products = df.groupby("CustomerNo")["ProductNo"].apply(set).to_dict()
    print(f"[Preprocessing] Total unique customers: {len(customer_products)}")
    return customer_products


def preprocess_pipeline(filepath: str):
    """Run full preprocessing pipeline."""
    df = load_data(filepath)
    df = clean_data(df)
    product_df = build_product_dataframe(df)
    customer_products = build_customer_product_matrix(df)
    return df, product_df, customer_products
