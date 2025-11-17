import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# --- Settings ---
# Please specify the path to the 1M dataset
# (Assumes 'preprocessed_1m.csv' is inside a 'preprocessed_1m' folder)
DATA_FILE = Path.cwd() / "preprocessed_1m" / "preprocessed_1m.csv"

# Threshold mentioned in the paper
THRESHOLD = 10

# Output plot filename (as requested)
OUTPUT_PLOT = "item_rating_distribution.png"

def analyze_cold_start(data_path, threshold):
    print(f"Loading data from: {data_path}")
    if not data_path.exists():
        print(f"[ERROR] Data file not found: {data_path}")
        print("Please check the DATA_FILE path at the top of the script.")
        return

    df = pd.read_csv(data_path)
    print(f"Data loaded: {len(df):,} total ratings.")

    # 1. Analyze item rating counts
    item_counts = df.groupby('movieId_enc').size()
    sparse_items = item_counts[item_counts < threshold]
    total_items = len(item_counts)
    pct_sparse_items = (len(sparse_items) / total_items) * 100

    print("\n--- Item Analysis ---")
    print(f"Total unique items (movies): {total_items:,}")
    print(f"Items with fewer than {threshold} ratings: {len(sparse_items):,} items")
    print(f"  -> {pct_sparse_items:.2f}% of all items")

    # 2. Visualize item distribution (Long-Tail plot)
    print(f"\n--- Saving plot ---")
    try:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot histogram on a log scale to see the long-tail
        item_counts.hist(bins=350, ax=ax, log=False)
        
        ax.set_title('Item Rating Distribution')
        ax.set_xlabel('Number of Ratings per Item')
        ax.set_ylabel('Number of Items (Count)')
            
        plt.tight_layout()
        
        plt.savefig(OUTPUT_PLOT)
        print(f"Plot saved: {OUTPUT_PLOT}")
    except Exception as e:
        print(f"An error occurred while generating the plot: {e}")

if __name__ == "__main__":
    analyze_cold_start(DATA_FILE, THRESHOLD)