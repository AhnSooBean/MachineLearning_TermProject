import pandas as pd
import math
from surprise import Dataset, Reader, KNNBasic
import pickle
import sys
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# --- Settings ---
BASE = Path(__file__).parent
MODEL_MF_PKL = BASE / "_artifacts_train_only" / "kernelmf_model.pkl"

# Compute DCG given a list of binary relevance values.
# (Relates to ranking relevance: higher relevance near top of the list is better)
def compute_dcg(rel_list, k):
    dcg = 0.0
    for i, rel in enumerate(rel_list[:k], start=1):
        # Use the formula: (2^rel - 1) / log2(1 + i)
        dcg += (2**rel - 1) / math.log2(1 + i) if rel > 0 else 0.0
    return dcg

# Compute nDCG@k for a single user.
# (Compares actual DCG with ideal DCG for normalized result)
def compute_ndcg_at_k(recommended_items, true_items, k):
    # Binary relevance for recommended list: 1 if item is in true_items, else 0
    rels = [1 if item in true_items else 0 for item in recommended_items]
    DCG = compute_dcg(rels, k)
    # Compute ideal DCG (IDCG) using all relevant items (binary relevance=1) in top ranks
    R = len(true_items)                       # total relevant items for user
    ideal_rels = [1] * min(R, k)              # ideal list of length min(R, k) with all hits
    IDCG = compute_dcg(ideal_rels, k)
    return (DCG / IDCG) if IDCG > 0 else 0.0  # nDCG (0 if no relevant items)

# Train KNNBasic and evaluate average nDCG@k.
# (Memory-based collaborative filtering evaluation)
def evaluate_knn(train_df, test_df, test_truth, k=10):
    # 1. Train KNNBasic (item-based) on the training data
    reader = Reader(rating_scale=(1, 5))
    train_data = Dataset.load_from_df(train_df[['userId_enc', 'movieId_enc', 'rating']], reader)
    trainset = train_data.build_full_trainset()
    model_knn = KNNBasic(k=40, sim_options={'name': 'cosine', 'user_based': False}, verbose=False)
    model_knn.fit(trainset)

    # 2. Prepare sets of known (seen) items and all items
    all_items = set(train_df['movieId_enc'].unique())
    items_known_by_user = train_df.groupby('userId_enc')['movieId_enc'].apply(set).to_dict()

    # 3. Compute nDCG for each user in test
    ndcg_scores = []
    for user, true_items in tqdm(test_truth.items(), desc="Evaluating KNN"):
        # Skip users not in training (model cannot recommend) 
        if user not in items_known_by_user:
            continue
        # Candidate items = all items minus those seen by user in train
        seen_items = items_known_by_user[user]
        candidates = all_items - seen_items
        if not candidates:
            continue  # skip if no new item to recommend
        # Predict scores for all candidate items
        predictions = []
        for item in candidates:
            pred = model_knn.predict(user, item)   # predict rating for (user, item)
            predictions.append((item, pred.est))
        # Get Top-K items with highest predicted ratings
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_k_items = [item for item, score in predictions[:k]]
        # Calculate nDCG for this user and collect it
        ndcg = compute_ndcg_at_k(top_k_items, true_items, k)
        ndcg_scores.append(ndcg)
        
    avg_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0

    # 4. Compute RMSE/MAE
    reader_test = Reader(rating_scale=(1, 5))
    test_data = Dataset.load_from_df(test_df[['userId_enc', 'movieId_enc', 'rating']], reader_test)
    testset = test_data.build_full_trainset().build_testset() # Generate (u, i, r) triples
    predictions = model_knn.test(testset) # Predict rating for each (u, i)
    true_ratings = [pred.r_ui for pred in predictions]
    pred_ratings = [pred.est for pred in predictions]
    rmse = np.sqrt(mean_squared_error(true_ratings, pred_ratings))
    mae = mean_absolute_error(true_ratings, pred_ratings)

    return avg_ndcg, rmse, mae

# Load KernelMF model and evaluate average nDCG@k, RMSE, and MAE.
# (Model-based collaborative filtering evaluation)
def evaluate_mf(train_df, test_df, k=10):
    # 1. Load the pre-trained KernelMF model (from pickle file)
    if not MODEL_MF_PKL.exists():
        print(f"[ERROR] Model file not found: {MODEL_MF_PKL}")
        return np.nan, np.nan, np.nan

    with open(MODEL_MF_PKL, "rb") as f:
        model_mf = pickle.load(f)
    print(f"[MF] Loaded pre-trained model: {MODEL_MF_PKL}")

    # 2. Prepare known items per user from training data
    items_known_by_user = train_df.groupby('userId_enc')['movieId_enc'].apply(set).to_dict()

    # 3. Build test set ground-truth per user
    test_truth = test_df.groupby('userId_enc')['movieId_enc'].apply(set).to_dict()

    # Lists for metrics
    ndcg_scores = []
    y_true, y_pred = [], []  # Lists for RMSE and MAE

    # 4. Calculate nDCG@K
    for user, true_items in tqdm(test_truth.items(), desc="Evaluating MF"):
        if user not in items_known_by_user:
            continue  # skip if user not in training data
        seen_items = items_known_by_user[user]
        # Get Top-K recommendations from the model (excluding seen items)
        recs = model_mf.recommend(user=user, items_known=seen_items, n_items=k)
        top_k_items = recs['item_id'].tolist() if isinstance(recs, pd.DataFrame) else list(recs)
        # Calculate nDCG for this user
        ndcg = compute_ndcg_at_k(top_k_items, true_items, k)
        ndcg_scores.append(ndcg)

    avg_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0

    # 5. Calculate RMSE & MAE
    X_test = test_df.rename(columns={'userId_enc': 'user_id', 'movieId_enc': 'item_id'})
    y_true = X_test['rating'].values
    y_pred = model_mf.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    return avg_ndcg, rmse, mae


# --- Main Execution ---
if __name__ == "__main__":
    # Load training and test datasets
    print("[INFO] Loading data...")
    train_path = BASE / 'preprocessed_1m' / 'preprocessed_1m.csv'
    test_path = BASE / 'preprocessed_100k' / 'preprocessed_100k.csv'

    if not train_path.exists() or not test_path.exists():
        print(f"[ERROR] Data files not found.")
        sys.exit(1)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Build ground-truth relevant item sets for each user from the test data
    test_truth = test_df.groupby('userId_enc')['movieId_enc'].apply(set).to_dict()

    # Set the rank cutoff K (e.g., 10 for nDCG@10)
    K = 10

    # Evaluate each model and compute average nDCG@K
    avg_ndcg_knn, rmse_knn, mae_knn = evaluate_knn(train_df, test_df, test_truth, k=K)
    avg_ndcg_mf, rmse_mf, mae_mf = evaluate_mf(train_df, test_df, k=K)

    # Output the results
    print(f"\n[Evaluation Results]")
    print(f"Memory-based (KNNBasic) model average nDCG@{K}: {avg_ndcg_knn:.4f}")
    print(f"Memory-based (KNNBasic) RMSE: {rmse_knn:.4f}")
    print(f"Memory-based (KNNBasic) MAE:  {mae_knn:.4f}")
    print(f"\nModel-based (KernelMF) model average nDCG@{K}: {avg_ndcg_mf:.4f}")
    print(f"Model-based (KernelMF) RMSE: {rmse_mf:.4f}")
    print(f"Model-based (KernelMF) MAE:  {mae_mf:.4f}")
