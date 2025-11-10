import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm

# Surprise library (Memory-based CF)
from surprise import Dataset, Reader, KNNBasic
from surprise.accuracy import rmse as surprise_rmse

# KernelMF library (Model-based CF)
# (Must have 'matrix_factorization.py' in the same folder)
try:
    from matrix_factorization import KernelMF
except ImportError:
    print("="*60)
    print("[ERROR] 'matrix_factorization.py' file not found.")
    print("        Please place it in the same folder as evaluation.py.")
    print("="*60)
    exit()

# Scikit-learn (for calculating evaluation metrics)
from sklearn.metrics import mean_squared_error, mean_absolute_error, ndcg_score

# --- Settings ---
BASE = Path(__file__).parent
K_FOR_NDCG = 10  # Top-K items to calculate nDCG for (Top-10)

# Path to the 1M trained model file
MODEL_MF_PKL = BASE / "_artifacts_train_only" / "kernelmf_model.pkl"


def load_data():
    """Loads the training (1m) and test (100k) data."""
    print("[INFO] Loading data...")
    
    train_path = BASE / 'preprocessed_1m' / 'preprocessed_1m.csv'
    test_path = BASE / 'preprocessed_100k' / 'preprocessed_100k.csv'

    if not train_path.exists() or not test_path.exists():
        print(f"[ERROR] Data files not found.")
        print(f"  - 1m path: {train_path}")
        print(f"  - 100k path: {test_path}")
        return None, None

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Prepare Surprise Reader and Datasets
    reader = Reader(rating_scale=(1, 5))
    train_data = Dataset.load_from_df(train_df[['userId_enc', 'movieId_enc', 'rating']], reader)
    test_data = Dataset.load_from_df(test_df[['userId_enc', 'movieId_enc', 'rating']], reader)

    # Build Surprise trainset and testset
    trainset = train_data.build_full_trainset()
    testset_for_surprise = test_data.build_full_trainset().build_testset()
    
    print(f"[INFO] Train set (1m): {len(train_df):,} records, {train_df['userId_enc'].nunique():,} users")
    print(f"[INFO] Test set (100k): {len(test_df):,} records, {test_df['userId_enc'].nunique():,} users")
    
    return train_df, test_df, trainset, testset_for_surprise


def evaluate_knn(trainset, testset_for_surprise, test_df):
    """Trains and evaluates the Memory-based (KNNBasic) model."""
    print("\n--- 1. Evaluating Memory-based (KNNBasic) ---")
    
    # 1. Train model (on 1m data)
    print("[KNN] Training KNNBasic model on 1m data...")
    sim_options = {"name": "cosine", "user_based": False} # Item-based
    model_knn = KNNBasic(k=40, sim_options=sim_options, verbose=False)
    model_knn.fit(trainset)

    # 2. Evaluate RMSE / MAE (on 100k data)
    print("[KNN] Calculating RMSE/MAE on 100k test set...")
    predictions_knn = model_knn.test(testset_for_surprise)
    
    true_ratings = [p.r_ui for p in predictions_knn]
    pred_ratings = [p.est for p in predictions_knn]
    
    rmse = np.sqrt(mean_squared_error(true_ratings, pred_ratings))
    mae = mean_absolute_error(true_ratings, pred_ratings)
    print(f"[KNN] RMSE = {rmse:.4f}, MAE = {mae:.4f}")

    # 3. Evaluate nDCG@k (on 100k data)
    print(f"[KNN] Calculating nDCG@{K_FOR_NDCG} on 100k test set...")
    ndcg_scores = []
    test_users = test_df['userId_enc'].unique()

    for user in tqdm(test_users, desc="nDCG (KNN)"):
        # Get the list of items this user rated in the 100k set
        user_test_data = test_df[test_df['userId_enc'] == user]
        if user_test_data.empty:
            continue
        
        # Ground truth ratings (y_true)
        y_true = [user_test_data['rating'].values]
        
        # Predicted ratings (y_pred)
        y_pred = [model_knn.predict(user, item).est for item in user_test_data['movieId_enc']]
        
        # Calculate nDCG
        score = ndcg_score(y_true, [y_pred], k=K_FOR_NDCG)
        ndcg_scores.append(score)

    avg_ndcg = np.mean(ndcg_scores)
    print(f"[KNN] nDCG@{K_FOR_NDCG} = {avg_ndcg:.4f}")
    
    return rmse, mae, avg_ndcg


def evaluate_mf(train_df, test_df):
    """Loads and evaluates the Model-based (KernelMF) model."""
    print("\n--- 2. Evaluating Model-based (KernelMF) ---")

    # 1. Load model (Training logic is removed)
    if not MODEL_MF_PKL.exists():
        print(f"[ERROR] Model file not found: {MODEL_MF_PKL}")
        print("        Ensure the pre-trained .pkl file exists in the _artifacts_train_only folder.")
        # Return NaN values on failure to avoid crashing the summary table
        return np.nan, np.nan, np.nan 
    
    with open(MODEL_MF_PKL, "rb") as f:
        model_mf = pickle.load(f)
    print(f"[MF] Loaded pre-trained model: {MODEL_MF_PKL}")


    # 2. Evaluate RMSE / MAE (on 100k data)
    # Prepare the test set in the DataFrame format required by KernelMF.predict()
    print("[MF] Calculating RMSE/MAE on 100k test set... (Batch prediction)")
    X_test = test_df.rename(columns={"userId_enc": "user_id", "movieId_enc": "item_id"})
    
    true_ratings = X_test['rating'].values
    
    # Pass the entire DataFrame to get all predictions at once
    pred_ratings = model_mf.predict(X_test) 

    rmse = np.sqrt(mean_squared_error(true_ratings, pred_ratings))
    mae = mean_absolute_error(true_ratings, pred_ratings)
    print(f"[MF] RMSE = {rmse:.4f}, MAE = {mae:.4f}")

    # 3. Evaluate nDCG@k (on 100k data)
    print(f"[MF] Calculating nDCG@{K_FOR_NDCG} on 100k test set...")
    ndcg_scores = []
    test_users = test_df['userId_enc'].unique()

    for user in tqdm(test_users, desc="nDCG (MF)"):
        user_test_data = test_df[test_df['userId_enc'] == user]
        if user_test_data.empty:
            continue
            
        # Ground truth ratings (y_true)
        y_true = [user_test_data['rating'].values]
        
        # Prepare this user's test data as a DataFrame for predict(X)
        X_user_test = user_test_data.rename(columns={"userId_enc": "user_id", "movieId_enc": "item_id"})
        
        # Get all predictions for this user's test items at once
        y_pred = [model_mf.predict(X_user_test)] # predict(X) returns an ndarray
        
        score = ndcg_score(y_true, y_pred, k=K_FOR_NDCG)
        ndcg_scores.append(score)

    avg_ndcg = np.mean(ndcg_scores)
    print(f"[MF] nDCG@{K_FOR_NDCG} = {avg_ndcg:.4f}")
    
    return rmse, mae, avg_ndcg


def main():
    # 1. Load data
    train_df, test_df, trainset, testset_for_surprise = load_data()
    if train_df is None:
        return

    # 2. Run evaluation for each model
    knn_rmse, knn_mae, knn_ndcg = evaluate_knn(trainset, testset_for_surprise, test_df)
    mf_rmse, mf_mae, mf_ndcg = evaluate_mf(train_df, test_df)

    # 3. Print final summary
    print("\n" + "="*60)
    print("           [Final Evaluation Summary (Testset: 100k)]")
    print("="*60)
    print(f"| {'Model':<20} | {'RMSE (↓)':<10} | {'MAE (↓)':<10} | {'nDCG@10 (↑)':<12} |")
    print(f"|{'-'*22}|{'-'*12}|{'-'*12}|{'-'*14}|")
    print(f"| {'Memory-based (KNN)':<20} | {knn_rmse:<10.4f} | {knn_mae:<10.4f} | {knn_ndcg:<12.4f} |")
    print(f"| {'Model-based (KernelMF)':<20} | {mf_rmse:<10.4f} | {mf_mae:<10.4f} | {mf_ndcg:<12.4f} |")
    print("="*60)
    print("(↓) Lower is better / (↑) Higher is better")

if __name__ == "__main__":
    main()