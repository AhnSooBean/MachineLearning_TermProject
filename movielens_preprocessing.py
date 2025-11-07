# ------------------------------------------------------------
# - MovieLens 100k: 전체를 모델 학습/구현용으로 사용 (train/test 분리 X)
# - MovieLens 1M  : 전체를 테스트용으로 사용

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import os


# ========================================
# (1) MovieLens 100k 로드 함수
# ========================================
def load_movielens_100k(base_path="./ml-100k"):
    """MovieLens 100k 데이터 로드 및 결합"""
    ratings = pd.read_csv(
        os.path.join(base_path, "u.data"),
        sep="\t",
        names=["userId", "movieId", "rating", "timestamp"],
        encoding="latin-1"
    )
    movies = pd.read_csv(
        os.path.join(base_path, "u.item"),
        sep="|",
        encoding="latin-1",
        names=["movieId", "title", "release_date", "video_release_date", "IMDb_URL",
               "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
               "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
               "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
    )
    movies = movies[["movieId", "title"]]
    df = pd.merge(ratings, movies, on="movieId")
    print(f"[100k] Loaded {len(df):,} rows")
    return df


# ========================================
# (2) MovieLens 1M 로드 함수
# ========================================
def load_movielens_1m(base_path="./ml-1m"):
    """MovieLens 1M 데이터 로드 및 결합"""
    ratings = pd.read_csv(
        os.path.join(base_path, "ratings.dat"),
        sep="::",
        engine="python",
        names=["userId", "movieId", "rating", "timestamp"],
        encoding="latin-1"
    )
    movies = pd.read_csv(
        os.path.join(base_path, "movies.dat"),
        sep="::",
        engine="python",
        names=["movieId", "title", "genres"],
        encoding="latin-1"
    )
    df = pd.merge(ratings, movies, on="movieId")
    print(f"[1M] Loaded {len(df):,} rows")
    return df


# ========================================
# (3) 100k 전처리 함수 (전체 사용)
# ========================================
def preprocess_100k(df):
    df = df.copy()

    user_enc = LabelEncoder()
    item_enc = LabelEncoder()
    df["userId_enc"] = user_enc.fit_transform(df["userId"])
    df["movieId_enc"] = item_enc.fit_transform(df["movieId"])

    os.makedirs("./encoders_100k", exist_ok=True)
    joblib.dump(user_enc, "./encoders_100k/user_encoder_100k.pkl")
    joblib.dump(item_enc, "./encoders_100k/item_encoder_100k.pkl")

    os.makedirs("./preprocessed_100k", exist_ok=True)
    df.to_csv("./preprocessed_100k/preprocessed_100k.csv", index=False)

    print(f"[100k] Preprocessing complete → preprocessed_100k/preprocessed_100k.csv")

# ========================================
# (4) 1M 전처리 함수 (전체 사용)
# ========================================
def preprocess_1m(df):
    """1M 데이터를 전체 테스트용으로 인코딩 및 저장"""
    df = df.copy()

    user_enc = LabelEncoder()
    item_enc = LabelEncoder()

    df["userId_enc"] = user_enc.fit_transform(df["userId"])
    df["movieId_enc"] = item_enc.fit_transform(df["movieId"])

    # 인코더 저장
    os.makedirs("./encoders_1m", exist_ok=True)
    joblib.dump(user_enc, "./encoders_1m/user_encoder_1m.pkl")
    joblib.dump(item_enc, "./encoders_1m/item_encoder_1m.pkl")

    os.makedirs("./preprocessed_1m", exist_ok=True)
    df.to_csv("./preprocessed_1m.csv", index=False)

    print(f"[1M] Preprocessing complete → {len(df):,} rows saved to ./preprocessed_1m/full.csv")


# ========================================
# (5) 실행부
# ========================================
if __name__ == "__main__":
    # MovieLens 100k 전체 학습용
    df_100k = load_movielens_100k("./ml-100k")
    preprocess_100k(df_100k)

    # MovieLens 1M 전체 테스트용
    df_1m = load_movielens_1m("./ml-1m")
    preprocess_1m(df_1m)