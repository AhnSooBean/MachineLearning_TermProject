# MovieLens 1M 데이터로 사용자별 Top-5 영화 추천
# matrix_factorization 모듈의 KernelMF 모델 사용
# - 처음 실행할 때 모델이 없으면 새로 학습한 후 저장
# - 이후 실행할 때는 저장된 모델을 불러와 바로 추천 수행
# 추천 시, 사용자가 이미 본 영화는 제외하고 새로운 영화만 추천
import numpy as np
import pandas as pd
import pickle # 학습된 모델을 저장하고 불러오는 모듈
from pathlib import Path
from matrix_factorization import KernelMF

# 입력 데이터 경로
CSVpath  = "C:/Users/ans52/OneDrive/바탕 화면/Matrix Factorization/preprocessed_1m.csv"

# 학습된 모델 경로
OUT_DIR = Path("C:/Users/ans52/OneDrive/바탕 화면/Matrix Factorization/_artifacts_train_only")
OUT_DIR.mkdir(parents=True, exist_ok=True) # 폴더가 없으면 자동으로 생성

MODEL_PKL = OUT_DIR / "kernelmf_model.pkl" # 학습 완료 후 저장할 모델 파일
OUT_CSV   = OUT_DIR / "recommendations_top5.csv" # 최종 추천 결과 저장용 CSV

df = pd.read_csv(CSVpath) # 데이터 로드

# 필수 컬럼이 모두 존재하는지 확인
need = ["userId_enc", "movieId_enc", "rating"]
for c in need:
    if c not in df.columns:
        raise ValueError(f"Required column missing: {c} | Actual column: {df.columns.tolist()}")

# 데이터 타입 고정 (학습/추천 과정에서 dtype 불일치 방지)
df["userId_enc"]  = df["userId_enc"].astype("int64")
df["movieId_enc"] = df["movieId_enc"].astype("int64")
df["rating"] = df["rating"].astype("float32")

# title이나 genres 있으면 함께 사용
meta_cols = ["movieId_enc"]
if "title" in df.columns:
    meta_cols.append("title")
if "genres" in df.columns:
    meta_cols.append("genres")

meta_map = None
if len(meta_cols) > 1: # movieId_enc 외에 추가 정보가 있을 때
    meta_map = (
        df[meta_cols]
        .drop_duplicates(subset=["movieId_enc"]) # 영화별 중복 제거
        .rename(columns={"movieId_enc": "item_id"}) # movieId_enc, item_id으로 이름 변경
    )

    # genres 문자열 전처리
    if "genres" in meta_map.columns:
        meta_map["genres"] = meta_map["genres"].astype(str).str.strip() # 공백 제거

# 모델 입력 형식으로 컬럼명 정렬 (KernelMF는 user_id, item_id, rating 형식)
X = df.rename(columns={"userId_enc": "user_id", "movieId_enc": "item_id"})[["user_id", "item_id"]]
y = df["rating"].values

print(f"[INFO] rows={len(df):,}, users={X['user_id'].nunique():,}, items={X['item_id'].nunique():,}")

if MODEL_PKL.exists(): # 학습된 모델이 있으면 로드
    with open(MODEL_PKL, "rb") as f:
        model = pickle.load(f)
    print(f"[LOAD] {MODEL_PKL}")

else: # 학습된 모델이 없으면 새로 학습
    print("[TRAIN] Saved model not found. Training on 1M now...")
    model = KernelMF(
        n_epochs=20, # 학습 반복 횟수
        n_factors=64, # 잠재 요인 수. 32~200 범위
        lr=0.005, # 학습률
        reg=0.01, # 정규화 강도(과적합 방지)
        verbose=1 # 학습 로그 표시 여부
    )
    model.fit(X, y)

    # 학습 완료 후 모델을 피클로 저장 (다음 실행부터 바로 불러오기 가능)
    with open(MODEL_PKL, "wb") as f:
        pickle.dump(model, f)
    print(f"[SAVE] {MODEL_PKL}")

# 추천 생성 시, 학습 데이터에 등장했던 (user, item)쌍은 "이미 본 아이템"으로 제외
# 추천 시 이 목록은 제외하여 중복 추천 방지
items_known_by_user = (
    X.groupby("user_id")["item_id"]
     .apply(lambda s: np.unique(s.values)) # 중복 제거 및 numpy 배열화
     .to_dict()
)

# 추천 생성 결과를 DataFrame으로 변환
def _to_df_recs(recs):
    # 모델에 따라 recommend 반환 타입 처리
    # DataFrame (2D): ['item_id','score'] 또는 ['item_id']
    if isinstance(recs, pd.DataFrame): # score 컬럼이 없다면 빈 점수 컬럼 추가
        if "score" not in recs.columns:
            recs["score"] = np.nan
        return recs[["item_id", "score"]].copy()
    
    # list (1D): item_id만 있음, score는 NaN으로
    arr = np.asarray(recs).reshape(-1)
    return pd.DataFrame({"item_id": arr, "score": np.nan})

all_users = X["user_id"].unique() # 모든 사용자 ID 목록
rows = []

print(f"[RECOMMEND] generating top-5 for {len(all_users):,} users...")

# 사용자별로 추천 수행
for idx, u in enumerate(all_users, start=1):
    # 이미 본 영화 목록
    known = items_known_by_user.get(u, np.array([], dtype=np.int64))

    # 모델에게 추천 요청
    # - user: 추천 대상 사용자 ID
    # - items_known: 제외할 아이템 목록(이미 본 것)
    # - n_items: 상위 몇 개를 반환할지
    recs = model.recommend(user=int(u), items_known=known, n_items=5)

    # 반환 타입을 DataFrame으로 통일
    recs_df = _to_df_recs(recs)
    recs_df.insert(0, "user_id", int(u)) # (user_id, item_id, score) 형식

    # title/genres 있으면 추천 결과에 결합
    if meta_map is not None:
        recs_df = recs_df.merge(meta_map, on="item_id", how="left")
    rows.append(recs_df)

    # 진행 상황 출력 (1000명 단위)
    if idx % 1000 == 0 or idx == len(all_users):
        print(f"  - {idx:,}/{len(all_users):,} users done")

# 모든 사용자 추천 결과를 하나의 DataFrame으로 결합
out = pd.concat(rows, ignore_index=True)

# 각 사용자 안에서 점수 높은 순으로 정렬 (점수 미제공이면 NaN이 뒤로)
# 사용자 전체는 user_id 오름차순으로 정렬
out = out.sort_values(by=["user_id", "score"], ascending=[True, False], na_position="last")

# UTF-8 with BOM으로 CSV 저장해 엑셀에서 한글 컬럼/값이 깨지지 않음
out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
print(f"\n[DONE] saved -> {OUT_CSV}")