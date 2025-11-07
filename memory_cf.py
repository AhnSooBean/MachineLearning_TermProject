#pip install surprise 하기 
import pandas as pd
from surprise import Dataset, Reader, KNNBasic
import joblib
import numpy as np

# 데이터 로드
file_path = 'preprocessed_100k/preprocessed_100k.csv'
df = pd.read_csv(file_path)

# Surprise 데이터셋 변환
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['userId_enc', 'movieId_enc', 'rating']], reader)
trainset = data.build_full_trainset()

# 설정: True : 사용자 기반 / False : 아이템 기반
USER_BASED = False   # True → 사용자 기반 / False → 아이템 기반

sim_options = {"name": "cosine", "user_based": USER_BASED}
model = KNNBasic(k=40, sim_options=sim_options) # 임시 k값 40
model.fit(trainset)

# 특정 사용자에 대해 추천 예측
target_user_id_enc = 20 # 암시 사용자 30 , (0~942) 까지 가능

# 사용자 존재 여부 확인
if target_user_id_enc not in trainset._raw2inner_id_users:
    raise ValueError(f"User {target_user_id_enc} not found in training set")

inner_uid = trainset.to_inner_uid(target_user_id_enc)
sim_matrix = model.sim

# 평점 데이터 가져오기
user_ratings = trainset.ur[inner_uid]
all_items = set(range(trainset.n_items))
rated_items = {iid for (iid, _) in user_ratings}
unrated_items = list(all_items - rated_items)

# 추천 점수 계산
pred_ratings = []

if USER_BASED:
    # 사용자 기반
    neighbors = np.argsort(sim_matrix[inner_uid])[::-1][1:model.k+1]

    for iid in unrated_items:
        sim_sum, weighted_sum = 0, 0
        for neighbor in neighbors:
            for (neighbor_iid, neighbor_rating) in trainset.ur[neighbor]:
                if neighbor_iid == iid:
                    sim = sim_matrix[inner_uid, neighbor]
                    weighted_sum += sim * neighbor_rating
                    sim_sum += abs(sim)
                    break
        if sim_sum > 0:
            est = weighted_sum / sim_sum
            pred_ratings.append((iid, est))

else:
    # 아이템 기반
    for iid in unrated_items:
        sim_sum, weighted_sum = 0, 0
        for (rated_iid, rating) in user_ratings:
            sim = sim_matrix[iid, rated_iid]
            if sim > 0:  # 음의 유사도는 제외
                weighted_sum += sim * rating
                sim_sum += sim
        if sim_sum > 0:
            est = weighted_sum / sim_sum
            pred_ratings.append((iid, est))

# 예측 결과 정렬
pred_ratings.sort(key=lambda x: x[1], reverse=True)

# 영화 제목 복원
item_encoder = joblib.load('encoders_100k/item_encoder_100k.pkl')
movie_id_to_title = df.drop_duplicates(subset='movieId_enc').set_index('movieId_enc')['title']

# 상위 N개 추천
N_RECOMMENDATIONS = 10 # 몇개를 추천할지 값 변경
print(f"사용자(enc={target_user_id_enc})에게 추천 영화 Top {N_RECOMMENDATIONS}:")
for iid, est in pred_ratings[:N_RECOMMENDATIONS]:
    movie_id_enc = trainset.to_raw_iid(iid)
    movie_title = movie_id_to_title.get(int(movie_id_enc), "제목 없음")
    original_movie_id = item_encoder.inverse_transform([int(movie_id_enc)])[0]
    print(f"  - 제목: {movie_title} (Orig ID: {original_movie_id}) | 예상 평점: {est:.2f}")