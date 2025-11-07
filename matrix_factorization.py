# SGD 기반 Matrix Factorization for Model-based Collaborative Filtering
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Iterable
import numpy as np
import pandas as pd

"""
KernelMF: SGD 기반 MF (Matrix Factorization)
- 목표: A ≈ U @ V^T  (A: 평점행렬, U: 사용자 잠재요인, V: 아이템 잠재요인)
- 예측식:  r_hat = μ + b_u + b_i + <P_u, Q_i> (user/item bias 미사용 시 r_hat = <P_u, Q_i>)
- 손실함수: ∑(r - r_hat)^2 + λ(||P_u||^2 + ||Q_i||^2 [+ ||b_u||^2 + ||b_i||^2])
    r # 실제 평점
    r_hat # 모델의 예측 평점
    λ # 정규화 강도
    α # 학습률
- SGD 업데이트
    P_u ← P_u + α (e * Q_i - λ P_u) # 사용자 잠재 요인 벡터 P_u
    Q_i ← Q_i + α (e * P_u - λ Q_i) # 아이템 잠재 요인 벡터 Q_i
    b_u ← b_u + α (e - λ b_u) # 사용자 편향 b_u
    b_i ← b_i + α (e - λ b_i) # 아이템 편향 b_i
"""
@dataclass
class KernelMF:
    # 학습 하이퍼파라미터 설정
    n_factors: int = 64 # 잠재 요인 차원 수(k)
    n_epochs: int = 20 # 전체 학습 반복 횟수(epoch)
    lr: float = 0.005 # 학습률(α)
    reg: float = 0.01 # L2 정규화 강도(λ)
    use_bias: bool = True # 전역/사용자/아이템 편향 사용 여부
    verbose: int = 0 # 1 이상이면 epoch마다 학습 RMSE 로그 출력
    random_state: int = 42 # 난수 시드(결과 재현성)

    # 학습 후 파라미터
    global_mean_: float = 0.0 # 전체 평균 평점(μ)
    P_: Optional[np.ndarray] = None # 사용자 행렬 (n_users, k)
    Q_: Optional[np.ndarray] = None # 아이템 행렬 (n_items, k)
    bu_: Optional[np.ndarray] = None # 사용자 편향 (n_users,)
    bi_: Optional[np.ndarray] = None # 아이템 편향 (n_items,)
    n_users_: int = 0 # 학습 시 사용자 수
    n_items_: int = 0 # 학습 시 아이템 수

    # 내부 파라미터 초기화: U(=P_), V(=Q_), b_u, b_i
    def _init_params(self, n_users: int, n_items: int):
        # P_, Q_: 작은 정규분포 난수(표준편차 0.01)로 초기화
        # bu_, bi_: 0으로 초기화(use_bias=True인 경우)
        rng = np.random.RandomState(self.random_state)
        self.n_users_, self.n_items_ = n_users, n_items

        # P_, Q_: (n, k) 실수 행렬. 초기에는 작은 값 → 점진적 학습 안정화
        self.P_ = 0.01 * rng.standard_normal((n_users, self.n_factors)).astype(np.float32)
        self.Q_ = 0.01 * rng.standard_normal((n_items, self.n_factors)).astype(np.float32)
        
        if self.use_bias:
            # 편향은 0에서 시작 → 데이터가 있으면 SGD가 방향성 부여
            self.bu_ = np.zeros(n_users, dtype=np.float32)
            self.bi_ = np.zeros(n_items, dtype=np.float32)
        else:
            self.bu_ = None
            self.bi_ = None

    # 단일 (u, i) 쌍의 평점 예측
    # r_hat = μ + b_u + b_i + <P_u, Q_i> (use_bias=False면 <P_u, Q_i>)
    def _predict_pair(self, u: int, i: int) -> float:
        dot = float(np.dot(self.P_[u], self.Q_[i]))
        if self.use_bias:
            return self.global_mean_ + self.bu_[u] + self.bi_[i] + dot
        return dot

    # 전체 학습 (SGD)
    # 입력: X[user_id, item_id] (DataFrame), y=rating (1D array)
    # user_id/item_id는 0..N-1의 연속 정수(LabelEncoded)
    def fit(self, X: pd.DataFrame, y: Iterable[float]):
        # 입력 데이터를 numpy 배열로 변환
        u = X["user_id"].astype(int).to_numpy()
        i = X["item_id"].astype(int).to_numpy()
        r = np.asarray(y, dtype=np.float32)

        # 사용자/아이템 개수 (max index + 1)
        n_users = int(u.max()) + 1 if len(u) else 0
        n_items = int(i.max()) + 1 if len(i) else 0

        # 파라미터 초기화 + 전체 평균 μ 계산
        self._init_params(n_users, n_items)
        self.global_mean_ = float(r.mean()) if len(r) else 0.0

        # 샘플 인덱스 셔플용 배열
        idx = np.arange(len(r))
        rng = np.random.RandomState(self.random_state)

        # 메인 학습 루프 (SGD)
        for epoch in range(self.n_epochs):
            rng.shuffle(idx) # epoch마다 샘플 순서 섞어 학습 다양성 확보
            for t in idx:
                uu = u[t] # 사용자 인덱스
                ii = i[t] # 아이템 인덱스
                rr = r[t] # 실제 평점

                # 현재 파라미터로 예측
                pred = self._predict_pair(uu, ii)
                err = rr - pred # 오차 e = r - r_hat

                # 업데이트에 사용할 참조 캐시(속도↑)
                pu = self.P_[uu] # shape (k,)
                qi = self.Q_[ii] # shape (k,)

                # SGD 업데이트 (L2 정규화 포함)
                # P_u ← P_u + α (e * Q_i - λ P_u)
                self.P_[uu] += self.lr * (err * qi - self.reg * pu)
                # Q_i ← Q_i + α (e * P_u - λ Q_i)
                self.Q_[ii] += self.lr * (err * pu - self.reg * qi)

                # 편향 사용 시, bu/bi 업데이트
                if self.use_bias:
                    # b_u ← b_u + α (e - λ b_u)
                    self.bu_[uu] += self.lr * (err - self.reg * self.bu_[uu])
                    # b_i ← b_i + α (e - λ b_i)
                    self.bi_[ii] += self.lr * (err - self.reg * self.bi_[ii])

            # 모니터링: 각 epoch마다 RMSE 출력
            if self.verbose:
                preds = self.predict(X)
                rmse = np.sqrt(np.mean((r - preds) ** 2))
                print(f"[MF] epoch {epoch+1}/{self.n_epochs}  rmse={rmse:.4f}")

        return self

    # 신규 상호작용에 대해 사용자 파라미터(P_, bu_)만 빠르게 업데이트
    # 아이템 파라미터 Q, b_i는 고정
    def update_users(
        self,
        X_update: pd.DataFrame,
        y_update: Iterable[float],
        lr: Optional[float] = None,
        n_epochs: Optional[int] = None,
        verbose: int = 0,
    ):
        if self.P_ is None or self.Q_ is None:
            raise RuntimeError("Call fit() before update_users().")
        
        # 기본 하이퍼파라미터 재사용
        lr = self.lr if lr is None else lr
        n_epochs = self.n_epochs if n_epochs is None else n_epochs

        # 업데이트용 데이터
        u = X_update["user_id"].astype(int).to_numpy()
        i = X_update["item_id"].astype(int).to_numpy()
        r = np.asarray(y_update, dtype=np.float32)

        # 새로운 user id가 들어오면 P_/b_u 확장
        unknown_user_mask = u >= self.n_users_
        if np.any(unknown_user_mask):
            n_new_users = int(u.max()) + 1
            grow = n_new_users - self.n_users_

            # 신규 사용자 요인/편향은 0으로 시작 (학습으로 점차 맞춰짐)
            self.P_ = np.vstack(
                [self.P_, np.zeros((grow, self.n_factors), dtype=np.float32)]
            )
            if self.use_bias:
                self.bu_ = np.concatenate([self.bu_, np.zeros(grow, dtype=np.float32)])
            self.n_users_ = n_new_users

        # 아이템은 확장하지 않음 → 범위 넘어가면 마지막 인덱스로 제한
        i = np.where(i >= self.n_items_, self.n_items_ - 1, i)

        # 학습 반복
        idx = np.arange(len(r))
        rng = np.random.RandomState(self.random_state + 7) # fit과 다른 시드로 셔플

        for epoch in range(n_epochs):
            rng.shuffle(idx)
            for t in idx:
                uu = u[t]
                ii = i[t]
                rr = r[t]

                pred = self._predict_pair(uu, ii) # 예측
                err = rr - pred # 오차

                # 사용자 쪽(P_, bu_)만 업데이트 (Q_i, b_i 는 고정)
                pu = self.P_[uu]
                qi = self.Q_[ii]
                self.P_[uu] += lr * (err * qi - self.reg * pu)
                if self.use_bias:
                    self.bu_[uu] += lr * (err - self.reg * self.bu_[uu])

            if verbose:
                preds = self.predict(pd.DataFrame({"user_id": u, "item_id": i}))
                rmse = np.sqrt(np.mean((r - preds) ** 2))
                print(f"[MF-update-users] epoch {epoch+1}/{n_epochs}  rmse={rmse:.4f}")

        return self

    # 평점 예측 (벡터화된 배치 예측)
    # use_bias=True: μ + b_u + b_i + <P_u, Q_i>
    # use_bias=False: <P, Q>
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        u = X["user_id"].astype(int).to_numpy()
        i = X["item_id"].astype(int).to_numpy()

        # 입력된 user_id/item_id가 범위 넘어가면 마지막 인덱스로 제한
        u = np.where(u >= self.n_users_, self.n_users_ - 1, u)
        i = np.where(i >= self.n_items_, self.n_items_ - 1, i)

        if self.use_bias:
            return (
                self.global_mean_
                + self.bu_[u]
                + self.bi_[i]
                + np.sum(self.P_[u] * self.Q_[i], axis=1) # 요소곱 후 합 == 내적
            ).astype(np.float32)

        return (np.sum(self.P_[u] * self.Q_[i], axis=1)).astype(np.float32)

    # 단일 사용자에 대해 Top-N 추천
    # 전체 아이템에 대한 점수 계산 뒤, 이미 본/필터 아이템 제외
    # argpartition으로 Top-N 후보를 빠르게 추출 후 점수 기준 정렬
    def recommend(
        self,
        user: int, # 사용자 ID
        items_known: Optional[Iterable[int]] = None, # 이미 본 아이템
        n_items: int = 10, # 아이템 개수
        filter_items: Optional[Iterable[int]] = None, # 추가 제외 아이템
    ) -> pd.DataFrame:
        if self.P_ is None or self.Q_ is None:
            raise RuntimeError("Call fit() before recommend().")
        if user >= self.n_users_:
            raise ValueError(f"user {user} is out of range (n_users={self.n_users_})")

        # 모든 아이템 점수 계산 및 점수 내림차순 정렬
        # use_bias=True: μ + b_u + b_i + <Q_i, P_u>
        # use_bias=False: <Q_i, P_u>
        user_vec = self.P_[user]
        scores = self.Q_.dot(user_vec)
        if self.use_bias:
            scores = (
                scores
                + self.global_mean_
                + (self.bu_[user] if self.bu_ is not None else 0.0)
                + (self.bi_ if self.bi_ is not None else 0.0)
            )

        # 이미 본 아이템/추가 제외 아이템을 -inf 처리해, 추천 대상에서 제거
        mask = np.zeros(self.n_items_, dtype=bool)
        if items_known is not None:
            mask[np.asarray(list(items_known), dtype=int)] = True
        if filter_items is not None:
            mask[np.asarray(list(filter_items), dtype=int)] = True
        scores = np.where(mask, -np.inf, scores)

        # Top-N 인덱스 추출 (전체 정렬 대신 argpartition 사용해 속도 향상)
        if len(scores) == 0: # 아이템 수가 0인 상황 방지
            return pd.DataFrame({"item_id": np.array([], dtype=int), "score": np.array([], dtype=float)})
        
        # kth 인덱스가 배열 길이보다 큰 상황 방지
        top_idx = np.argpartition(-scores, kth=min(n_items, len(scores)-1))[:n_items]
        
        # 후보만 실제 점수로 정렬
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        top_scores = scores[top_idx]

        # 결과를 DataFrame으로 반환
        return pd.DataFrame({"item_id": top_idx.astype(int), "score": top_scores.astype(float)})