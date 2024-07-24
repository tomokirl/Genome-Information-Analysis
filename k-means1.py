import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# データの読み込み 今回は乳がんの診断に関するデータを使用
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
columns = ["ID", "diagnosis"] + [f"feature_{i}" for i in range(1, 31)]
data = pd.read_csv(url, header=None, names=columns)

# 前処理
data = data.drop(columns=["ID"])#ID列の削除
data["diagnosis"] = data["diagnosis"].map({"M": 1, "B": 0})#診断結果の数値化
scaler = StandardScaler()# データの標準化
scaled_data = scaler.fit_transform(data.iloc[:, 1:])  # 数値データのみをスケール

def k_means(X, K, max_iterations=100, tolerance=1e-4):
    # X: データセット (N x d の行列)、K: クラスタの数、max_iterations: 最大反復回数、tolerance: 収束判定の閾値

    #初期のクラスタ重心μkをランダムに選んだデータ点の平均値で初期化
    N, d = X.shape
    centroids = X[np.random.choice(N, K, replace=False)]

    iteration = 0
    while iteration < max_iterations:
        #各データポイントの所属クラスタを更新 (riの更新)
        labels = np.zeros(N, dtype=int)

        for i in range(N):
            # 各データポイントxiに対して最も近いクラスタを選択
            distances = np.linalg.norm(X[i] - centroids, axis=1)
            labels[i] = np.argmin(distances)

        #クラスタ重心μkを更新
        new_centroids = np.zeros((K, d))
        for k in range(K):
            # クラスタkに属するデータポイントを集計
            cluster_points = X[labels == k]
            # クラスタ重心を計算
            if len(cluster_points) > 0:
                new_centroids[k] = np.mean(cluster_points, axis=0)
            else:
                # クラスタが空の場合、ランダムに再初期化
                new_centroids[k] = X[np.random.choice(N)]

        # 収束判定 (クラスタ重心の移動が十分小さい場合)
        if np.all(np.linalg.norm(new_centroids - centroids, axis=1) < tolerance):
            print(f"Converged after {iteration+1} iterations.")
            break

        centroids = new_centroids
        iteration += 1

    return labels, centroids

# K-meansクラスタリングの実行
K = 2
labels, centroids = k_means(scaled_data, K)
data['Cluster'] = labels

# クラスタ中心点の出力
print("クラスタ中心点:\n", centroids)

# 各クラスターの平均を数値データのみで計算
cluster_analysis = data.groupby('Cluster').mean(numeric_only=True)
print(cluster_analysis)

# シルエットスコアの計算
silhouette_avg = silhouette_score(scaled_data, data['Cluster'])
print("シルエットスコア:", silhouette_avg)

# 元のラベルとクラスタリング結果のクロス集計表
crosstab = pd.crosstab(data['diagnosis'], data['Cluster'])
print("クロス集計表:\n", crosstab)

# 可視化　
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='feature_1', y='feature_2', hue='Cluster', palette='viridis', style=data['diagnosis'])
plt.title('K-means Clustering of Breast Cancer Wisconsin Dataset')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
