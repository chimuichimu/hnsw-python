import math
from typing import Set

import numpy as np


class Node:
    """HNSWのグラフを構成するノードを表すクラス"""

    def __init__(self, vector: np.ndarray, layer_idx: int):
        """
        ノードの初期化

        Args:
            vector (np.ndarray): ノードのベクトル値
            layer_idx (int): ノードが属するレイヤーのインデックス
        """
        self.vector = vector
        self.layer_idx = layer_idx
        self.neighborhood: Set[Node] = set()

    def add_neighborhood(self, q):
        """
        与えられたノードをノードの近傍に追加する

        Args:
            q (Node): 追加するノード
        """
        if self.layer_idx == q.layer_idx:
            self.neighborhood.add(q)


class HNSW:
    """Hierarchical Navigable Small World (HNSW) のグラフ構造を管理するクラス"""

    def __init__(self, M: int, M_max: int, ef_construction: int, mL: float):
        """
        グラフ構造の初期化

        Args:
            M (int): 各ノードが新しく追加される際に接続される近傍ノードの数
            M_max (int): 一つのレイヤーにおけるノードが持つことができるエッジの最大数
            ef_construction (int): 構築中時に探索する近傍ノードの数
            mL (float): ノードが属するレイヤーを決定する時の正規化パラメータ
        """
        self.nodes: Set[Node] = set()
        self.highest_layer_num: int = 0
        self.entry_point: Node = None
        self.M = M
        self.M_max = M_max
        self.ef_construction = ef_construction
        self.mL = mL

    def _calc_similarity(self, a: Node, b: Node) -> float:
        """
         与えられた二つのノード間のコサイン類似度を計算する

        Args:
            a (Node): ノードA
            b (Node): ノードB

        Returns:
            float: ノード間のコサイン類似度
        """
        v1 = a.vector
        v2 = b.vector
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def _select_neighbors(self, q: Node, candidates: Set[Node], m: int) -> Set[Node]:
        """
        与えられた候補の中から、クエリノードに最も類似した上位m個のノードを選択する

        Args:
            q (Node): 探索対象のクエリ
            candidates (Set[Node]): 探索対象の候補ノードの集合
            m (int): 類似するノード上位何個を返すか

        Returns:
            Set[Node]: 選択された近傍ノードの集合
        """
        return set(
            sorted(candidates, key=lambda x: self._calc_similarity(q, x), reverse=True)[
                :m
            ]
        )

    def _search_layer(self, q: Node, ep: Node, ef: int) -> Set[Node]:
        """
        対象のレイヤーで与えられたクエリに類似するノードを探索する

        Args:
            q (Node): 探索対象のクエリ
            ep (Node): 探索の開始点
            ef (int): 返却する類似ノード数

        Returns:
            Set[Node]: 探索によって見つかった近傍ノードの集合
        """
        nodes_visited = set([ep])
        candidates = set([ep])
        neighbors = set([ep])

        while candidates:
            c = max(candidates, key=lambda x: self._calc_similarity(q, x))
            f = min(neighbors, key=lambda x: self._calc_similarity(q, x))

            if self._calc_similarity(q, c) < self._calc_similarity(q, f):
                break

            for e in c.neighborhood:
                if e not in nodes_visited:
                    nodes_visited.add(e)
                    f = min(neighbors, key=lambda x: self._calc_similarity(q, x))
                    if (
                        self._calc_similarity(q, e) > self._calc_similarity(q, f)
                        or len(neighbors) < ef
                    ):
                        candidates.add(e)
                        neighbors.add(e)
                        if len(neighbors) > ef:
                            f = min(
                                neighbors, key=lambda x: self._calc_similarity(q, x)
                            )
                            neighbors.remove(f)

            candidates.remove(c)

        return neighbors

    def knn_search(self, vector: np.ndarray, k: int, ef: int) -> Set[Node]:
        """
        指定されたベクトルに最も類似するノードをk個探索するK近傍探索を行う

        Args:
            vector (np.ndarray): 探索クエリとして使用されるベクトル
            k (int): 返される類似ノードの数
            ef (int): 探索過程で考慮する近傍ノードの最大数

        Returns:
            Set[Node]: クエリベクトルに類似するk個のノード
        """
        q = Node(vector, 0)
        ep = self.entry_point

        for _ in range(self.highest_layer_num, 0, -1):
            candidates = self._search_layer(q, ep, 1)
            ep = max(candidates, key=lambda x: self._calc_similarity(q, x))

        candidates = self._search_layer(q, ep, ef)
        return self._select_neighbors(q, candidates, k)

    def insert(self, vector: np.ndarray) -> None:
        """
        入力されたベクトルから新たにノードをグラフに追加する

        Args:
            vector (np.ndarray): 新たに追加するクエリのもとになるベクトル
        """
        # クエリのノードを追加する最も高位のレイヤーを確率的に決定
        l_max = math.floor(-math.log(np.random.uniform()) * self.mL)

        # 1つ目のノード登録
        if self.entry_point is None:
            for lc in range(l_max, -1, -1):
                # ノードの作成
                q = Node(vector, lc)
                self.nodes.add(q)
                if lc == l_max:
                    self.entry_point = q
            self.highest_layer_num = l_max
            return

        # 2つ目以降のノード登録
        ep = self.entry_point
        candidates = {}

        # 最上位レイヤーから出発し、ノードを登録するレイヤーの開始点まで移動
        for lc in range(self.highest_layer_num, l_max, -1):
            q = Node(vector, lc)
            candidates = self._search_layer(q, ep, 1)
            ep = max(candidates, key=lambda x: self._calc_similarity(q, x))

        # 各レイヤーにクエリのノードを登録し、近傍とのエッジを作成する
        for lc in range(min(self.highest_layer_num, l_max), -1, -1):
            # ノードの作成
            q = Node(vector, lc)
            self.nodes.add(q)

            # レイヤー内でのエッジを張る近傍ノードを決定
            candidates = self._search_layer(q, ep, self.ef_construction)
            neighbors = self._select_neighbors(q, candidates, self.M)

            # エッジの作成
            for e in neighbors:
                # クエリと近傍に双方向のエッジを作成
                q.add_neighborhood(e)
                e.add_neighborhood(q)

                # ノードあたりのエッジ数が上限を超えないようにする
                if len(e.neighborhood) > self.M_max:
                    e.neighborhood = self._select_neighbors(
                        e, e.neighborhood, self.M_max
                    )

        if l_max > self.highest_layer_num:
            self.entry_point = q
            self.highest_layer_num = l_max


# Example of usage
hnsw = HNSW(M=5, Mmax=10, efConstruction=200, mL=1.0)

# Inserting some data points into the HNSW graph
data_points = [
    np.array([0.5, 0.6]),
    np.array([0.1, 0.2]),
    np.array([0.8, 0.9]),
    np.array([0.3, 0.4]),
]
for point in data_points:
    hnsw.insert(point)

# Querying a point to find its k-nearest neighbors
query_point = np.array([0.6, 0.7])
k_neighbors = hnsw.knn_search(query_point, K=3, ef=10)

# Print the query results
print("K-Nearest Neighbors:")
for neighbor in k_neighbors:
    print(hnsw.nodes[neighbor]["data"])
