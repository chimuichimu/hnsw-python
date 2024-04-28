import math
from typing import Set, List

import numpy as np


class Node:
    """HNSWのグラフを構成するノードを表すクラス"""

    def __init__(self, vector: np.ndarray, layer_idx: int):
        """
        Args:
            vector (np.ndarray): ノードのベクトル値
            layer_idx (int): ノードが属するレイヤーのインデックス
        """
        self.vector = vector
        self.layer_idx = layer_idx
        self.neighborhood: Set[Node] = set()  # エッジが張られているノードの集合

    def add_neighborhood(self, q):
        if self.layer_idx == q.layer_idx:
            self.neighborhood.add(q)


class HNSW:
    """Hierarchical Navigable Small World (HNSW) のグラフ構造を保持するクラス"""

    def __init__(
        self, m_conn: int, m_conn_max: int, ef_construction: int, ml: float
    ) -> None:
        """
        Args:
            m_conn (int): ノードを新しく追加するとき、エッジをいくつ張るか
            m_conn_max (int): 一つのレイヤーあたりに各ノードが保持できるエッジの最大数
            ef_construction (int): hoge
            ml (float): ノードが属するレイヤーを決定する時の正規化パラメータ
        """
        self.nodes: Set[Node] = set()
        self.highest_layer_num: int = 0
        self.entry_point: Node = None
        self.m_conn = m_conn
        self.m_conn_max = m_conn_max
        self.ef_construction = ef_construction
        self.ml = ml

    def _calc_similarity(self, a: Node, b: Node) -> float:
        """
        2点のノードのベクトル間のcos類似度を計算する

        Args:
            a (Node): ノードA
            b (Node): ノードB

        Returns:
            float: ノードAB間のcos類似度
        """
        v1 = a.vector
        v2 = b.vector
        return calc_cos_similarity(v1, v2)

    def _select_neighbors(self, q: Node, candidates: Set[Node], m: int) -> Set[Node]:
        """
        与えられた候補の中から、最もクエリと類似するノードの上位 m 個を返す

        Args:
            q (Node): 探索対象のクエリ
            candidates (Set[Node]): 探索対象の候補ノードの集合
            m (int): 類似するノード上位何個を返すか

        Returns:
            Set[Node]: 近傍ノードの集合
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

    def knn_search(self, vector: np.ndarray, k: int, ef: int) -> List[Node]:
        """
        作成したインデックスを基に K 近傍探索を行う

        Args:
            vector (np.ndarray): 探索対象のクエリ
            k (int): 類似する上位 k 番目まで取得するか
            ef (int) : 探索対象とする候補数
        """
        q = Node(vector, 0)
        ep = self.entry_point

        for lc in range(self.highest_layer_num, 0, -1):
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
        # クエリのノードを最大どのレイヤーまで追加するかを決定
        l_max = math.floor(-math.log(np.random.uniform()) * self.ml)

        # 初回のノード登録
        if self.entry_point is None:
            for lc in range(l_max, -1, -1):
                q = Node(vector, lc)
                if lc == l_max:
                    self.entry_point = q
                self.nodes.add(q)
            self.highest_layer_num = l_max
            return

        # 初回以降のノード登録
        ep = self.entry_point
        candidates = {}

        for lc in range(self.highest_layer_num, l_max, -1):
            q = Node(vector, lc)
            candidates = self._search_layer(q, ep, 1)
            ep = max(candidates, key=lambda x: self._calc_similarity(q, x))

        for lc in range(min(self.highest_layer_num, l_max), -1, -1):
            # ノードの登録
            q = Node(vector, lc)
            self.nodes.add(q)

            # レイヤー内でのエッジを張るノードを選択
            candidates = self._search_layer(q, ep, self.ef_construction)
            neighbors = self._select_neighbors(q, candidates, self.m_conn)

            # エッジを登録
            for e in neighbors:
                # クエリと近傍に双方向のエッジを作成
                q.add_neighborhood(e)
                e.add_neighborhood(q)

                # エッジ数が上限を超える場合はエッジを絞る
                if len(e.neighborhood) > self.m_conn_max:
                    e.neighborhood = self._select_neighbors(
                        e, e.neighborhood, self.m_conn_max
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
