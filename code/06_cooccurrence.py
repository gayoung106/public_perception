# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import networkx as nx
import os
from itertools import combinations
from collections import Counter
import platform
from stopwords import STOPWORDS

# ===============================
# 1. 환경 설정
# ===============================
if platform.system() == "Windows":
    font_path = "C:/Windows/Fonts/malgun.ttf"
    font_name = fm.FontProperties(fname=font_path).get_name()
    plt.rc("font", family=font_name)
else:
    plt.rc("font", family="AppleGothic")

plt.rcParams["axes.unicode_minus"] = False
STOPWORDS_LIST = set(STOPWORDS())

output_dir = "../result"
os.makedirs(output_dir, exist_ok=True)

# ===============================
# 2. 공출현 네트워크 함수 (정직한 버전)
# ===============================
def build_cooccurrence_network(
    df,
    gov_name,
    top_n_words=40,          # 시각화용 노드 수
    min_edge_ratio=0.002    # 전체 문서 대비 0.2% 이상 공출현
):
    print(f"\n📌 담론 네트워크 분석 시작: {gov_name}")

    # ---------------------------
    # (1) 정부별 전체 데이터 사용 (샘플링 제거)
    # ---------------------------
    target_df = df[df["정부"] == gov_name].copy()

    if target_df.empty:
        print("⚠ 데이터 없음")
        return

    # ---------------------------
    # (2) 문서별 단어 집합 구성
    # ---------------------------
    documents = []
    all_words = []

    for doc in target_df["text"].dropna():
        words = {
            w for w in str(doc).split()
            if len(w) > 1 and w not in STOPWORDS_LIST
        }
        if words:
            documents.append(words)
            all_words.extend(words)

    total_docs = len(documents)
    if total_docs == 0:
        print("⚠ 유효 문서 없음")
        return

    # ---------------------------
    # (3) 전체 데이터 기준 빈도 상위 단어 선택
    #     (⚠ 분석 개입 아님, 시각화 제한용)
    # ---------------------------
    top_words = {
        w for w, _ in Counter(all_words).most_common(top_n_words)
    }

    # ---------------------------
    # (4) 공출현 계산 (전체 문서 기준)
    # ---------------------------
    pair_counter = Counter()

    for words in documents:
        filtered = sorted(words & top_words)
        if len(filtered) >= 2:
            pair_counter.update(combinations(filtered, 2))

    # ---------------------------
    # (5) 공출현 비율 기준 필터링
    # ---------------------------
    edges = []
    for (w1, w2), count in pair_counter.items():
        ratio = count / total_docs
        if ratio >= min_edge_ratio:
            edges.append((w1, w2, ratio))

    if not edges:
        print("⚠ 유의미한 공출현 없음")
        return

    # ---------------------------
    # (6) 그래프 생성
    # ---------------------------
    G = nx.Graph()
    for w1, w2, weight in edges:
        G.add_edge(w1, w2, weight=weight)

    G.remove_nodes_from(list(nx.isolates(G)))

    if G.number_of_nodes() == 0:
        print("⚠ 네트워크 생성 실패")
        return

    # ---------------------------
    # (7) 중심성 계산
    # ---------------------------
    centrality = nx.degree_centrality(G)

    # ===============================
    # 3. 시각화
    # ===============================
    plt.figure(figsize=(16, 14))

    pos = nx.spring_layout(
        G,
        k=1.2,
        iterations=250,
        seed=42
    )

    # 에지
    weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_w = max(weights)
    edge_widths = [(w / max_w) * 4 for w in weights]

    nx.draw_networkx_edges(
        G,
        pos,
        width=edge_widths,
        alpha=0.25,
        edge_color="gray"
    )

    # 노드
    node_sizes = [centrality[n] * 18000 for n in G.nodes()]
    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=node_sizes,
        node_color="#E3F2FD",
        edgecolors="#1E88E5",
        linewidths=1.5,
        alpha=0.95
    )

    # 라벨
    for node, (x, y) in pos.items():
        font_size = 11 + centrality[node] * 28
        plt.text(
            x, y, node,
            fontsize=font_size,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.65,
                pad=0.25
            )
        )

    plt.title(
        f"담론 네트워크 분석: {gov_name}",
        fontsize=24,
        fontweight="bold",
        pad=30
    )
    plt.axis("off")

    save_path = os.path.join(
        output_dir,
        f"cooccurrence_network_{gov_name}.png"
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ 저장 완료: {save_path}")
    print("🔑 상위 중심 단어:",
          sorted(centrality, key=centrality.get, reverse=True)[:10])

# ===============================
# 4. 실행부
# ===============================
if __name__ == "__main__":
    df_path = "../datas/preprocessed_2013_2022.csv"

    if not os.path.exists(df_path):
        raise FileNotFoundError("❌ 데이터 파일을 찾을 수 없습니다.")

    df = pd.read_csv(df_path, encoding="utf-8-sig")

    build_cooccurrence_network(df, "박근혜정부")
    build_cooccurrence_network(df, "문재인정부")
