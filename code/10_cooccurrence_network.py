# ============================================================
#   파일명: 10_cooccurrence_network.py
#   목적: 시기별 공출현 단어 네트워크 분석 및 시각화
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import networkx as nx
from itertools import combinations
from collections import Counter
from konlpy.tag import Okt
import platform
from stopwords import STOPWORDS

# 한글 폰트 설정
if platform.system() == "Windows":
    plt.rc('font', family='Malgun Gothic')
elif platform.system() == "Darwin":  # macOS
    plt.rc('font', family='AppleGothic')
else:  # Linux or 기타
    plt.rc('font', family='NanumGothic')

# 마이너스 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

okt = Okt()

# ------------------------------------------------------------
# 🔹 명사 추출 함수
# ------------------------------------------------------------
def extract_nouns(text):
    words = okt.nouns(str(text))
    words = [w for w in words if len(w) > 1 and w not in STOPWORDS]
    return words

# ------------------------------------------------------------
# 🔹 공출현 네트워크 생성 함수
# ------------------------------------------------------------
def build_cooccurrence_network(file_path, title, output_name):
    print(f" {title} 데이터 처리 중...")

    df = pd.read_csv(file_path)
    df = df.dropna(subset=['clean_text'])

    # 문서별 명사 추출
    df['nouns'] = df['clean_text'].apply(extract_nouns)

    # 모든 문서에서 단어쌍(조합) 추출
    pairs = []
    for words in df['nouns']:
        pairs += list(combinations(set(words), 2))  # 중복 제거 후 조합 생성

    counter = Counter(pairs)
    edges_df = pd.DataFrame(counter.most_common(200), columns=['단어쌍', '빈도'])
    edges_df[['단어1', '단어2']] = pd.DataFrame(edges_df['단어쌍'].tolist(), index=edges_df.index)

    # 네트워크 생성
    G = nx.Graph()
    for _, row in edges_df.iterrows():
        G.add_edge(row['단어1'], row['단어2'], weight=row['빈도'])

    # 중심성 계산
    centrality = nx.degree_centrality(G)
    central_df = pd.DataFrame(centrality.items(), columns=['단어', '중심성']).sort_values('중심성', ascending=False)

    # 🔹 시각화
    # 🔹 시각화
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, k=0.4, seed=42)

    node_sizes = [centrality[v] * 3000 for v in G.nodes()]
    edge_weights = [G[u][v]['weight'] * 0.3 for u, v in G.edges()]

    # 🔹 중심성 상위 노드만 라벨링
    top_nodes = central_df.head(20)['단어'].tolist()
    labels = {node: node for node in G.nodes() if node in top_nodes}

    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color='#DCEEFF',
        alpha=0.9
    )

    nx.draw_networkx_edges(
        G, pos,
        width=edge_weights,
        alpha=0.35
    )

    nx.draw_networkx_labels(
        G, pos,
        labels=labels,
        font_size=14,
        font_weight='bold',
        font_family=plt.rcParams['font.family'][0],
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.85)
    )

    plt.title(f"공출현 네트워크: {title}", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_name, dpi=300)
    plt.show()

    # 결과 저장
    edges_df.to_csv(output_name.replace('.png', '_edges.csv'), index=False, encoding='utf-8-sig')
    central_df.to_csv(output_name.replace('.png', '_centrality.csv'), index=False, encoding='utf-8-sig')

    print(f"완료: {title} ({len(G.nodes())}개 단어, {len(G.edges())}개 연결)")
    print(f"🔹 중심 단어 상위 10개:")
    print(central_df.head(10))
    print()

# ------------------------------------------------------------
# 🔹 실행 구간
# ------------------------------------------------------------
file_2015_2019 = "../datas/clean_2015_2019.csv"
file_2020_2024 = "../datas/clean_2020_2024.csv"

build_cooccurrence_network(file_2015_2019, "2015–2019", "../datas/network_2015_2019.png")
build_cooccurrence_network(file_2020_2024, "2020–2024", "../datas/network_2020_2024.png")
