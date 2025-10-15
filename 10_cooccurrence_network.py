# ============================================================
#   파일명: 10_cooccurrence_network.py
#   목적: 시기별 공출현 단어 네트워크 분석 및 시각화
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from itertools import combinations
from collections import Counter
from konlpy.tag import Okt
import platform

# ================================
# ⚙️ 폰트 설정 (macOS 한글 깨짐 방지)
# ================================
if platform.system() == 'Darwin':  # macOS
    plt.rc('font', family='AppleGothic')
else:  # Windows / Linux 대비
    plt.rc('font', family='Malgun Gothic')

plt.rcParams['axes.unicode_minus'] = False  # 마이너스 깨짐 방지

okt = Okt()

# ------------------------------------------------------------
# 🔹 불용어 리스트 (이전 단계 동일)
# ------------------------------------------------------------
EXCLUDE_WORDS = [
    # 정치 및 기관명
    '정부', '공직자', '공무원', '정책', '국가', '제도',
    '위원회', '부처', '국회', '장관', '대통령', '청와대', '서울시',
    '의원', '기자', '위원', '위원장', '관계자', '단체', '기관', '본부',
    '발표', '회의', '예산', '조사', '보고서', '언론',

    # 정치인 및 정당
    '윤석열', '윤석렬', '김건희', '이재명', '조국', '문재인', '박근혜',
    '이명박', '국민의힘', '민주당', '더불어민주당', '국힘', '혁신당', '민주',
    '진보', '보수', '더불어',

    # 지명 및 지역
    '한국', '대한민국', '서울', '일본', '대만', '지방', '강진군', '전국', '인천',

    # 불필요 연결어 / 기능어
    '관련', '통해', '의견', '논의', '추진', '방안', '지적',
    '확대', '요구', '대상', '중요', '사실', '입장', '시작', '처리',

    # 사건·사고 및 뉴스용어
    '사건', '사안', '피해자', '성폭력', '성희롱', '비서', '비서실',
    '투표', '선거', '대선', '탄핵', '촛불', '법안', '검찰', '심판',

    # 역사/의례/국가보훈
    '전쟁', '호국', '현충일', '독립운동', '독립운동가', '태극기', '유공', '보훈', '희생',

    # 일반 사회어
    '학교', '학생', '아이', '유치원', '부모', '친구', '급식',
    '출산율', '가족', '노인', '주택', '지역', '부담',

    # 기사 작성용 표현
    '대표', '기획', '기분', '결정', '사실', '지난달', '시간', '내년', '사정', '선임', '발생', '신고',

    # 기타 불필요 단어
    '효능', '연구원', '연구소', '사회학', '독자', '평가', '대책', '악성'
]

# ------------------------------------------------------------
# 🔹 명사 추출 함수
# ------------------------------------------------------------
def extract_nouns(text):
    words = okt.nouns(str(text))
    words = [w for w in words if len(w) > 1 and w not in EXCLUDE_WORDS]
    return words

# ------------------------------------------------------------
# 🔹 공출현 네트워크 생성 함수
# ------------------------------------------------------------
def build_cooccurrence_network(file_path, title, output_name):
    print(f"▶ {title} 데이터 처리 중...")

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
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, k=0.4, seed=42)
    node_sizes = [centrality[v] * 3000 for v in G.nodes()]
    edge_weights = [G[u][v]['weight'] * 0.3 for u, v in G.edges()]

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='skyblue', alpha=0.7)
    nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.4)
    nx.draw_networkx_labels(G, pos, font_family='AppleGothic', font_size=10)

    plt.title(f"공출현 네트워크: {title}", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_name, dpi=300)
    plt.show()

    # 결과 저장
    edges_df.to_csv(output_name.replace('.png', '_edges.csv'), index=False, encoding='utf-8-sig')
    central_df.to_csv(output_name.replace('.png', '_centrality.csv'), index=False, encoding='utf-8-sig')

    print(f"✅ 완료: {title} ({len(G.nodes())}개 단어, {len(G.edges())}개 연결)")
    print(f"🔹 중심 단어 상위 10개:")
    print(central_df.head(10))
    print()

# ------------------------------------------------------------
# 🔹 실행 구간
# ------------------------------------------------------------
file_2015_2019 = "../datas/clean_2015_2019.csv"
file_2020_2025 = "../datas/clean_2020_2025.csv"

build_cooccurrence_network(file_2015_2019, "2015–2019", "../datas/network_2015_2019.png")
build_cooccurrence_network(file_2020_2025, "2020–2025", "../datas/network_2020_2025.png")
