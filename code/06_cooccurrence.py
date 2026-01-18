import pandas as pd
import os
import math
from itertools import combinations
from collections import Counter

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

# Windows 한글 폰트 설정
font_path = "C:/Windows/Fonts/malgun.ttf"  
font = font_manager.FontProperties(fname=font_path)
rc("font", family=font.get_name())
rc("axes", unicode_minus=False)

# ===============================
# 1. 기본 설정
# ===============================
OUTPUT_DIR = "../result"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===============================
# 2. PMI 계산 함수
# ===============================
def compute_pmi(documents, keywords, min_co_cnt=10):
    """
    documents : list[set[str]]  # 문서별 키워드 집합
    keywords  : set[str]
    min_co_cnt: 최소 공동출현 문서 수
    """
    total_docs = len(documents)
    word_doc_freq = Counter()
    pair_counter = Counter()

    for words in documents:
        filtered = words & keywords
        word_doc_freq.update(filtered)

        if len(filtered) >= 2:
            pair_counter.update(combinations(sorted(filtered), 2))

    pmi_result = {}

    for (w1, w2), co_cnt in pair_counter.items():
        if co_cnt < min_co_cnt:
            continue

        p_xy = co_cnt / total_docs
        p_x = word_doc_freq[w1] / total_docs
        p_y = word_doc_freq[w2] / total_docs

        if p_xy > 0 and p_x > 0 and p_y > 0:
            pmi_value = math.log2(p_xy / (p_x * p_y))
            pmi_result[(w1, w2)] = {
                "pmi": pmi_value,
                "co_cnt": co_cnt
            }

    return pmi_result

# ===============================
# 3. 차별 공출현 테이블 생성
# ===============================
def build_differential_cooccurrence_table(
    text_df,
    keywords,
    min_delta_pmi=0.3,
    min_co_cnt=10,
    top_n=30
):
    print("\n차별 공출현(PMI) 분석 시작")

    docs = {}
    for gov in ["박근혜정부", "문재인정부"]:
        gov_docs = []

        for doc in text_df[text_df["정부"] == gov]["text"].dropna():
            words = set(str(doc).split()) & keywords
            if len(words) >= 2:
                gov_docs.append(words)

        docs[gov] = gov_docs
        print(f" - {gov}: 문서 수 {len(gov_docs)}")

    pmi_pk = compute_pmi(docs["박근혜정부"], keywords, min_co_cnt)
    pmi_mj = compute_pmi(docs["문재인정부"], keywords, min_co_cnt)

    rows = []
    all_pairs = set(pmi_pk.keys()) | set(pmi_mj.keys())

    for (w1, w2) in all_pairs:
        pk = pmi_pk.get((w1, w2), {"pmi": 0, "co_cnt": 0})
        mj = pmi_mj.get((w1, w2), {"pmi": 0, "co_cnt": 0})

        delta = mj["pmi"] - pk["pmi"]

        if abs(delta) >= min_delta_pmi:
            rows.append({
                "키워드1": w1,
                "키워드2": w2,
                "PMI_박근혜": round(pk["pmi"], 3),
                "PMI_문재인": round(mj["pmi"], 3),
                "ΔPMI": round(delta, 3),
                "공동출현_박근혜": pk["co_cnt"],
                "공동출현_문재인": mj["co_cnt"],
                "강화_시기": "문재인정부" if delta > 0 else "박근혜정부"
            })

    df = pd.DataFrame(rows).sort_values("ΔPMI", ascending=False)

    df_final = pd.concat([
        df.head(top_n),
        df.tail(top_n)
    ])

    save_path = os.path.join(
        OUTPUT_DIR,
        "differential_cooccurrence_table.csv"
    )
    df_final.to_csv(save_path, index=False, encoding="utf-8-sig")

    print(f"저장 완료: {save_path}")
    return df_final

# ===============================
# 4. PMI 네트워크 시각화
# ===============================
def plot_differential_pmi_network(
    df,
    min_abs_delta=0.5,
    max_edges=40,
    save_path="../result/figure_pmi_network.png"
):
    df_plot = df[abs(df["ΔPMI"]) >= min_abs_delta]
    df_plot = df_plot.sort_values("ΔPMI", key=abs, ascending=False).head(max_edges)

    G = nx.Graph()

    for _, row in df_plot.iterrows():
        w1 = row["키워드1"]
        w2 = row["키워드2"]
        delta = row["ΔPMI"]

        G.add_edge(
            w1, w2,
            weight=abs(delta),
            style="solid" if delta > 0 else "dashed"
        )

    pos = nx.spring_layout(G, seed=42, k=0.8)

    plt.figure(figsize=(10, 8))

    solid_edges = [(u, v) for u, v, d in G.edges(data=True) if d["style"] == "solid"]
    dashed_edges = [(u, v) for u, v, d in G.edges(data=True) if d["style"] == "dashed"]

    nx.draw_networkx_nodes(
        G, pos,
        node_size=1200,
        node_color="white",
        edgecolors="black"
    )

    # 🔥 여기 핵심 수정
    nx.draw_networkx_labels(
        G,
        pos,
        font_size=10,
        font_family=font.get_name()
    )

    nx.draw_networkx_edges(
        G, pos,
        edgelist=solid_edges,
        width=[G[u][v]["weight"] * 1.5 for u, v in solid_edges],
        style="solid"
    )

    nx.draw_networkx_edges(
        G, pos,
        edgelist=dashed_edges,
        width=[G[u][v]["weight"] * 1.5 for u, v in dashed_edges],
        style="dashed"
    )

    plt.title("Differential Co-occurrence Network Based on PMI", fontsize=12)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Figure saved to: {save_path}")

# ===============================
# 5. 실행부
# ===============================
if __name__ == "__main__":
    text_df = pd.read_csv(
        "../datas/preprocessed_2013_2022.csv",
        encoding="utf-8-sig"
    )

    CORE_FRAME = {
        "공무원", "조직", "성과", "인사", "근무",
        "체계", "연금", "노동", "개혁", "혁신"
    }

    MJ_UP = {
        "공정", "인권", "소통", "참여", "규제",
        "청년", "여성", "공공", "데이터", "디지털"
    }

    PK_UP = {
        "연금", "노조", "경영", "승진", "조직",
        "공무원", "개혁", "노동"
    }

    KEYWORDS = CORE_FRAME | MJ_UP | PK_UP

    df_result = build_differential_cooccurrence_table(
        text_df=text_df,
        keywords=KEYWORDS,
        min_delta_pmi=0.3,
        min_co_cnt=10,
        top_n=30
    )

    plot_differential_pmi_network(
        df_result,
        min_abs_delta=0.5,
        max_edges=40
    )
