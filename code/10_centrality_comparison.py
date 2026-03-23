# ============================================================
# 10_centrality_comparison.py
# H3 네트워크 중심성 지표 비교
#
# H3 가설: 문재인 정부에서 공무원 개혁/인사 관련 키워드의
#           네트워크 내 중심성이 박근혜 정부 대비 증가
#
# 문제: 기존 코드(06_cooccurrence.py)는 PMI 네트워크 시각화에
#       집중하였고, 정부별 중심성 지표 정량 비교가 누락됨.
#
# 이 파일에서 수행하는 분석:
#   1. 정부별 PMI 공출현 네트워크 구성
#   2. 4가지 중심성 지표 산출
#      - degree centrality  (연결 중심성)
#      - betweenness        (매개 중심성)
#      - closeness          (근접 중심성)
#      - eigenvector        (위세 중심성)
#   3. 정부별·키워드별 중심성 비교표 저장
#   4. 핵심 키워드(H3 관련) 중심성 변화 시각화
#   5. 통계 검정: Mann-Whitney U test (분포 차이)
#
# 출력:
#   - centrality_by_government.csv     : 정부별 전체 중심성 테이블
#   - centrality_h3_keywords.csv       : H3 핵심 키워드 중심성 비교
#   - centrality_delta.csv             : 정부별 중심성 변화(Δ)
#   - centrality_mannwhitney.csv       : MW U test 결과
#   - fig_centrality_comparison.png    : 중심성 비교 차트
#   - fig_network_pk.png / fig_network_mj.png : 정부별 네트워크
# ============================================================

import os
import math
import pandas as pd
import numpy as np
from collections import Counter
from itertools import combinations
from scipy import stats
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.colors as mcolors
import platform
from stopwords import STOPWORDS

# -------------------------------------------------------
# 0. 설정
# -------------------------------------------------------
if platform.system() == "Windows":
    font_path = "C:/Windows/Fonts/malgun.ttf"
    font_name = fm.FontProperties(fname=font_path).get_name()
    plt.rc("font", family=font_name)
else:
    plt.rc("font", family="NanumGothic")
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = "../result/centrality"
DATA_PATH  = "../datas/preprocessed_2013_2022.csv"
os.makedirs(OUTPUT_DIR, exist_ok=True)

STOPWORDS_SET = set(STOPWORDS(version="base"))

# H3 가설 관련 핵심 키워드 (공무원 개혁·인사 프레임)
H3_KEYWORDS = {
    "공무원", "인사", "개혁", "혁신", "성과",
    "연금", "노동", "조직", "체계", "근무",
    "공정", "소통", "참여", "규제", "공공",
    "청렴", "부패", "비리", "임용", "승진"
}

# 분석에 사용할 전체 키워드 집합 (H3 + 확장 프레임)
ANALYSIS_KEYWORDS = H3_KEYWORDS | {
    "행정", "정책", "예산", "감사", "평가",
    "채용", "복지", "급여", "징계", "노조",
    "디지털", "데이터", "스마트", "효율", "투명",
}

PMI_MIN_CO = 10     # 최소 공동출현 문서 수
PMI_THRESHOLD = 0.3 # 엣지 추가 최소 PMI

# -------------------------------------------------------
# 1. PMI 계산
# -------------------------------------------------------
def compute_pmi(documents, keywords, min_co_cnt=PMI_MIN_CO):
    total = len(documents)
    word_freq = Counter()
    pair_cnt  = Counter()

    for words in documents:
        filtered = words & keywords
        word_freq.update(filtered)
        if len(filtered) >= 2:
            pair_cnt.update(combinations(sorted(filtered), 2))

    pmi_results = {}
    for (w1, w2), cnt in pair_cnt.items():
        if cnt < min_co_cnt:
            continue
        p_xy = cnt / total
        p_x  = word_freq[w1] / total
        p_y  = word_freq[w2] / total
        if p_xy > 0 and p_x > 0 and p_y > 0:
            pmi_results[(w1, w2)] = {"pmi": math.log2(p_xy / (p_x * p_y)), "cnt": cnt}
    return pmi_results


# -------------------------------------------------------
# 2. PMI 네트워크 구성
# -------------------------------------------------------
def build_network(pmi_dict, pmi_thresh=PMI_THRESHOLD):
    G = nx.Graph()
    for (w1, w2), info in pmi_dict.items():
        if info["pmi"] >= pmi_thresh:
            G.add_edge(w1, w2, weight=info["pmi"], cnt=info["cnt"])
    return G


# -------------------------------------------------------
# 3. 중심성 계산
# -------------------------------------------------------
def compute_centrality(G):
    """4종 중심성을 DataFrame으로 반환"""
    if len(G.nodes) == 0:
        return pd.DataFrame()

    deg  = nx.degree_centrality(G)
    bet  = nx.betweenness_centrality(G, weight="weight", normalized=True)
    clo  = nx.closeness_centrality(G, distance="weight")
    try:
        eig = nx.eigenvector_centrality(G, weight="weight", max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        eig = {n: 0.0 for n in G.nodes}

    nodes = list(G.nodes)
    df = pd.DataFrame({
        "키워드": nodes,
        "연결중심성": [round(deg.get(n, 0), 6) for n in nodes],
        "매개중심성": [round(bet.get(n, 0), 6) for n in nodes],
        "근접중심성": [round(clo.get(n, 0), 6) for n in nodes],
        "위세중심성": [round(eig.get(n, 0), 6) for n in nodes],
    })
    return df.sort_values("연결중심성", ascending=False).reset_index(drop=True)


# -------------------------------------------------------
# 4. 네트워크 시각화
# -------------------------------------------------------
def visualize_network(G, gov_name, save_path, h3_kw=H3_KEYWORDS):
    if len(G.nodes) < 3:
        print(f"  {gov_name}: 노드 부족으로 시각화 생략")
        return

    pos = nx.spring_layout(G, seed=42, k=1.2)
    deg = nx.degree_centrality(G)
    sizes  = [5000 * deg.get(n, 0.01) + 200 for n in G.nodes]
    colors = ["#E74C3C" if n in h3_kw else "#3498DB" for n in G.nodes]

    edge_weights = [G[u][v]["weight"] for u, v in G.edges]
    max_w = max(edge_weights) if edge_weights else 1
    widths = [1 + 3 * (w / max_w) for w in edge_weights]

    plt.figure(figsize=(12, 10))
    nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color=colors, alpha=0.85)
    nx.draw_networkx_labels(G, pos, font_size=9,
                            font_family=fm.FontProperties(fname=font_path).get_name()
                            if platform.system() == "Windows" else "NanumGothic")
    nx.draw_networkx_edges(G, pos, width=widths, alpha=0.5, edge_color="gray")

    from matplotlib.patches import Patch
    legend = [
        Patch(color="#E74C3C", label="H3 핵심 키워드"),
        Patch(color="#3498DB", label="일반 키워드"),
    ]
    plt.legend(handles=legend, loc="upper left")
    plt.title(f"{gov_name} 키워드 공출현 네트워크 (PMI ≥ {PMI_THRESHOLD})", fontsize=13)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  네트워크 저장: {save_path}")


# -------------------------------------------------------
# 5. 메인
# -------------------------------------------------------
def main():
    print("데이터 로딩 중...")
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df = df.dropna(subset=["text", "정부"])
    df = df[df["정부"].isin(["박근혜정부", "문재인정부"])]

    gov_data = {}
    for gov in ["박근혜정부", "문재인정부"]:
        sub = df[df["정부"] == gov]
        docs = []
        for text in sub["text"].dropna():
            words = set(text.split()) & ANALYSIS_KEYWORDS
            if len(words) >= 2:
                docs.append(words)
        gov_data[gov] = docs
        print(f"  {gov}: {len(docs):,}건 (분석 키워드 포함)")

    # 중심성 저장용
    centrality_all = {}
    networks       = {}

    for gov, docs in gov_data.items():
        print(f"\n[{gov}] PMI 계산 중...")
        pmi = compute_pmi(docs, ANALYSIS_KEYWORDS)
        G   = build_network(pmi)
        networks[gov] = G
        print(f"  노드: {G.number_of_nodes()}, 엣지: {G.number_of_edges()}")

        cent_df = compute_centrality(G)
        cent_df["정부"] = gov
        centrality_all[gov] = cent_df

        # 네트워크 시각화
        net_path = os.path.join(OUTPUT_DIR,
            "fig_network_pk.png" if gov == "박근혜정부" else "fig_network_mj.png")
        visualize_network(G, gov, net_path)

    # -------------------------------------------------------
    # 6. 전체 중심성 테이블 저장
    # -------------------------------------------------------
    merged = pd.concat(centrality_all.values())
    merged.to_csv(
        os.path.join(OUTPUT_DIR, "centrality_by_government.csv"),
        index=False, encoding="utf-8-sig"
    )
    print("\n저장: centrality_by_government.csv")

    # -------------------------------------------------------
    # 7. H3 핵심 키워드 중심성 정부별 비교표
    # -------------------------------------------------------
    h3_rows = []
    METRICS = ["연결중심성", "매개중심성", "근접중심성", "위세중심성"]
    for kw in sorted(H3_KEYWORDS):
        row = {"키워드": kw}
        for gov in ["박근혜정부", "문재인정부"]:
            sub = centrality_all[gov]
            match = sub[sub["키워드"] == kw]
            for m in METRICS:
                col = f"{m}_{gov[:1]}"  # 박/문 첫글자
                row[col] = match[m].values[0] if len(match) > 0 else 0.0
        # 변화량 (문 - 박)
        for m in METRICS:
            row[f"Δ{m}"] = round(row.get(f"{m}_문", 0) - row.get(f"{m}_박", 0), 6)
        h3_rows.append(row)

    h3_df = pd.DataFrame(h3_rows)
    # 컬럼명 정리
    rename = {}
    for m in METRICS:
        rename[f"{m}_박"] = f"{m}(박근혜)"
        rename[f"{m}_문"] = f"{m}(문재인)"
    h3_df = h3_df.rename(columns=rename)
    h3_df.to_csv(
        os.path.join(OUTPUT_DIR, "centrality_h3_keywords.csv"),
        index=False, encoding="utf-8-sig"
    )
    print("저장: centrality_h3_keywords.csv")

    # -------------------------------------------------------
    # 8. Mann-Whitney U test: 정부별 중심성 분포 차이
    # -------------------------------------------------------
    mw_rows = []
    shared_nodes = (
        set(centrality_all["박근혜정부"]["키워드"]) &
        set(centrality_all["문재인정부"]["키워드"])
    )
    pk_sub = centrality_all["박근혜정부"][
        centrality_all["박근혜정부"]["키워드"].isin(shared_nodes)
    ].set_index("키워드")
    mj_sub = centrality_all["문재인정부"][
        centrality_all["문재인정부"]["키워드"].isin(shared_nodes)
    ].set_index("키워드")

    for m in METRICS:
        u_stat, p_val = stats.mannwhitneyu(
            pk_sub[m].values, mj_sub[m].values, alternative="two-sided"
        )
        mw_rows.append({
            "중심성_지표": m,
            "U_statistic": round(u_stat, 2),
            "p_value": round(p_val, 6),
            "유의미(p<0.05)": "YES" if p_val < 0.05 else "NO",
            "평균_박근혜": round(pk_sub[m].mean(), 6),
            "평균_문재인": round(mj_sub[m].mean(), 6),
            "Δ평균(문-박)": round(mj_sub[m].mean() - pk_sub[m].mean(), 6),
        })

    mw_df = pd.DataFrame(mw_rows)
    mw_df.to_csv(
        os.path.join(OUTPUT_DIR, "centrality_mannwhitney.csv"),
        index=False, encoding="utf-8-sig"
    )
    print("저장: centrality_mannwhitney.csv")
    print("\n[ 중심성 분포 Mann-Whitney U test 결과 ]")
    print(mw_df.to_string(index=False))

    # -------------------------------------------------------
    # 9. 시각화: H3 키워드 중심성 변화 (Δ 막대 차트)
    # -------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for ax, m in zip(axes.flatten(), METRICS):
        pk_col = f"{m}(박근혜)"
        mj_col = f"{m}(문재인)"
        if pk_col not in h3_df.columns:
            continue
        plot_df = h3_df[["키워드", pk_col, mj_col]].copy()
        plot_df = plot_df.sort_values(mj_col, ascending=True)
        x = np.arange(len(plot_df))
        width = 0.35
        ax.barh(x - width/2, plot_df[pk_col], width, label="박근혜정부", color="#4C72B0")
        ax.barh(x + width/2, plot_df[mj_col], width, label="문재인정부", color="#DD8452")
        ax.set_yticks(x)
        ax.set_yticklabels(plot_df["키워드"])
        ax.set_title(m)
        ax.set_xlabel("중심성 값")
        ax.legend(fontsize=8)
        ax.grid(axis="x", alpha=0.3)

    plt.suptitle("H3 핵심 키워드 정부별 중심성 비교", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_centrality_comparison.png"), dpi=300)
    plt.close()
    print("저장: fig_centrality_comparison.png")

    # -------------------------------------------------------
    # 10. 전체 중심성 δ 상위 키워드 (변화 폭 큰 것)
    # -------------------------------------------------------
    delta_rows = []
    for kw in shared_nodes:
        row = {"키워드": kw}
        for m in METRICS:
            pk_val = pk_sub.loc[kw, m] if kw in pk_sub.index else 0
            mj_val = mj_sub.loc[kw, m] if kw in mj_sub.index else 0
            row[f"Δ{m}"] = round(mj_val - pk_val, 6)
        delta_rows.append(row)

    delta_df = pd.DataFrame(delta_rows)
    delta_df["Δ종합"] = delta_df[[f"Δ{m}" for m in METRICS]].mean(axis=1)
    delta_df = delta_df.sort_values("Δ종합", ascending=False)
    delta_df.to_csv(
        os.path.join(OUTPUT_DIR, "centrality_delta.csv"),
        index=False, encoding="utf-8-sig"
    )
    print("저장: centrality_delta.csv")

    print("\n중심성 증가 상위 10 키워드 (문재인 > 박근혜):")
    print(delta_df.head(10)[["키워드", "Δ종합", "Δ연결중심성", "Δ매개중심성"]].to_string(index=False))

    print("\n중심성 감소 상위 10 키워드 (박근혜 > 문재인):")
    print(delta_df.tail(10)[["키워드", "Δ종합", "Δ연결중심성", "Δ매개중심성"]].to_string(index=False))

    print("\n[완료] 10_centrality_comparison.py 실행 완료")
    print(f"결과 디렉토리: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
