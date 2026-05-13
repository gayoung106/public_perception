from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "manuscript" / "revised" / "final_versions" / "appendix_tables.md"

APPENDIX_TOP_N = 30
PAIRWISE_TOP_N = 20

# 논문 부록 표시용 제외어. 원 CSV와 분석 결과는 보존하고, 부록 표에서만
# 공직사회 담론 해석과 직접 관련성이 낮은 인명·국가정상·사건성 고유명사를 제외한다.
APPENDIX_NOISE_KEYWORDS = {
    "김희옥",
    "마크롱",
    "류재복",
    "김성태",
    "기무사",
    "방북",
    "남북정상회담",
    "텍스트",
    "문재인",
    "박근혜",
    "트럼프",
    "김정은",
    "바이든",
    "윤석열",
    "이재명",
    "안철수",
    "홍준표",
    "황교안",
    "최순실",
}

TOPIC_LABELS = {
    0: "행정운영·인사관리",
    1: "정책·규제·재정 이슈",
    2: "생활·경험 관련 보도 맥락",
    3: "산업혁신·플랫폼·경쟁력",
    4: "코로나19·방역·보건",
    5: "성과·전략·안보성 맥락",
    6: "정책비판·개혁·책임성",
    7: "교육·인재·역량",
    8: "일자리·노동·고용",
    9: "정보·절차·조직운영",
}


def read_csv(relative_path: str) -> pd.DataFrame:
    return pd.read_csv(PROJECT_ROOT / relative_path)


def fmt_val(value):
    if pd.isna(value):
        return ""
    if isinstance(value, bool):
        return "예" if value else "아니오"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if abs(value) >= 1000:
            return f"{value:,.0f}"
        if abs(value) >= 100:
            return f"{value:.2f}"
        if abs(value) >= 1:
            return f"{value:.3f}"
        return f"{value:.6f}"
    return str(value)


def md_table(df: pd.DataFrame, columns=None, rename=None, max_rows=None) -> str:
    if columns is not None:
        df = df[columns].copy()
    else:
        df = df.copy()
    if max_rows is not None:
        df = df.head(max_rows)
    if rename:
        df = df.rename(columns=rename)

    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        vals = [fmt_val(row[col]).replace("|", "/") for col in cols]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def append_section(parts, title: str, source: str, note: str, table: str):
    parts.append(f"## {title}\n")
    parts.append(f"출처: `{source}`. {note}\n")
    parts.append(table)


def appendix_filter_keywords(df: pd.DataFrame, keyword_col: str = "keyword") -> pd.DataFrame:
    """Filter only the displayed appendix table, not the underlying analysis output."""
    if keyword_col not in df.columns:
        return df.copy()
    return df[~df[keyword_col].astype(str).isin(APPENDIX_NOISE_KEYWORDS)].copy()


def take_top_by_group(df: pd.DataFrame, group_col: str, rank_col: str, n: int) -> pd.DataFrame:
    return (
        df.sort_values([group_col, rank_col])
        .groupby(group_col, group_keys=False)
        .head(n)
        .copy()
    )


def add_lda_topic_context(df: pd.DataFrame) -> pd.DataFrame:
    topic_words = read_csv("results/three_period/lda_topic_words.csv")
    top_words = (
        topic_words.sort_values(["topic", "rank"])
        .groupby("topic")["keyword"]
        .apply(lambda words: ", ".join(words.head(8).astype(str)))
        .to_dict()
    )
    out = df.copy()
    out["topic_label"] = out["topic"].map(TOPIC_LABELS)
    out["top_keywords"] = out["topic"].map(top_words)
    return out


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    parts = [
        "# 논문 부록 표\n",
        (
            "이 파일은 코로나 robustness 분석, 3시기 temporal split 분석, "
            "concept cluster analysis 결과를 논문 부록 표 형식으로 정리한 것이다. "
            "모든 표는 언론 기사에 나타난 공직사회 관련 담론의 가시성과 결합 구조를 보여주는 보조 자료이며, "
            "실제 공무원의 인식·경험·태도 또는 조직문화 상태를 직접 측정한 결과로 해석하지 않는다.\n"
        ),
    ]

    source = "results/covid_filtered/tfidf_original_vs_covid_filtered_top100.csv"
    df = read_csv(source).sort_values("rank_original")
    append_section(
        parts,
        "Appendix Table 6-1. 코로나 제거 TF-IDF vs. 원분석 상위 100개 키워드 비교",
        source,
        f"Retained는 원분석 top100 키워드가 코로나 관련 어휘 제거 후에도 top100에 남아 있는지를 의미한다. 지면 제약을 고려하여 부록 본문에는 원분석 순위 기준 상위 {APPENDIX_TOP_N}개만 제시하고, 전체 top100은 원 CSV에 보존하였다.",
        md_table(
            df,
            [
                "rank_original",
                "keyword",
                "tfidf_park_original",
                "tfidf_moon_original",
                "importance_original",
                "rank_covid_filtered",
                "tfidf_park_covid_filtered",
                "tfidf_moon_covid_filtered",
                "importance_covid_filtered",
                "retained_in_top_n",
                "rank_change_filtered_minus_original",
            ],
            {
                "rank_original": "원분석 순위",
                "keyword": "키워드",
                "tfidf_park_original": "박근혜 TF-IDF(원분석)",
                "tfidf_moon_original": "문재인 TF-IDF(원분석)",
                "importance_original": "원분석 중요도",
                "rank_covid_filtered": "코로나 제거 순위",
                "tfidf_park_covid_filtered": "박근혜 TF-IDF(코로나 제거)",
                "tfidf_moon_covid_filtered": "문재인 TF-IDF(코로나 제거)",
                "importance_covid_filtered": "코로나 제거 중요도",
                "retained_in_top_n": "Top100 유지",
                "rank_change_filtered_minus_original": "순위 변화",
            },
            max_rows=APPENDIX_TOP_N,
        ),
    )

    source = "results/covid_filtered/log_odds_original_vs_covid_filtered_top100.csv"
    df = appendix_filter_keywords(read_csv(source)).sort_values(["rank_original", "rank_covid_filtered"], na_position="last")
    append_section(
        parts,
        "Appendix Table 6-2. 코로나 제거 log-odds vs. 원분석 상위 100개 키워드 비교",
        source,
        f"Direction stable은 코로나 관련 어휘 제거 후에도 키워드의 정부 시기 방향성이 유지되는지를 의미한다. 학술 부록의 해석 가능성을 높이기 위해 인명·시사 사건성 고유명사 등 공직사회 담론과 직접 관련성이 낮은 표시 노이즈를 제외하고 상위 {APPENDIX_TOP_N}개를 제시하였다. 제외는 부록 표시용이며 원 CSV는 변경하지 않았다.",
        md_table(
            df,
            [
                "rank_original",
                "keyword",
                "direction_original",
                "log_odds_z_original",
                "rank_covid_filtered",
                "direction_covid_filtered",
                "log_odds_z_covid_filtered",
                "retained_in_top_n",
                "direction_stable",
                "rank_change_filtered_minus_original",
            ],
            {
                "rank_original": "원분석 순위",
                "keyword": "키워드",
                "direction_original": "원분석 방향",
                "log_odds_z_original": "원분석 z",
                "rank_covid_filtered": "코로나 제거 순위",
                "direction_covid_filtered": "코로나 제거 방향",
                "log_odds_z_covid_filtered": "코로나 제거 z",
                "retained_in_top_n": "Top100 유지",
                "direction_stable": "방향성 유지",
                "rank_change_filtered_minus_original": "순위 변화",
            },
            max_rows=APPENDIX_TOP_N,
        ),
    )

    source = "results/covid_filtered/robustness_stability_summary.csv"
    df = read_csv(source)
    append_section(
        parts,
        "Appendix Table 6-3. 코로나 제거 강건성 요약",
        source,
        "수치는 코로나 관련 lexical shock을 제거한 뒤 주요 결과가 얼마나 유지되는지를 요약한다.",
        md_table(
            df,
            rename={
                "method": "분석 방법",
                "comparison": "비교",
                "stability_metric": "안정성 지표",
                "value": "값",
            },
        ),
    )

    source = "results/covid_filtered/covid_token_stopword_audit.csv"
    df = read_csv(source)
    added = df[df["added_by_covid_filter"]].copy().sort_values("token")
    already = df[df["already_in_base_stopwords"]].copy().sort_values("token")
    parts.append("\n## Appendix Table 6-4. COVID_TOKENS 목록 및 기존 불용어 중복 점검\n")
    parts.append(
        f"출처: `{source}`. 전체 COVID token audit는 {len(df)}개 항목이며, "
        f"이 중 코로나 robustness 필터에서 추가 제거된 어휘는 {len(added)}개, "
        f"기존 base stopwords에 이미 포함된 어휘는 {len(already)}개이다.\n"
    )
    parts.append("### Appendix Table 6-4A. 코로나 robustness 필터에서 추가 제거된 어휘\n")
    parts.append(
        md_table(
            added,
            ["token", "already_in_base_stopwords", "added_by_covid_filter"],
            {
                "token": "어휘",
                "already_in_base_stopwords": "기존 base stopwords 포함",
                "added_by_covid_filter": "COVID filter 추가 제거",
            },
        )
    )
    parts.append("\n### Appendix Table 6-4B. 기존 base stopwords에 이미 포함된 코로나 관련 어휘\n")
    parts.append(
        md_table(
            already,
            ["token", "already_in_base_stopwords", "added_by_covid_filter"],
            {
                "token": "어휘",
                "already_in_base_stopwords": "기존 base stopwords 포함",
                "added_by_covid_filter": "COVID filter 추가 제거",
            },
        )
    )

    source = "results/three_period/article_count_by_period.csv"
    df = read_csv(source)
    append_section(
        parts,
        "Appendix Table 7-1. 시기별 기사 수 현황",
        source,
        "",
        md_table(
            df,
            ["period_label", "period", "article_count"],
            {"period_label": "시기", "period": "시기 ID", "article_count": "기사 수"},
        ),
    )

    source = "results/three_period/tfidf_top_keywords_by_period.csv"
    df = read_csv(source)
    wide = df.pivot_table(index="rank", columns="period_label", values="keyword", aggfunc="first").reset_index().sort_values("rank")
    for label in df["period_label"].unique():
        sub = df[df["period_label"] == label].set_index("rank")
        wide[label] = wide["rank"].map(
            lambda rank: f"{sub.loc[rank, 'keyword']} ({sub.loc[rank, 'tfidf']:.4f})" if rank in sub.index else ""
        )
    append_section(
        parts,
        "Appendix Table 7-2. 3시기 TF-IDF 상위 키워드 비교 표",
        source,
        f"각 셀은 `키워드 (TF-IDF)` 형식이다. 지면 제약을 고려하여 시기별 상위 {APPENDIX_TOP_N}개를 제시하고, 전체 top100은 원 CSV에 보존하였다.",
        md_table(wide, rename={"rank": "순위"}, max_rows=APPENDIX_TOP_N),
    )

    source = "results/three_period/log_odds_pairwise_top_keywords.csv"
    df = appendix_filter_keywords(read_csv(source)).sort_values(["comparison", "rank_abs_z"])
    df = take_top_by_group(df, "comparison", "rank_abs_z", PAIRWISE_TOP_N)
    append_section(
        parts,
        "Appendix Table 7-3. 3시기 log-odds 쌍별 비교 표",
        source,
        f"Direction은 해당 키워드가 상대적으로 강화된 시기를 의미한다. 학술 부록의 해석 가능성을 높이기 위해 인명·시사 사건성 고유명사 등 표시 노이즈를 제외하고, 세 비교쌍별 상위 {PAIRWISE_TOP_N}개 차별 키워드를 제시하였다. 제외는 부록 표시용이며 원 CSV는 변경하지 않았다.",
        md_table(
            df,
            ["comparison", "interpretation", "rank_abs_z", "keyword", "direction", "count_a", "count_b", "log_odds_z_b_vs_a"],
            {
                "comparison": "비교",
                "interpretation": "해석",
                "rank_abs_z": "순위",
                "keyword": "키워드",
                "direction": "강화 시기",
                "count_a": "A 시기 빈도",
                "count_b": "B 시기 빈도",
                "log_odds_z_b_vs_a": "Log-odds z(B vs A)",
            },
        ),
    )

    source = "results/three_period/lda_topic_period_comparison.csv"
    df = add_lda_topic_context(read_csv(source)).sort_values("topic")
    append_section(
        parts,
        "Appendix Table 7-4. 3시기 LDA 토픽 비중 비교 표",
        source,
        "토픽 비중은 각 시기 문서의 평균 토픽 분포를 의미한다. 토픽 번호만으로는 해석 맥락이 부족하므로, 상위 키워드와 부록용 해석 라벨을 함께 제시하였다. 라벨은 독립 변수가 아니라 독자의 이해를 돕기 위한 요약이다.",
        md_table(
            df,
            [
                "topic",
                "topic_label",
                "top_keywords",
                "park_2013_2017",
                "moon_pre_covid_2017_2019",
                "moon_post_covid_2020_2022",
                "government_change_pre_covid",
                "covid_period_shift",
                "cumulative_change",
                "dominant_effect",
            ],
            {
                "topic": "토픽",
                "topic_label": "부록용 해석 라벨",
                "top_keywords": "상위 키워드",
                "park_2013_2017": "박근혜 정부",
                "moon_pre_covid_2017_2019": "문재인 전기",
                "moon_post_covid_2020_2022": "문재인 후기",
                "government_change_pre_covid": "정부교체 선행 변화",
                "covid_period_shift": "코로나 후기 변화",
                "cumulative_change": "누적 변화",
                "dominant_effect": "지배적 변화 유형",
            },
        ),
    )

    decomp_columns = [
        "keyword",
        "effect_pattern",
        "tfidf_delta_government_change_pre_covid",
        "tfidf_delta_covid_period_shift",
        "tfidf_delta_cumulative_change",
        "log_odds_z_government_change_pre_covid",
        "log_odds_z_covid_period_shift",
        "log_odds_z_cumulative_change",
    ]
    decomp_rename = {
        "keyword": "키워드",
        "effect_pattern": "효과 유형",
        "tfidf_delta_government_change_pre_covid": "TF-IDF 정부교체 선행 변화",
        "tfidf_delta_covid_period_shift": "TF-IDF 코로나 후기 변화",
        "tfidf_delta_cumulative_change": "TF-IDF 누적 변화",
        "log_odds_z_government_change_pre_covid": "Log-odds 정부교체 선행 변화",
        "log_odds_z_covid_period_shift": "Log-odds 코로나 후기 변화",
        "log_odds_z_cumulative_change": "Log-odds 누적 변화",
    }
    source = "results/three_period/keyword_effect_decomposition.csv"
    df = appendix_filter_keywords(read_csv(source)).sort_values("absolute_cumulative_tfidf_change", ascending=False).head(APPENDIX_TOP_N)
    append_section(
        parts,
        "Appendix Table 7-5. 키워드 효과 분해 표",
        source,
        f"대용량 전체 파일에서 누적 TF-IDF 변화 절대값 기준 상위 {APPENDIX_TOP_N}개를 부록 표로 제시한다. 표시 노이즈는 제외하였으며, 전체 264,198개 키워드 결과는 원 CSV를 참조한다.",
        md_table(df, decomp_columns, decomp_rename),
    )

    source = "results/three_period/q1_pre_covid_change_keywords.csv"
    df = appendix_filter_keywords(read_csv(source)).head(PAIRWISE_TOP_N)
    append_section(
        parts,
        "Appendix Table 7-5A. 코로나 이전부터 나타난 변화: 정부교체 선행 변화 키워드",
        source,
        f"표시 노이즈를 제외한 상위 {PAIRWISE_TOP_N}개를 제시한다.",
        md_table(df, decomp_columns, decomp_rename),
    )

    source = "results/three_period/q2_post_covid_specific_keywords.csv"
    df = appendix_filter_keywords(read_csv(source)).head(PAIRWISE_TOP_N)
    append_section(
        parts,
        "Appendix Table 7-5B. 코로나 후기에만 나타난 급변: post-COVID specific keywords",
        source,
        f"표시 노이즈를 제외한 상위 {PAIRWISE_TOP_N}개를 제시한다.",
        md_table(df, decomp_columns, decomp_rename),
    )

    source = "results/concept_clusters/cluster_period_comparison.csv"
    df = read_csv(source)
    append_section(
        parts,
        "Appendix Table 7-6. 개념군 3시기 정규화 빈도 표",
        source,
        "정규화 빈도는 10,000 tokens당 출현 빈도이다.",
        md_table(
            df,
            [
                "label_ko",
                "cluster_id",
                "park_2013_2017",
                "moon_pre_covid_2017_2019",
                "moon_post_covid_2020_2022",
                "government_change_pre_covid",
                "covid_period_shift",
                "cumulative_change",
                "dominant_effect",
                "pre_covid_increase",
                "post_covid_increase",
                "post_covid_only_surge",
            ],
            {
                "label_ko": "개념군",
                "cluster_id": "개념군 ID",
                "park_2013_2017": "박근혜 정부",
                "moon_pre_covid_2017_2019": "문재인 전기",
                "moon_post_covid_2020_2022": "문재인 후기",
                "government_change_pre_covid": "정부교체 선행 변화",
                "covid_period_shift": "코로나 후기 변화",
                "cumulative_change": "누적 변화",
                "dominant_effect": "지배적 변화 유형",
                "pre_covid_increase": "코로나 이전 증가",
                "post_covid_increase": "코로나 이후 증가",
                "post_covid_only_surge": "코로나 이후 급증",
            },
        ),
    )

    source = "results/concept_clusters/concept_clusters_dictionary.csv"
    df = read_csv(source).sort_values(["cluster_id", "term"])
    grouped = df.groupby(["cluster_id", "label_ko"])["term"].apply(lambda terms: ", ".join(map(str, terms))).reset_index()
    append_section(
        parts,
        "Appendix Table 7-7. 개념군 사전 전체 목록",
        source,
        "각 개념군별 dictionary term 전체를 제시한다.",
        md_table(grouped, ["label_ko", "cluster_id", "term"], {"label_ko": "개념군", "cluster_id": "개념군 ID", "term": "사전 어휘"}),
    )
    parts.append(
        "\n주: term 단위 long format과 구성 논리는 원 CSV에 보존되어 있으며, "
        "지면 제약을 고려하여 본 부록에는 개념군별 통합 목록만 제시한다.\n"
    )

    OUT.write_text("\n\n".join(parts), encoding="utf-8")
    print(OUT)
    print(f"COVID filter 추가 제거 어휘: {len(added)}개")


if __name__ == "__main__":
    main()
