"""
Dictionary-based conceptual cluster analysis.

This script adds a concept-level robustness and interpretation layer on top of
the TF-IDF, log-odds, and LDA analyses. It keeps the existing preprocessing and
three-period design, then measures how predefined conceptual clusters change
over time.
"""

from __future__ import annotations

import json
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODE = "concept_clusters_three_period"
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "datas" / "preprocessed_2013_2022.csv"
DICTIONARY_PATH = PROJECT_ROOT / "code" / "concept_clusters.json"
THREE_PERIOD_DIR = PROJECT_ROOT / "results" / "three_period"
OUTPUT_DIR = PROJECT_ROOT / "results" / "concept_clusters"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATE_COL = "날짜"
YEAR_COL = "year"
TEXT_COL = "text"

PARK_PERIOD = "park_2013_2017"
MOON_PRE = "moon_pre_covid_2017_2019"
MOON_POST = "moon_post_covid_2020_2022"
PERIOD_ORDER = [PARK_PERIOD, MOON_PRE, MOON_POST]
PERIOD_LABELS = {
    PARK_PERIOD: "박근혜정부(2013-2017)",
    MOON_PRE: "문재인 전기/코로나 이전(2017-2019)",
    MOON_POST: "문재인 후기/코로나 이후(2020-2022)",
}

PERIOD_DEFINITIONS = [
    (PARK_PERIOD, "2013-02-25", "2017-03-10"),
    (MOON_PRE, "2017-05-10", "2019-12-31"),
    (MOON_POST, "2020-01-01", "2022-05-09"),
]

NORM_PER_TOKENS = 10_000


@dataclass(frozen=True)
class Cluster:
    cluster_id: str
    label_ko: str
    rationale: str
    terms: set[str]


def configure_font() -> None:
    if (Path("C:/Windows/Fonts/malgun.ttf")).exists():
        font_name = fm.FontProperties(fname="C:/Windows/Fonts/malgun.ttf").get_name()
        plt.rc("font", family=font_name)
    else:
        plt.rc("font", family="NanumGothic")
    plt.rcParams["axes.unicode_minus"] = False


def assign_period(date_value: pd.Timestamp) -> str | None:
    for period, start, end in PERIOD_DEFINITIONS:
        if pd.Timestamp(start) <= date_value <= pd.Timestamp(end):
            return period
    return None


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df = df.dropna(subset=[DATE_COL, TEXT_COL]).copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL])
    df["period"] = df[DATE_COL].apply(assign_period)
    df = df.dropna(subset=["period"]).copy()
    df["period"] = pd.Categorical(df["period"], categories=PERIOD_ORDER, ordered=True)
    df[YEAR_COL] = df[DATE_COL].dt.year
    df["month"] = df[DATE_COL].dt.to_period("M").astype(str)
    return df


def load_dictionary() -> tuple[list[Cluster], dict]:
    with open(DICTIONARY_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    clusters = []
    for item in raw["clusters"]:
        terms = {term.lower() for term in item["terms"]}
        clusters.append(
            Cluster(
                cluster_id=item["cluster_id"],
                label_ko=item["label_ko"],
                rationale=item["rationale"],
                terms=terms,
            )
        )
    return clusters, raw


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in str(text).split() if token]


def dictionary_tables(clusters: list[Cluster], raw_dictionary: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    term_to_clusters: dict[str, list[str]] = defaultdict(list)
    for cluster in clusters:
        for term in sorted(cluster.terms):
            rows.append(
                {
                    "cluster_id": cluster.cluster_id,
                    "label_ko": cluster.label_ko,
                    "term": term,
                    "rationale": cluster.rationale,
                }
            )
            term_to_clusters[term].append(cluster.cluster_id)

    dictionary_df = pd.DataFrame(rows)
    overlap_df = pd.DataFrame(
        [
            {
                "term": term,
                "cluster_count": len(cluster_ids),
                "clusters": ";".join(cluster_ids),
                "has_overlap": len(cluster_ids) > 1,
            }
            for term, cluster_ids in sorted(term_to_clusters.items())
        ]
    )

    shutil.copyfile(DICTIONARY_PATH, OUTPUT_DIR / "concept_clusters_dictionary.json")
    dictionary_df.to_csv(
        OUTPUT_DIR / "concept_clusters_dictionary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    overlap_df.to_csv(
        OUTPUT_DIR / "concept_term_overlap_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    with open(OUTPUT_DIR / "concept_clusters_dictionary_pretty.json", "w", encoding="utf-8") as f:
        json.dump(raw_dictionary, f, ensure_ascii=False, indent=2)
    return dictionary_df, overlap_df


def count_cluster_terms(df: pd.DataFrame, clusters: list[Cluster]) -> tuple[pd.DataFrame, pd.DataFrame]:
    cluster_lookup = {cluster.cluster_id: cluster for cluster in clusters}
    term_rows = []
    doc_rows = []

    for row in df[[DATE_COL, YEAR_COL, "month", "period", TEXT_COL]].itertuples(index=False):
        date_value, year, month, period, text = row
        tokens = tokenize(text)
        token_count = len(tokens)
        token_counter = Counter(tokens)

        row_base = {
            DATE_COL: date_value,
            YEAR_COL: year,
            "month": month,
            "period": str(period),
            "total_tokens": token_count,
        }
        doc_cluster_counts = {}

        for cluster_id, cluster in cluster_lookup.items():
            matched_terms = cluster.terms & set(token_counter)
            cluster_count = sum(token_counter[term] for term in matched_terms)
            doc_cluster_counts[cluster_id] = cluster_count
            for term in matched_terms:
                term_rows.append(
                    {
                        "period": str(period),
                        YEAR_COL: year,
                        "month": month,
                        "cluster_id": cluster_id,
                        "label_ko": cluster.label_ko,
                        "term": term,
                        "count": token_counter[term],
                    }
                )

        doc_rows.append({**row_base, **doc_cluster_counts})

    doc_counts = pd.DataFrame(doc_rows)
    term_counts = pd.DataFrame(term_rows)
    if term_counts.empty:
        term_counts = pd.DataFrame(
            columns=["period", YEAR_COL, "month", "cluster_id", "label_ko", "term", "count"]
        )
    return doc_counts, term_counts


def aggregate_cluster_counts(
    doc_counts: pd.DataFrame,
    clusters: list[Cluster],
    group_cols: list[str],
) -> pd.DataFrame:
    rows = []
    for group_key, sub in doc_counts.groupby(group_cols, observed=True):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        base = dict(zip(group_cols, group_key))
        total_tokens = int(sub["total_tokens"].sum())
        article_count = int(len(sub))
        for cluster in clusters:
            cluster_tokens = int(sub[cluster.cluster_id].sum())
            docs_with_cluster = int((sub[cluster.cluster_id] > 0).sum())
            rows.append(
                {
                    **base,
                    "cluster_id": cluster.cluster_id,
                    "label_ko": cluster.label_ko,
                    "article_count": article_count,
                    "total_tokens": total_tokens,
                    "cluster_tokens": cluster_tokens,
                    "docs_with_cluster": docs_with_cluster,
                    "normalized_frequency_per_10k_tokens": (
                        cluster_tokens / total_tokens * NORM_PER_TOKENS if total_tokens else 0
                    ),
                    "article_share": docs_with_cluster / article_count if article_count else 0,
                }
            )
    return pd.DataFrame(rows)


def period_comparison(period_summary: pd.DataFrame) -> pd.DataFrame:
    metric = "normalized_frequency_per_10k_tokens"
    wide = period_summary.pivot(index=["cluster_id", "label_ko"], columns="period", values=metric).reset_index()
    for period in PERIOD_ORDER:
        if period not in wide:
            wide[period] = 0.0
    wide["government_change_pre_covid"] = wide[MOON_PRE] - wide[PARK_PERIOD]
    wide["covid_period_shift"] = wide[MOON_POST] - wide[MOON_PRE]
    wide["cumulative_change"] = wide[MOON_POST] - wide[PARK_PERIOD]
    wide["dominant_effect"] = np.where(
        wide["government_change_pre_covid"].abs() >= wide["covid_period_shift"].abs(),
        "government_change_pre_covid",
        "covid_period_shift",
    )
    wide["pre_covid_increase"] = wide["government_change_pre_covid"] > 0
    wide["post_covid_increase"] = wide["covid_period_shift"] > 0
    wide["post_covid_only_surge"] = (
        (wide["government_change_pre_covid"] <= 0)
        & (wide["covid_period_shift"] > 0)
        & (wide["cumulative_change"] > 0)
    )
    return wide.sort_values("cumulative_change", key=lambda x: x.abs(), ascending=False)


def question_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    lookup = comparison.set_index("cluster_id")

    def row(question: str, cluster_id: str, criterion: str, answer: bool, interpretation: str) -> dict:
        values = lookup.loc[cluster_id]
        return {
            "question": question,
            "cluster_id": cluster_id,
            "label_ko": values["label_ko"],
            "criterion": criterion,
            "answer": "yes" if answer else "no",
            "park": values[PARK_PERIOD],
            "moon_pre_covid": values[MOON_PRE],
            "moon_post_covid": values[MOON_POST],
            "government_change_pre_covid": values["government_change_pre_covid"],
            "covid_period_shift": values["covid_period_shift"],
            "cumulative_change": values["cumulative_change"],
            "interpretation": interpretation,
        }

    fairness = lookup.loc["fairness"]
    performance = lookup.loc["performance_npm"]
    digital = lookup.loc["digital_transformation"]
    crisis = lookup.loc["public_health_crisis"]

    return pd.DataFrame(
        [
            row(
                "공정성 담론은 코로나 이전부터 증가했는가?",
                "fairness",
                "moon_pre_covid > park",
                fairness["government_change_pre_covid"] > 0,
                "문재인 전기 값이 박근혜 시기보다 높으면 정부교체 이후 선행 변화로 해석한다.",
            ),
            row(
                "성과주의 담론은 약화되었는가?",
                "performance_npm",
                "moon_post_covid < park and cumulative_change < 0",
                (performance[MOON_POST] < performance[PARK_PERIOD])
                and (performance["cumulative_change"] < 0),
                "후기 값이 박근혜 시기보다 낮고 누적 변화가 음수이면 약화로 해석한다.",
            ),
            row(
                "디지털 담론은 코로나 이전부터 증가했는가?",
                "digital_transformation",
                "moon_pre_covid > park",
                digital["government_change_pre_covid"] > 0,
                "문재인 전기부터 증가하면 코로나 이전의 정책 기조 변화와 연결한다.",
            ),
            row(
                "방역 담론은 코로나 이후에만 급증했는가?",
                "public_health_crisis",
                "pre change <= 0 and post change > 0",
                (crisis["government_change_pre_covid"] <= 0)
                and (crisis["covid_period_shift"] > 0),
                "전기 변화가 약하거나 음수이고 후기 변화가 양수이면 외생충격형 급증으로 해석한다.",
            ),
        ]
    )


def supportive_alignment(clusters: list[Cluster]) -> dict[str, pd.DataFrame]:
    outputs = {}
    dictionary_terms = {
        cluster.cluster_id: cluster.terms for cluster in clusters
    }

    tfidf_path = THREE_PERIOD_DIR / "tfidf_pairwise_period_comparison.csv"
    if tfidf_path.exists():
        tfidf = pd.read_csv(tfidf_path, encoding="utf-8-sig")
        rows = []
        for cluster_id, terms in dictionary_terms.items():
            for comparison, sub in tfidf.groupby("comparison"):
                hits = sub[sub["keyword"].astype(str).str.lower().isin(terms)]
                rows.append(
                    {
                        "source": "tfidf_pairwise_top100",
                        "cluster_id": cluster_id,
                        "comparison": comparison,
                        "hit_count": len(hits),
                        "hit_terms": ";".join(hits["keyword"].astype(str).head(30)),
                    }
                )
        outputs["support_tfidf_alignment"] = pd.DataFrame(rows)

    log_path = THREE_PERIOD_DIR / "log_odds_pairwise_top_keywords.csv"
    if log_path.exists():
        log_df = pd.read_csv(log_path, encoding="utf-8-sig")
        rows = []
        for cluster_id, terms in dictionary_terms.items():
            for comparison, sub in log_df.groupby("comparison"):
                hits = sub[sub["keyword"].astype(str).str.lower().isin(terms)]
                rows.append(
                    {
                        "source": "log_odds_pairwise_top100",
                        "cluster_id": cluster_id,
                        "comparison": comparison,
                        "hit_count": len(hits),
                        "hit_terms": ";".join(hits["keyword"].astype(str).head(30)),
                    }
                )
        outputs["support_log_odds_alignment"] = pd.DataFrame(rows)

    lda_path = THREE_PERIOD_DIR / "lda_topic_words.csv"
    if lda_path.exists():
        lda = pd.read_csv(lda_path, encoding="utf-8-sig")
        rows = []
        for cluster_id, terms in dictionary_terms.items():
            for topic, sub in lda.groupby("topic"):
                hits = sub[sub["keyword"].astype(str).str.lower().isin(terms)]
                rows.append(
                    {
                        "source": "lda_topic_words_top15",
                        "cluster_id": cluster_id,
                        "topic": topic,
                        "hit_count": len(hits),
                        "hit_terms": ";".join(hits["keyword"].astype(str)),
                    }
                )
        outputs["support_lda_alignment"] = pd.DataFrame(rows)

    return outputs


def trend_feasibility(monthly_summary: pd.DataFrame, yearly_summary: pd.DataFrame) -> pd.DataFrame:
    monthly_counts = (
        monthly_summary[["period", "month", "article_count"]]
        .drop_duplicates()
        .groupby("period", observed=True)
        .agg(months=("month", "nunique"), min_monthly_articles=("article_count", "min"))
        .reset_index()
    )
    yearly_counts = (
        yearly_summary[["period", YEAR_COL, "article_count"]]
        .drop_duplicates()
        .groupby("period", observed=True)
        .agg(years=(YEAR_COL, "nunique"), min_yearly_articles=("article_count", "min"))
        .reset_index()
    )
    feasibility = monthly_counts.merge(yearly_counts, on="period", how="outer")
    feasibility["monthly_trend_feasible"] = feasibility["months"] >= 6
    feasibility["yearly_trend_feasible"] = feasibility["years"] >= 2
    feasibility["note"] = "Monthly/yearly trend tables are saved; short boundary years should be interpreted cautiously."
    return feasibility


def save_figures(period_summary: pd.DataFrame, yearly_summary: pd.DataFrame, monthly_summary: pd.DataFrame) -> None:
    configure_font()
    metric = "normalized_frequency_per_10k_tokens"

    fig, ax = plt.subplots(figsize=(11, 6))
    labels = [PERIOD_LABELS[p] for p in PERIOD_ORDER]
    x = np.arange(len(PERIOD_ORDER))
    width = 0.15
    for i, (cluster_id, sub) in enumerate(period_summary.groupby("cluster_id", sort=False)):
        values = [
            sub.loc[sub["period"].astype(str) == period, metric].iloc[0]
            if not sub.loc[sub["period"].astype(str) == period].empty
            else 0
            for period in PERIOD_ORDER
        ]
        ax.bar(x + (i - 2) * width, values, width, label=sub["label_ko"].iloc[0])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Normalized frequency per 10,000 tokens")
    ax.set_title("Conceptual Cluster Frequency by Period")
    ax.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_cluster_period_frequency.png", dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 6))
    for _, sub in yearly_summary.groupby("label_ko", sort=False):
        sub = sub.sort_values(YEAR_COL)
        ax.plot(sub[YEAR_COL], sub[metric], marker="o", label=sub["label_ko"].iloc[0])
    ax.set_ylabel("Normalized frequency per 10,000 tokens")
    ax.set_title("Yearly Trend of Conceptual Clusters")
    ax.legend(ncol=2, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_cluster_yearly_trend.png", dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(13, 6))
    monthly_plot = monthly_summary.copy()
    monthly_plot["month_dt"] = pd.to_datetime(monthly_plot["month"] + "-01")
    for _, sub in monthly_plot.groupby("label_ko", sort=False):
        sub = sub.sort_values("month_dt")
        ax.plot(sub["month_dt"], sub[metric], linewidth=1.2, label=sub["label_ko"].iloc[0])
    ax.set_ylabel("Normalized frequency per 10,000 tokens")
    ax.set_title("Monthly Trend of Conceptual Clusters")
    ax.legend(ncol=2, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_cluster_monthly_trend.png", dpi=300)
    plt.close()


def save_excel(tables: dict[str, pd.DataFrame], path: Path) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, table in tables.items():
            table.to_excel(writer, sheet_name=sheet_name[:31], index=False)


def save_metadata(df: pd.DataFrame, raw_dictionary: dict) -> None:
    metadata = {
        "mode": MODE,
        "data_path": str(DATA_PATH),
        "dictionary_path": str(DICTIONARY_PATH),
        "output_dir": str(OUTPUT_DIR),
        "random_seed": RANDOM_SEED,
        "dictionary_version": raw_dictionary.get("version"),
        "matching": raw_dictionary.get("matching"),
        "periods": [
            {"period": period, "label": PERIOD_LABELS[period], "start": start, "end": end}
            for period, start, end in PERIOD_DEFINITIONS
        ],
        "normalization": f"cluster term count per {NORM_PER_TOKENS} total tokens",
        "article_share": "documents with at least one cluster term divided by article count",
        "input_documents": int(len(df)),
        "period_counts": df["period"].astype(str).value_counts().to_dict(),
        "methodological_note": "Dictionary counts are exact-token measures and should be interpreted as concept proxies, not exhaustive semantic measures.",
    }
    with open(OUTPUT_DIR / "reproducibility_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def main() -> None:
    print("[1/8] Loading data and dictionary...")
    df = load_data()
    clusters, raw_dictionary = load_dictionary()
    dictionary_df, overlap_df = dictionary_tables(clusters, raw_dictionary)

    print("[2/8] Counting cluster terms by article...")
    doc_counts, term_counts = count_cluster_terms(df, clusters)
    term_counts.to_csv(OUTPUT_DIR / "concept_term_counts_long.csv", index=False, encoding="utf-8-sig")

    print("[3/8] Aggregating period, yearly, and monthly normalized frequencies...")
    period_summary = aggregate_cluster_counts(doc_counts, clusters, ["period"])
    yearly_summary = aggregate_cluster_counts(doc_counts, clusters, ["period", YEAR_COL])
    monthly_summary = aggregate_cluster_counts(doc_counts, clusters, ["period", "month"])
    period_summary.to_csv(OUTPUT_DIR / "cluster_frequency_by_period.csv", index=False, encoding="utf-8-sig")
    yearly_summary.to_csv(OUTPUT_DIR / "cluster_frequency_by_year.csv", index=False, encoding="utf-8-sig")
    monthly_summary.to_csv(OUTPUT_DIR / "cluster_frequency_by_month.csv", index=False, encoding="utf-8-sig")

    print("[4/8] Building period comparison and question-answer tables...")
    comparison = period_comparison(period_summary)
    questions = question_summary(comparison)
    feasibility = trend_feasibility(monthly_summary, yearly_summary)
    comparison.to_csv(OUTPUT_DIR / "cluster_period_comparison.csv", index=False, encoding="utf-8-sig")
    questions.to_csv(OUTPUT_DIR / "cluster_research_question_summary.csv", index=False, encoding="utf-8-sig")
    feasibility.to_csv(OUTPUT_DIR / "trend_feasibility.csv", index=False, encoding="utf-8-sig")

    print("[5/8] Linking concept clusters to TF-IDF, log-odds, and LDA outputs...")
    support_tables = supportive_alignment(clusters)
    for name, table in support_tables.items():
        table.to_csv(OUTPUT_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")

    print("[6/8] Saving figures...")
    save_figures(period_summary, yearly_summary, monthly_summary)

    print("[7/8] Saving appendix workbook and metadata...")
    appendix_tables = {
        "dictionary": dictionary_df,
        "overlap_audit": overlap_df,
        "period_frequency": period_summary,
        "period_comparison": comparison,
        "research_questions": questions,
        "yearly_trend": yearly_summary,
        "monthly_trend": monthly_summary,
        "trend_feasibility": feasibility,
    }
    appendix_tables.update(support_tables)
    save_excel(appendix_tables, OUTPUT_DIR / "appendix_concept_cluster_analysis.xlsx")
    save_metadata(df, raw_dictionary)

    print("[8/8] Complete.")
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
