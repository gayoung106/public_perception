"""
Three-period discourse analysis.

Mode
----
mode = "three_period"

Purpose
-------
Extends the original two-government comparison into a three-period design:

1. Park Geun-hye administration
   2013-02-25 to 2017-03-10
2. Moon Jae-in early period, before COVID-19
   2017-05-10 to 2019-12-31
3. Moon Jae-in late period, after COVID-19 onset
   2020-01-01 to 2022-05-09

The goal is to separate, as far as a descriptive text-mining design allows,
government-change effects from COVID-19 external-shock effects.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from gensim import corpora
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from sklearn.feature_extraction.text import TfidfVectorizer

from stopwords import STOPWORDS


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
MODE = "three_period"
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "datas" / "preprocessed_2013_2022.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "three_period"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATE_COL = "날짜"
YEAR_COL = "year"
GOV_COL = "정부"
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

PAIRWISE_COMPARISONS = [
    {
        "comparison": "government_change_pre_covid",
        "period_a": PARK_PERIOD,
        "period_b": MOON_PRE,
        "interpretation": "문재인 전기에서 이미 나타난 정부교체 관련 변화",
    },
    {
        "comparison": "covid_period_shift",
        "period_a": MOON_PRE,
        "period_b": MOON_POST,
        "interpretation": "문재인 후기에서 추가 강화된 코로나 이후 변화",
    },
    {
        "comparison": "cumulative_change",
        "period_a": PARK_PERIOD,
        "period_b": MOON_POST,
        "interpretation": "정부교체와 코로나 이후 변화가 누적된 총 변화",
    },
]

TFIDF_MAX_FEATURES = 3000
TFIDF_MIN_DF = 10
TFIDF_MAX_DF = 0.7
TOP_N = 100

LOG_ODDS_MIN_COUNT = 30

LDA_NUM_TOPICS = 10
LDA_PASSES = 5
LDA_ITERATIONS = 100
LDA_NO_BELOW = 10
LDA_NO_ABOVE = 0.5
LDA_SAMPLE_SIZE = 200_000

STOPWORDS_SET = set(STOPWORDS(version="base"))


@dataclass(frozen=True)
class PeriodDefinition:
    period: str
    label: str
    start: str
    end: str


PERIOD_DEFINITIONS = [
    PeriodDefinition(PARK_PERIOD, PERIOD_LABELS[PARK_PERIOD], "2013-02-25", "2017-03-10"),
    PeriodDefinition(MOON_PRE, PERIOD_LABELS[MOON_PRE], "2017-05-10", "2019-12-31"),
    PeriodDefinition(MOON_POST, PERIOD_LABELS[MOON_POST], "2020-01-01", "2022-05-09"),
]


def assign_period(date_value: pd.Timestamp) -> str | None:
    for period_def in PERIOD_DEFINITIONS:
        if pd.Timestamp(period_def.start) <= date_value <= pd.Timestamp(period_def.end):
            return period_def.period
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
    return df


def tokenize(text: str, stopwords: set[str] = STOPWORDS_SET) -> list[str]:
    return [
        token
        for token in str(text).split()
        if len(token) > 1 and token not in stopwords
    ]


def cleaned_corpus(texts: Iterable[str], stopwords: set[str] = STOPWORDS_SET) -> list[str]:
    return [" ".join(tokenize(text, stopwords)) for text in texts]


def article_counts(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_period = (
        df.groupby("period", observed=True)
        .size()
        .reset_index(name="article_count")
        .assign(period_label=lambda x: x["period"].map(PERIOD_LABELS))
    )
    by_period_year = (
        df.groupby(["period", YEAR_COL], observed=True)
        .size()
        .reset_index(name="article_count")
        .assign(period_label=lambda x: x["period"].map(PERIOD_LABELS))
        .sort_values(["period", YEAR_COL])
    )
    return by_period, by_period_year


def compute_tfidf_by_period(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        min_df=TFIDF_MIN_DF,
        max_df=TFIDF_MAX_DF,
        tokenizer=str.split,
        token_pattern=None,
        lowercase=False,
    )
    corpus = cleaned_corpus(df[TEXT_COL])
    matrix = vectorizer.fit_transform(corpus)
    terms = vectorizer.get_feature_names_out()

    rows = []
    for period in PERIOD_ORDER:
        idx = df["period"].astype(str).values == period
        scores = np.asarray(matrix[idx].mean(axis=0)).ravel()
        for term, score in zip(terms, scores):
            rows.append(
                {
                    "period": period,
                    "period_label": PERIOD_LABELS[period],
                    "keyword": term,
                    "tfidf": score,
                }
            )

    tfidf_long = pd.DataFrame(rows)
    top_keywords = (
        tfidf_long.sort_values(["period", "tfidf"], ascending=[True, False])
        .groupby("period", observed=True)
        .head(TOP_N)
        .assign(rank=lambda x: x.groupby("period", observed=True)["tfidf"].rank(ascending=False, method="first"))
        .sort_values(["period", "rank"])
    )
    return tfidf_long, top_keywords


def pairwise_tfidf(tfidf_long: pd.DataFrame) -> pd.DataFrame:
    wide = tfidf_long.pivot(index="keyword", columns="period", values="tfidf").fillna(0)
    rows = []
    for spec in PAIRWISE_COMPARISONS:
        a = spec["period_a"]
        b = spec["period_b"]
        comp = pd.DataFrame(
            {
                "comparison": spec["comparison"],
                "interpretation": spec["interpretation"],
                "keyword": wide.index,
                "period_a": a,
                "period_b": b,
                "tfidf_a": wide[a].values,
                "tfidf_b": wide[b].values,
                "delta_b_minus_a": (wide[b] - wide[a]).values,
            }
        )
        comp["pct_change_b_vs_a"] = (
            comp["delta_b_minus_a"] / comp["tfidf_a"].replace(0, 0.0001)
        ) * 100
        comp["abs_delta"] = comp["delta_b_minus_a"].abs()
        comp["rank_abs_delta"] = comp["abs_delta"].rank(ascending=False, method="first")
        rows.append(comp.sort_values("abs_delta", ascending=False))
    return pd.concat(rows, ignore_index=True)


def top_pairwise_tfidf(tfidf_pairwise: pd.DataFrame) -> pd.DataFrame:
    return (
        tfidf_pairwise.sort_values(["comparison", "abs_delta"], ascending=[True, False])
        .groupby("comparison", observed=True)
        .head(TOP_N)
        .sort_values(["comparison", "rank_abs_delta"])
    )


def count_words(texts: Iterable[str]) -> Counter:
    counter: Counter = Counter()
    for text in texts:
        counter.update(tokenize(text))
    return counter


def log_odds_for_pair(
    df: pd.DataFrame,
    period_a: str,
    period_b: str,
    comparison: str,
    interpretation: str,
) -> pd.DataFrame:
    counts_a = count_words(df.loc[df["period"].astype(str) == period_a, TEXT_COL])
    counts_b = count_words(df.loc[df["period"].astype(str) == period_b, TEXT_COL])

    prior = Counter()
    prior.update(counts_a)
    prior.update(counts_b)
    alpha_0 = sum(prior.values())
    n_a = sum(counts_a.values())
    n_b = sum(counts_b.values())

    rows = []
    for word in sorted(set(counts_a) | set(counts_b)):
        c_a = counts_a.get(word, 0)
        c_b = counts_b.get(word, 0)
        c_prior = prior[word]
        if c_a + c_b < LOG_ODDS_MIN_COUNT:
            continue
        log_odds = math.log((c_b + c_prior) / (n_b - c_b + alpha_0 - c_prior)) - math.log(
            (c_a + c_prior) / (n_a - c_a + alpha_0 - c_prior)
        )
        var = (1 / (c_b + c_prior)) + (1 / (c_a + c_prior))
        z_score = log_odds / math.sqrt(var)
        rows.append(
            {
                "comparison": comparison,
                "interpretation": interpretation,
                "keyword": word,
                "period_a": period_a,
                "period_b": period_b,
                "count_a": c_a,
                "count_b": c_b,
                "log_odds_z_b_vs_a": z_score,
                "direction": period_b if z_score > 0 else period_a,
                "abs_log_odds_z": abs(z_score),
            }
        )
    return pd.DataFrame(rows).sort_values("log_odds_z_b_vs_a", ascending=False)


def compute_log_odds(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_rows = []
    for spec in PAIRWISE_COMPARISONS:
        all_rows.append(
            log_odds_for_pair(
                df,
                spec["period_a"],
                spec["period_b"],
                spec["comparison"],
                spec["interpretation"],
            )
        )
    log_long = pd.concat(all_rows, ignore_index=True)
    top_log = (
        log_long.sort_values(["comparison", "abs_log_odds_z"], ascending=[True, False])
        .groupby("comparison", observed=True)
        .head(TOP_N)
        .assign(rank_abs_z=lambda x: x.groupby("comparison", observed=True)["abs_log_odds_z"].rank(ascending=False, method="first"))
        .sort_values(["comparison", "rank_abs_z"])
    )
    return log_long, top_log


def effect_decomposition(tfidf_pairwise: pd.DataFrame, log_long: pd.DataFrame) -> pd.DataFrame:
    tfidf_wide = tfidf_pairwise.pivot_table(
        index="keyword",
        columns="comparison",
        values="delta_b_minus_a",
        aggfunc="first",
    )
    log_wide = log_long.pivot_table(
        index="keyword",
        columns="comparison",
        values="log_odds_z_b_vs_a",
        aggfunc="first",
    )
    merged = tfidf_wide.add_prefix("tfidf_delta_").join(
        log_wide.add_prefix("log_odds_z_"), how="outer"
    ).fillna(0)

    gov_col = "tfidf_delta_government_change_pre_covid"
    covid_col = "tfidf_delta_covid_period_shift"
    cumulative_col = "tfidf_delta_cumulative_change"

    def classify(row: pd.Series) -> str:
        gov = row.get(gov_col, 0)
        covid = row.get(covid_col, 0)
        cumulative = row.get(cumulative_col, 0)
        if gov > 0 and abs(gov) >= abs(covid):
            return "pre_existing_government_change"
        if gov > 0 and covid > 0:
            return "cumulative_reconfiguration"
        if gov <= 0 and covid > 0 and cumulative > 0:
            return "post_covid_specific_increase"
        if gov < 0 and covid < 0:
            return "decline_across_moon_period"
        return "mixed_or_weak_pattern"

    merged["effect_pattern"] = merged.apply(classify, axis=1)
    merged["absolute_cumulative_tfidf_change"] = merged[cumulative_col].abs()
    return (
        merged.reset_index()
        .sort_values("absolute_cumulative_tfidf_change", ascending=False)
        .reset_index(drop=True)
    )


def prepare_lda_frame(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) <= LDA_SAMPLE_SIZE:
        return df.copy()
    return df.sample(LDA_SAMPLE_SIZE, random_state=RANDOM_SEED).copy()


def train_lda(texts: list[list[str]]) -> tuple[LdaModel, corpora.Dictionary]:
    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=LDA_NO_BELOW, no_above=LDA_NO_ABOVE)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lda = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=LDA_NUM_TOPICS,
        random_state=RANDOM_SEED,
        passes=LDA_PASSES,
        iterations=LDA_ITERATIONS,
        alpha="symmetric",
        eta="auto",
    )
    return lda, dictionary


def compute_lda(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    lda_df = prepare_lda_frame(df)
    lda_df["tokens"] = lda_df[TEXT_COL].apply(tokenize)
    lda_df = lda_df[lda_df["tokens"].apply(len) > 0].copy()
    texts = lda_df["tokens"].tolist()

    lda, dictionary = train_lda(texts)

    topic_word_rows = []
    for topic_id in range(LDA_NUM_TOPICS):
        for rank, (word, prob) in enumerate(lda.show_topic(topic_id, topn=15), start=1):
            topic_word_rows.append(
                {
                    "topic": topic_id,
                    "rank": rank,
                    "keyword": word,
                    "probability": round(float(prob), 6),
                }
            )

    dist_rows = []
    for period in PERIOD_ORDER:
        sub = lda_df[lda_df["period"].astype(str) == period]
        vectors = []
        for tokens in sub["tokens"]:
            bow = dictionary.doc2bow(tokens)
            dist = np.zeros(LDA_NUM_TOPICS)
            for topic_id, prob in lda.get_document_topics(bow, minimum_probability=0):
                dist[topic_id] = prob
            vectors.append(dist)
        mat = np.vstack(vectors)
        for topic_id, mean_share in enumerate(mat.mean(axis=0)):
            dist_rows.append(
                {
                    "period": period,
                    "period_label": PERIOD_LABELS[period],
                    "topic": topic_id,
                    "mean_topic_share": round(float(mean_share), 6),
                    "documents": len(sub),
                }
            )

    topic_words = pd.DataFrame(topic_word_rows)
    topic_dist = pd.DataFrame(dist_rows)
    topic_wide = topic_dist.pivot(index="topic", columns="period", values="mean_topic_share").reset_index()
    topic_wide["government_change_pre_covid"] = topic_wide[MOON_PRE] - topic_wide[PARK_PERIOD]
    topic_wide["covid_period_shift"] = topic_wide[MOON_POST] - topic_wide[MOON_PRE]
    topic_wide["cumulative_change"] = topic_wide[MOON_POST] - topic_wide[PARK_PERIOD]
    topic_wide["dominant_effect"] = np.where(
        topic_wide["government_change_pre_covid"].abs() >= topic_wide["covid_period_shift"].abs(),
        "government_change_pre_covid",
        "covid_period_shift",
    )

    coherence = CoherenceModel(
        model=lda,
        texts=texts,
        dictionary=dictionary,
        coherence="c_v",
        processes=1,
    ).get_coherence()

    return {
        "lda_topic_words": topic_words,
        "lda_topic_distribution": topic_dist,
        "lda_topic_period_comparison": topic_wide,
        "lda_metadata": pd.DataFrame(
            [
                {
                    "mode": MODE,
                    "input_documents": len(df),
                    "lda_documents": len(lda_df),
                    "sample_size_limit": LDA_SAMPLE_SIZE,
                    "sampled": len(df) > LDA_SAMPLE_SIZE,
                    "random_seed": RANDOM_SEED,
                    "num_topics": LDA_NUM_TOPICS,
                    "passes": LDA_PASSES,
                    "iterations": LDA_ITERATIONS,
                    "coherence_c_v": round(float(coherence), 6),
                    "lda_implementation": "gensim.models.LdaModel",
                    "workers": 1,
                }
            ]
        ),
    }


def answer_tables(
    effect_table: pd.DataFrame,
    log_top: pd.DataFrame,
    lda_topic_comparison: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    pre_existing = effect_table[
        effect_table["effect_pattern"].isin(
            ["pre_existing_government_change", "cumulative_reconfiguration"]
        )
    ].head(TOP_N)
    post_specific = effect_table[
        effect_table["effect_pattern"] == "post_covid_specific_increase"
    ].head(TOP_N)
    dominant_effect = (
        lda_topic_comparison[["topic", "government_change_pre_covid", "covid_period_shift", "cumulative_change", "dominant_effect"]]
        .sort_values("cumulative_change", key=lambda x: x.abs(), ascending=False)
    )
    log_covid = log_top[log_top["comparison"] == "covid_period_shift"].copy()
    return {
        "q1_pre_covid_change_keywords": pre_existing,
        "q2_post_covid_specific_keywords": post_specific,
        "q2_post_covid_log_odds": log_covid,
        "q3_topic_dominant_effect": dominant_effect,
    }


def save_excel(tables: dict[str, pd.DataFrame], path: Path) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, table in tables.items():
            table.to_excel(writer, sheet_name=sheet_name[:31], index=False)


def save_metadata(df: pd.DataFrame) -> None:
    metadata = {
        "mode": MODE,
        "data_path": str(DATA_PATH),
        "output_dir": str(OUTPUT_DIR),
        "random_seed": RANDOM_SEED,
        "periods": [period.__dict__ for period in PERIOD_DEFINITIONS],
        "comparisons": PAIRWISE_COMPARISONS,
        "stopwords": {
            "version": "base",
            "count": len(STOPWORDS_SET),
        },
        "tfidf": {
            "max_features": TFIDF_MAX_FEATURES,
            "min_df": TFIDF_MIN_DF,
            "max_df": TFIDF_MAX_DF,
            "vocabulary": "pooled three-period corpus",
        },
        "log_odds": {
            "prior": "comparison-specific combined corpus informative Dirichlet prior",
            "min_total_count": LOG_ODDS_MIN_COUNT,
        },
        "lda": {
            "num_topics": LDA_NUM_TOPICS,
            "passes": LDA_PASSES,
            "iterations": LDA_ITERATIONS,
            "random_state": RANDOM_SEED,
            "sample_size_limit": LDA_SAMPLE_SIZE,
            "implementation": "gensim.models.LdaModel",
            "workers": 1,
        },
        "documents": {
            "input_documents": int(len(df)),
            "period_counts": df["period"].astype(str).value_counts().to_dict(),
        },
    }
    with open(OUTPUT_DIR / "reproducibility_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def main() -> None:
    if MODE != "three_period":
        raise ValueError("This script currently supports mode='three_period' only.")

    print("[1/7] Loading data and assigning periods...")
    df = load_data()
    counts_period, counts_year = article_counts(df)
    counts_period.to_csv(OUTPUT_DIR / "article_count_by_period.csv", index=False, encoding="utf-8-sig")
    counts_year.to_csv(OUTPUT_DIR / "article_count_by_period_year.csv", index=False, encoding="utf-8-sig")

    print("[2/7] Running TF-IDF by period...")
    tfidf_long, tfidf_top = compute_tfidf_by_period(df)
    tfidf_pairwise = pairwise_tfidf(tfidf_long)
    tfidf_pairwise_top = top_pairwise_tfidf(tfidf_pairwise)
    tfidf_long.to_csv(OUTPUT_DIR / "tfidf_by_period_all_terms.csv", index=False, encoding="utf-8-sig")
    tfidf_top.to_csv(OUTPUT_DIR / "tfidf_top_keywords_by_period.csv", index=False, encoding="utf-8-sig")
    tfidf_pairwise.to_csv(OUTPUT_DIR / "tfidf_pairwise_period_comparison_all_terms.csv", index=False, encoding="utf-8-sig")
    tfidf_pairwise_top.to_csv(OUTPUT_DIR / "tfidf_pairwise_period_comparison.csv", index=False, encoding="utf-8-sig")

    print("[3/7] Running pairwise log-odds...")
    log_long, log_top = compute_log_odds(df)
    log_long.to_csv(OUTPUT_DIR / "log_odds_pairwise_all_terms.csv", index=False, encoding="utf-8-sig")
    log_top.to_csv(OUTPUT_DIR / "log_odds_pairwise_top_keywords.csv", index=False, encoding="utf-8-sig")

    print("[4/7] Building government-vs-COVID decomposition tables...")
    effects = effect_decomposition(tfidf_pairwise, log_long)
    effects.to_csv(OUTPUT_DIR / "keyword_effect_decomposition.csv", index=False, encoding="utf-8-sig")

    print("[5/7] Running unified LDA(k=10) over three periods...")
    lda_outputs = compute_lda(df)
    for name, table in lda_outputs.items():
        table.to_csv(OUTPUT_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")

    print("[6/7] Saving appendix tables...")
    q_tables = answer_tables(effects, log_top, lda_outputs["lda_topic_period_comparison"])
    for name, table in q_tables.items():
        table.to_csv(OUTPUT_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")

    appendix_tables = {
        "article_counts": counts_period,
        "tfidf_top": tfidf_top,
        "tfidf_pairwise": tfidf_pairwise_top,
        "log_odds_top": log_top,
        "effect_decomposition": effects.head(300),
        "lda_topic_dist": lda_outputs["lda_topic_distribution"],
        "lda_topic_compare": lda_outputs["lda_topic_period_comparison"],
        "q1_pre_covid_change": q_tables["q1_pre_covid_change_keywords"],
        "q2_post_covid_specific": q_tables["q2_post_covid_specific_keywords"],
        "q3_topic_effect": q_tables["q3_topic_dominant_effect"],
    }
    save_excel(appendix_tables, OUTPUT_DIR / "appendix_three_period_tables.xlsx")
    save_metadata(df)

    print("[7/7] Complete.")
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
