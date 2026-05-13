"""
Formal COVID-19 robustness analysis.

Purpose
-------
This script extends the earlier exploratory COVID split check into a
publication-level robustness analysis for the reviewer concern that
government-change effects may be confounded with the COVID-19 external shock.

Design
------
1. Recompute the main Park-vs-Moon comparison under two conditions:
   - original_base: STOPWORDS(version="base")
   - covid_filtered: STOPWORDS(version="base") plus COVID_TOKENS
2. Apply the same comparison logic to TF-IDF, log-odds, and LDA.
3. Save appendix-ready CSV and XLSX tables under results/covid_filtered/.
4. Use one random seed consistently and prefer single-core LDA for
   reproducibility.
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
# Reproducibility and analysis parameters
# ---------------------------------------------------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "datas" / "preprocessed_2013_2022.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "covid_filtered"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATE_COL = "날짜"
YEAR_COL = "year"
GOV_COL = "정부"
TEXT_COL = "text"
PARK = "박근혜정부"
MOON = "문재인정부"

TFIDF_MAX_FEATURES = 3000
TFIDF_MIN_DF = 10
TFIDF_MAX_DF = 0.7
TFIDF_TOP_N = 100

LOG_ODDS_MIN_COUNT = 30

LDA_NUM_TOPICS = 10
LDA_PASSES = 5
LDA_ITERATIONS = 100
LDA_NO_BELOW = 10
LDA_NO_ABOVE = 0.5
LDA_SAMPLE_SIZE = 200_000


# COVID terms are intentionally kept outside stopwords.py so the main analysis
# remains unchanged and the filtered specification is auditable.
COVID_TOKENS = {
    "코로나",
    "코로나19",
    "코로나바이러스",
    "신종코로나",
    "신종코로나바이러스",
    "바이러스",
    "확진",
    "확진자",
    "감염",
    "감염증",
    "신종",
    "마스크",
    "백신",
    "백신접종",
    "접종",
    "방역",
    "방역당국",
    "자가격리",
    "격리",
    "거리두기",
    "사회적거리두기",
    "집합금지",
    "봉쇄",
    "팬데믹",
    "위드코로나",
    "오미크론",
    "델타",
    "중대본",
    "중앙방역대책본부",
    "질병관리청",
    "재난지원금",
    "비대면",
}


@dataclass(frozen=True)
class Condition:
    name: str
    stopwords: set[str]
    description: str


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df = df.dropna(subset=[TEXT_COL, GOV_COL]).copy()
    df = df[df[GOV_COL].isin([PARK, MOON])].copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df[YEAR_COL] = df[DATE_COL].dt.year
    return df


def tokenize(text: str, stopwords: set[str]) -> list[str]:
    return [
        token
        for token in str(text).split()
        if len(token) > 1 and token not in stopwords
    ]


def cleaned_corpus(texts: Iterable[str], stopwords: set[str]) -> list[str]:
    return [" ".join(tokenize(text, stopwords)) for text in texts]


def save_excel(tables: dict[str, pd.DataFrame], path: Path) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, table in tables.items():
            table.to_excel(writer, sheet_name=sheet_name[:31], index=False)


def build_conditions() -> list[Condition]:
    base_stopwords = set(STOPWORDS(version="base"))
    return [
        Condition(
            name="original_base",
            stopwords=base_stopwords,
            description="Original main specification using STOPWORDS(version='base').",
        ),
        Condition(
            name="covid_filtered",
            stopwords=base_stopwords | COVID_TOKENS,
            description="Robustness specification excluding base stopwords and COVID_TOKENS.",
        ),
    ]


def audit_covid_tokens(base_stopwords: set[str]) -> pd.DataFrame:
    rows = []
    for token in sorted(COVID_TOKENS):
        rows.append(
            {
                "token": token,
                "already_in_base_stopwords": token in base_stopwords,
                "added_by_covid_filter": token not in base_stopwords,
            }
        )
    return pd.DataFrame(rows)


def compute_tfidf_change(df: pd.DataFrame, condition: Condition) -> pd.DataFrame:
    park_docs = cleaned_corpus(df.loc[df[GOV_COL] == PARK, TEXT_COL], condition.stopwords)
    moon_docs = cleaned_corpus(df.loc[df[GOV_COL] == MOON, TEXT_COL], condition.stopwords)

    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        min_df=TFIDF_MIN_DF,
        max_df=TFIDF_MAX_DF,
        tokenizer=str.split,
        token_pattern=None,
        lowercase=False,
    )
    tfidf_park = vectorizer.fit_transform(park_docs)
    tfidf_moon = vectorizer.transform(moon_docs)

    terms = vectorizer.get_feature_names_out()
    result = pd.DataFrame(
        {
            "keyword": terms,
            "tfidf_park": np.asarray(tfidf_park.mean(axis=0)).ravel(),
            "tfidf_moon": np.asarray(tfidf_moon.mean(axis=0)).ravel(),
        }
    )
    result["importance"] = result["tfidf_park"] + result["tfidf_moon"]
    result["delta_moon_minus_park"] = result["tfidf_moon"] - result["tfidf_park"]
    result["pct_change_moon_vs_park"] = (
        result["delta_moon_minus_park"] / result["tfidf_park"].replace(0, 0.0001)
    ) * 100
    result["condition"] = condition.name
    return result.sort_values("importance", ascending=False).reset_index(drop=True)


def compare_keyword_tables(
    original: pd.DataFrame,
    filtered: pd.DataFrame,
    score_col: str,
    top_n: int,
) -> pd.DataFrame:
    orig_top = original.head(top_n).copy()
    filt_top = filtered.head(top_n).copy()

    merged = orig_top.merge(
        filt_top,
        on="keyword",
        how="outer",
        suffixes=("_original", "_covid_filtered"),
        indicator=True,
    )
    merged["retained_in_top_n"] = merged["_merge"] == "both"
    merged["rank_original"] = merged[f"{score_col}_original"].rank(
        ascending=False, method="min"
    )
    merged["rank_covid_filtered"] = merged[f"{score_col}_covid_filtered"].rank(
        ascending=False, method="min"
    )
    merged["rank_change_filtered_minus_original"] = (
        merged["rank_covid_filtered"] - merged["rank_original"]
    )
    return merged.drop(columns=["_merge"]).sort_values(
        ["retained_in_top_n", "rank_original"], ascending=[False, True]
    )


def count_words(texts: Iterable[str], stopwords: set[str]) -> Counter:
    counts: Counter = Counter()
    for text in texts:
        counts.update(tokenize(text, stopwords))
    return counts


def compute_log_odds(df: pd.DataFrame, condition: Condition) -> pd.DataFrame:
    park_counts = count_words(df.loc[df[GOV_COL] == PARK, TEXT_COL], condition.stopwords)
    moon_counts = count_words(df.loc[df[GOV_COL] == MOON, TEXT_COL], condition.stopwords)

    prior = Counter()
    prior.update(park_counts)
    prior.update(moon_counts)

    alpha_0 = sum(prior.values())
    n_moon = sum(moon_counts.values())
    n_park = sum(park_counts.values())
    rows = []

    for word in sorted(set(park_counts) | set(moon_counts)):
        c_moon = moon_counts.get(word, 0)
        c_park = park_counts.get(word, 0)
        c_prior = prior[word]

        if c_moon + c_park < LOG_ODDS_MIN_COUNT:
            continue

        log_odds = math.log(
            (c_moon + c_prior) / (n_moon - c_moon + alpha_0 - c_prior)
        ) - math.log(
            (c_park + c_prior) / (n_park - c_park + alpha_0 - c_prior)
        )
        var = (1 / (c_moon + c_prior)) + (1 / (c_park + c_prior))
        z_score = log_odds / math.sqrt(var)
        rows.append(
            {
                "keyword": word,
                "count_moon": c_moon,
                "count_park": c_park,
                "log_odds_z": z_score,
                "direction": MOON if z_score > 0 else PARK,
                "abs_log_odds_z": abs(z_score),
                "condition": condition.name,
            }
        )

    return pd.DataFrame(rows).sort_values(
        "log_odds_z", ascending=False
    ).reset_index(drop=True)


def compare_log_odds(original: pd.DataFrame, filtered: pd.DataFrame, top_n: int) -> pd.DataFrame:
    original_ranked = original.assign(
        rank_original=original["abs_log_odds_z"].rank(ascending=False, method="min")
    )
    filtered_ranked = filtered.assign(
        rank_covid_filtered=filtered["abs_log_odds_z"].rank(ascending=False, method="min")
    )
    original_top = original_ranked.nsmallest(top_n, "rank_original")
    filtered_top = filtered_ranked.nsmallest(top_n, "rank_covid_filtered")

    merged = original_top.merge(
        filtered_top,
        on="keyword",
        how="outer",
        suffixes=("_original", "_covid_filtered"),
        indicator=True,
    )
    merged["retained_in_top_n"] = merged["_merge"] == "both"
    merged["direction_stable"] = (
        merged["direction_original"] == merged["direction_covid_filtered"]
    )
    merged["rank_change_filtered_minus_original"] = (
        merged["rank_covid_filtered"] - merged["rank_original"]
    )
    return merged.drop(columns=["_merge"]).sort_values(
        ["retained_in_top_n", "rank_original"], ascending=[False, True]
    )


def prepare_lda_frame(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) <= LDA_SAMPLE_SIZE:
        return df.copy()
    return df.sample(LDA_SAMPLE_SIZE, random_state=RANDOM_SEED).copy()


def train_lda(texts: list[list[str]]) -> tuple[LdaModel, corpora.Dictionary, list[list[tuple[int, int]]]]:
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
    return lda, dictionary, corpus


def topic_distribution(
    df: pd.DataFrame,
    lda: LdaModel,
    dictionary: corpora.Dictionary,
    token_col: str,
    condition_name: str,
) -> pd.DataFrame:
    rows = []
    for gov in [PARK, MOON]:
        sub = df[df[GOV_COL] == gov]
        vectors = []
        for tokens in sub[token_col]:
            bow = dictionary.doc2bow(tokens)
            dist = np.zeros(LDA_NUM_TOPICS)
            for topic_id, prob in lda.get_document_topics(bow, minimum_probability=0):
                dist[topic_id] = prob
            vectors.append(dist)
        mat = np.vstack(vectors)
        for topic_id, mean_prob in enumerate(mat.mean(axis=0)):
            rows.append(
                {
                    "condition": condition_name,
                    "government": gov,
                    "topic": topic_id,
                    "mean_topic_share": round(float(mean_prob), 6),
                    "documents": len(sub),
                }
            )
    return pd.DataFrame(rows)


def lda_topic_words(lda: LdaModel, condition_name: str, topn: int = 15) -> pd.DataFrame:
    rows = []
    for topic_id in range(LDA_NUM_TOPICS):
        for rank, (word, prob) in enumerate(lda.show_topic(topic_id, topn=topn), start=1):
            rows.append(
                {
                    "condition": condition_name,
                    "topic": topic_id,
                    "rank": rank,
                    "keyword": word,
                    "probability": round(float(prob), 6),
                }
            )
    return pd.DataFrame(rows)


def lda_government_gap(dist_df: pd.DataFrame) -> pd.DataFrame:
    pivot = dist_df.pivot_table(
        index=["condition", "topic"],
        columns="government",
        values="mean_topic_share",
        aggfunc="first",
    ).reset_index()
    pivot["moon_minus_park"] = pivot[MOON] - pivot[PARK]
    pivot["abs_gap"] = pivot["moon_minus_park"].abs()
    return pivot.sort_values(["condition", "abs_gap"], ascending=[True, False])


def run_lda_robustness(df: pd.DataFrame, conditions: list[Condition]) -> dict[str, pd.DataFrame]:
    lda_df = prepare_lda_frame(df)
    original = conditions[0]
    filtered = conditions[1]

    lda_df["tokens_original"] = lda_df[TEXT_COL].apply(lambda x: tokenize(x, original.stopwords))
    lda_df["tokens_covid_filtered"] = lda_df[TEXT_COL].apply(
        lambda x: tokenize(x, filtered.stopwords)
    )
    lda_df = lda_df[lda_df["tokens_original"].apply(len) > 0].copy()

    original_texts = lda_df["tokens_original"].tolist()
    lda_original, dictionary_original, corpus_original = train_lda(original_texts)

    # Primary LDA robustness: infer original and COVID-filtered documents in the
    # same topic space learned from the original corpus.
    dist_original = topic_distribution(
        lda_df, lda_original, dictionary_original, "tokens_original", "original_base"
    )
    dist_filtered_same_model = topic_distribution(
        lda_df,
        lda_original,
        dictionary_original,
        "tokens_covid_filtered",
        "covid_filtered_same_topic_space",
    )

    topics_original = lda_topic_words(lda_original, "original_base")

    coherence_rows = []
    coherence_original = CoherenceModel(
        model=lda_original,
        texts=original_texts,
        dictionary=dictionary_original,
        coherence="c_v",
        processes=1,
    ).get_coherence()
    coherence_rows.append(
        {
            "condition": "original_base",
            "model": "original_topic_space",
            "coherence_type": "c_v",
            "coherence": round(float(coherence_original), 6),
        }
    )

    # Secondary sensitivity: train a separate filtered LDA model. Topic IDs from
    # this model are not directly comparable to the original model, so the main
    # comparison table uses the same-topic-space inference above.
    filtered_texts = [
        tokens for tokens in lda_df["tokens_covid_filtered"].tolist() if len(tokens) > 0
    ]
    lda_filtered, dictionary_filtered, _ = train_lda(filtered_texts)
    topics_filtered_model = lda_topic_words(lda_filtered, "covid_filtered_retrained")
    coherence_filtered = CoherenceModel(
        model=lda_filtered,
        texts=filtered_texts,
        dictionary=dictionary_filtered,
        coherence="c_v",
        processes=1,
    ).get_coherence()
    coherence_rows.append(
        {
            "condition": "covid_filtered",
            "model": "retrained_filtered_topic_space",
            "coherence_type": "c_v",
            "coherence": round(float(coherence_filtered), 6),
        }
    )

    dist_all = pd.concat([dist_original, dist_filtered_same_model], ignore_index=True)
    gap = lda_government_gap(dist_all)
    gap_comparison = gap.pivot_table(
        index="topic",
        columns="condition",
        values="moon_minus_park",
        aggfunc="first",
    ).reset_index()
    gap_comparison["gap_change_filtered_minus_original"] = (
        gap_comparison["covid_filtered_same_topic_space"]
        - gap_comparison["original_base"]
    )
    gap_comparison["abs_gap_change"] = gap_comparison[
        "gap_change_filtered_minus_original"
    ].abs()

    return {
        "lda_topic_words_original": topics_original,
        "lda_topic_words_filtered_retrained": topics_filtered_model,
        "lda_topic_distribution": dist_all,
        "lda_government_gap": gap,
        "lda_gap_comparison": gap_comparison.sort_values(
            "abs_gap_change", ascending=False
        ),
        "lda_coherence": pd.DataFrame(coherence_rows),
        "lda_sample_metadata": pd.DataFrame(
            [
                {
                    "input_documents": len(df),
                    "lda_documents": len(lda_df),
                    "sample_size_limit": LDA_SAMPLE_SIZE,
                    "sampled": len(df) > LDA_SAMPLE_SIZE,
                    "random_seed": RANDOM_SEED,
                    "num_topics": LDA_NUM_TOPICS,
                    "passes": LDA_PASSES,
                    "iterations": LDA_ITERATIONS,
                    "lda_implementation": "gensim.models.LdaModel",
                    "workers": 1,
                }
            ]
        ),
    }


def summarize_stability(
    tfidf_comparison: pd.DataFrame,
    log_odds_comparison: pd.DataFrame,
    lda_gap_comparison: pd.DataFrame,
) -> pd.DataFrame:
    tfidf_retention = tfidf_comparison["retained_in_top_n"].sum() / TFIDF_TOP_N
    log_retention = log_odds_comparison["retained_in_top_n"].sum() / TFIDF_TOP_N
    log_direction_stable = log_odds_comparison.loc[
        log_odds_comparison["retained_in_top_n"], "direction_stable"
    ].mean()

    return pd.DataFrame(
        [
            {
                "method": "TF-IDF",
                "comparison": f"top {TFIDF_TOP_N} original vs covid-filtered",
                "stability_metric": "top_keyword_retention_rate",
                "value": round(float(tfidf_retention), 4),
            },
            {
                "method": "Log-odds",
                "comparison": f"top {TFIDF_TOP_N} by |z| original vs covid-filtered",
                "stability_metric": "top_keyword_retention_rate",
                "value": round(float(log_retention), 4),
            },
            {
                "method": "Log-odds",
                "comparison": "retained top keywords",
                "stability_metric": "direction_stability_rate",
                "value": round(float(log_direction_stable), 4),
            },
            {
                "method": "LDA",
                "comparison": "same-topic-space government gap",
                "stability_metric": "mean_abs_topic_gap_change",
                "value": round(float(lda_gap_comparison["abs_gap_change"].mean()), 6),
            },
            {
                "method": "LDA",
                "comparison": "same-topic-space government gap",
                "stability_metric": "max_abs_topic_gap_change",
                "value": round(float(lda_gap_comparison["abs_gap_change"].max()), 6),
            },
        ]
    )


def save_metadata(df: pd.DataFrame, conditions: list[Condition]) -> None:
    yearly = (
        df.groupby([GOV_COL, YEAR_COL])
        .size()
        .reset_index(name="article_count")
        .sort_values([GOV_COL, YEAR_COL])
    )
    yearly.to_csv(OUTPUT_DIR / "sample_article_counts_by_year.csv", index=False, encoding="utf-8-sig")

    moon_period = df[df[GOV_COL] == MOON].copy()
    moon_period["covid_period"] = np.where(
        moon_period[YEAR_COL].between(2017, 2019),
        "pre_covid_2017_2019",
        "covid_period_2020_2022",
    )
    (
        moon_period.groupby(["covid_period", YEAR_COL])
        .size()
        .reset_index(name="article_count")
        .sort_values(["covid_period", YEAR_COL])
        .to_csv(OUTPUT_DIR / "moon_covid_period_article_counts.csv", index=False, encoding="utf-8-sig")
    )

    metadata = {
        "data_path": str(DATA_PATH),
        "output_dir": str(OUTPUT_DIR),
        "random_seed": RANDOM_SEED,
        "conditions": [
            {
                "name": condition.name,
                "description": condition.description,
                "stopword_count": len(condition.stopwords),
            }
            for condition in conditions
        ],
        "tfidf": {
            "max_features": TFIDF_MAX_FEATURES,
            "min_df": TFIDF_MIN_DF,
            "max_df": TFIDF_MAX_DF,
            "tokenizer": "str.split",
            "park_corpus_fit_moon_corpus_transform": True,
        },
        "log_odds": {
            "prior": "combined corpus informative Dirichlet prior",
            "min_total_count": LOG_ODDS_MIN_COUNT,
            "positive_direction": MOON,
        },
        "lda": {
            "num_topics": LDA_NUM_TOPICS,
            "passes": LDA_PASSES,
            "iterations": LDA_ITERATIONS,
            "random_state": RANDOM_SEED,
            "implementation": "gensim.models.LdaModel",
            "workers": 1,
            "primary_comparison": "same-topic-space inference after COVID token removal",
        },
    }
    with open(OUTPUT_DIR / "reproducibility_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def main() -> None:
    print("[1/6] Loading data and stopword conditions...")
    df = load_data()
    conditions = build_conditions()
    base_stopwords = conditions[0].stopwords

    covid_audit = audit_covid_tokens(base_stopwords)
    covid_audit.to_csv(OUTPUT_DIR / "covid_token_stopword_audit.csv", index=False, encoding="utf-8-sig")

    print("[2/6] Running TF-IDF robustness...")
    tfidf_outputs = {
        condition.name: compute_tfidf_change(df, condition) for condition in conditions
    }
    for name, table in tfidf_outputs.items():
        table.to_csv(OUTPUT_DIR / f"tfidf_{name}.csv", index=False, encoding="utf-8-sig")
    tfidf_comparison = compare_keyword_tables(
        tfidf_outputs["original_base"],
        tfidf_outputs["covid_filtered"],
        score_col="importance",
        top_n=TFIDF_TOP_N,
    )
    tfidf_comparison.to_csv(
        OUTPUT_DIR / "tfidf_original_vs_covid_filtered_top100.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("[3/6] Running log-odds robustness...")
    log_outputs = {condition.name: compute_log_odds(df, condition) for condition in conditions}
    for name, table in log_outputs.items():
        table.to_csv(OUTPUT_DIR / f"log_odds_{name}.csv", index=False, encoding="utf-8-sig")
    log_comparison = compare_log_odds(
        log_outputs["original_base"], log_outputs["covid_filtered"], top_n=TFIDF_TOP_N
    )
    log_comparison.to_csv(
        OUTPUT_DIR / "log_odds_original_vs_covid_filtered_top100.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("[4/6] Running LDA robustness...")
    lda_outputs = run_lda_robustness(df, conditions)
    for name, table in lda_outputs.items():
        table.to_csv(OUTPUT_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")

    print("[5/6] Saving appendix workbook and reproducibility metadata...")
    stability_summary = summarize_stability(
        tfidf_comparison,
        log_comparison,
        lda_outputs["lda_gap_comparison"],
    )
    stability_summary.to_csv(
        OUTPUT_DIR / "robustness_stability_summary.csv", index=False, encoding="utf-8-sig"
    )
    save_metadata(df, conditions)

    save_excel(
        {
            "summary": stability_summary,
            "covid_token_audit": covid_audit,
            "tfidf_compare": tfidf_comparison,
            "log_odds_compare": log_comparison,
            "lda_gap_compare": lda_outputs["lda_gap_comparison"],
            "lda_coherence": lda_outputs["lda_coherence"],
        },
        OUTPUT_DIR / "appendix_covid_filtered_robustness.xlsx",
    )

    print("[6/6] Complete.")
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
