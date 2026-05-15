import pandas as pd
from collections import Counter
import numpy as np
from pathlib import Path
from stopwords import STOPWORDS

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "datas" / "preprocessed_2013_2022.csv"
RESULT_DIR = BASE_DIR / "result"
STOPWORDS_LIST = set(STOPWORDS(version="base"))

def count_words(texts, stopwords):
    counter = Counter()
    for t in texts:
        words = [
            w for w in str(t).split()
            if w not in stopwords and len(w) > 1
        ]
        counter.update(words)
    return counter

def log_odds(count1, count2, prior):
    vocab = set(count1) | set(count2)
    results = []

    for w in vocab:
        c1 = count1.get(w, 0)
        c2 = count2.get(w, 0)
        p = prior.get(w, 1)

        lo = np.log((c1 + p) / (c2 + p))
        var = 1 / (c1 + p) + 1 / (c2 + p)
        z = lo / np.sqrt(var)

        results.append((w, z))

    return pd.DataFrame(results, columns=["단어", "log_odds_z"])

if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig", usecols=["정부", "text"])

    moon_docs = df[df["정부"] == "문재인정부"]["text"].dropna()
    park_docs = df[df["정부"] == "박근혜정부"]["text"].dropna()

    moon_counts = count_words(moon_docs, STOPWORDS_LIST)
    park_counts = count_words(park_docs, STOPWORDS_LIST)

    prior = moon_counts + park_counts

    log_df = log_odds(moon_counts, park_counts, prior)

    log_df["강화_정부"] = log_df["log_odds_z"].apply(
        lambda x: "문재인정부" if x > 0 else "박근혜정부"
    )

    log_df.sort_values("log_odds_z", ascending=False)\
          .head(50)\
          .to_csv(RESULT_DIR / "log_odds_top_moon.csv", index=False, encoding="utf-8-sig")

    log_df.sort_values("log_odds_z")\
          .head(50)\
          .to_csv(RESULT_DIR / "log_odds_top_park.csv", index=False, encoding="utf-8-sig")
