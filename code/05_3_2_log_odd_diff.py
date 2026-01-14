import pandas as pd
from collections import Counter
import numpy as np
from scipy.stats import norm

DATA_PATH = "../datas/preprocessed_2013_2022.csv"

def count_words(texts):
    counter = Counter()
    for t in texts:
        counter.update(t.split())
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
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

    moon_docs = df[df["정부"] == "문재인정부"]["text"].dropna()
    park_docs = df[df["정부"] == "박근혜정부"]["text"].dropna()

    moon_counts = count_words(moon_docs)
    park_counts = count_words(park_docs)

    prior = moon_counts + park_counts

    log_df = log_odds(moon_counts, park_counts, prior)

    log_df["강화_정부"] = log_df["log_odds_z"].apply(
        lambda x: "문재인정부" if x > 0 else "박근혜정부"
    )

    log_df.sort_values("log_odds_z", ascending=False)\
          .head(50)\
          .to_csv("../result/log_odds_top_moon.csv", index=False, encoding="utf-8-sig")

    log_df.sort_values("log_odds_z")\
          .head(50)\
          .to_csv("../result/log_odds_top_park.csv", index=False, encoding="utf-8-sig")
