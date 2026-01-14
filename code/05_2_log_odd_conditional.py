import pandas as pd
import numpy as np
from collections import Counter
from scipy.stats import norm
from stopwords import STOPWORDS

# ===============================
# 0. 버전 선택
# ===============================
VERSION = "with_conditional"  
# VERSION = "with_conditional"

STOPWORDS_LIST = set(STOPWORDS(version=VERSION))

# ===============================
# 1. 데이터 로드
# ===============================
df = pd.read_csv("../datas/preprocessed_2013_2022.csv", encoding="utf-8-sig")

park_docs = df[df["정부"] == "박근혜정부"]["text"].dropna().astype(str)
moon_docs = df[df["정부"] == "문재인정부"]["text"].dropna().astype(str)

# ===============================
# 2. 단어 카운트 (불용어 적용)
# ===============================
def count_words(docs, stopwords):
    counter = Counter()
    for doc in docs:
        words = [
            w for w in doc.split()
            if w not in stopwords and len(w) > 1
        ]
        counter.update(words)
    return counter

park_counts = count_words(park_docs, STOPWORDS_LIST)
moon_counts = count_words(moon_docs, STOPWORDS_LIST)

# 공통 단어 집합
vocab = set(park_counts) | set(moon_counts)

# informative prior
prior = Counter()
prior.update(park_counts)
prior.update(moon_counts)

# ===============================
# 3. log-odds with Dirichlet prior
# ===============================
rows = []

alpha_0 = sum(prior.values())
n1 = sum(moon_counts.values())
n2 = sum(park_counts.values())

for word in vocab:
    c1 = moon_counts.get(word, 0)
    c2 = park_counts.get(word, 0)
    c_prior = prior[word]

    # 희소 단어 제거
    if c1 + c2 < 30:
        continue

    log_odds = (
        np.log((c1 + c_prior) / (n1 - c1 + alpha_0 - c_prior)) -
        np.log((c2 + c_prior) / (n2 - c2 + alpha_0 - c_prior))
    )

    var = (1 / (c1 + c_prior)) + (1 / (c2 + c_prior))
    z = log_odds / np.sqrt(var)

    rows.append({
        "단어": word,
        "log_odds_z": z,
        "강화_정부": "문재인정부" if z > 0 else "박근혜정부"
    })

logodds_df = (
    pd.DataFrame(rows)
    .sort_values("log_odds_z", ascending=False)
)

# ===============================
# 4. 결과 저장
# ===============================
logodds_df.to_csv(
    f"../result/log_odds_keywords_{VERSION}.csv",
    index=False,
    encoding="utf-8-sig"
)

print(f"log-odds 분석 완료 ({VERSION})")
print(logodds_df.head(10))
print(logodds_df.tail(10))
