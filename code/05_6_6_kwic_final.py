import pandas as pd
import random

# ===============================
# 1. 설정
# ===============================
DATA_PATH = "../datas/preprocessed_2013_2022.csv"

MOON_KEYWORDS_PATH = "../result/final_keywords_moon.csv"
PARK_KEYWORDS_PATH = "../result/final_keywords_park.csv"

SAMPLES_PER_KEYWORD = 3   # 키워드당 문장 수 (논문용: 2~3 권장)
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

# ===============================
# 2. KWIC 추출 함수
# ===============================
def extract_kwic(df, keyword, gov, n_samples):
    subset = df[
        (df["정부"] == gov) &
        (df["text"].str.contains(keyword, na=False))
    ]

    if len(subset) == 0:
        return []

    samples = subset.sample(
        n=min(n_samples, len(subset)),
        random_state=RANDOM_SEED
    )

    return samples["text"].tolist()

# ===============================
# 3. 실행부
# ===============================
if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

    moon_keywords = pd.read_csv(MOON_KEYWORDS_PATH)["단어"].tolist()
    park_keywords = pd.read_csv(PARK_KEYWORDS_PATH)["단어"].tolist()

    results = []

    # 문재인정부 KWIC
    for kw in moon_keywords:
        texts = extract_kwic(df, kw, "문재인정부", SAMPLES_PER_KEYWORD)
        for t in texts:
            results.append({
                "정부": "문재인정부",
                "키워드": kw,
                "문장": t
            })

    # 박근혜정부 KWIC
    for kw in park_keywords:
        texts = extract_kwic(df, kw, "박근혜정부", SAMPLES_PER_KEYWORD)
        for t in texts:
            results.append({
                "정부": "박근혜정부",
                "키워드": kw,
                "문장": t
            })

    result_df = pd.DataFrame(results)

    result_df[result_df["정부"] == "문재인정부"] \
        .to_csv("../result/kwic_final_moon.csv", index=False, encoding="utf-8-sig")

    result_df[result_df["정부"] == "박근혜정부"] \
        .to_csv("../result/kwic_final_park.csv", index=False, encoding="utf-8-sig")

    print("최종 KWIC 추출 완료")
