import pandas as pd
import random

# ===============================
# 1. 설정
# ===============================
DATA_PATH = "../datas/preprocessed_2013_2022.csv"

TARGET_PAIRS = [
    ("개혁", "인권", "문재인정부"),
    ("공무원", "연금", "박근혜정부")
]

SAMPLES_PER_PAIR = 5
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

# ===============================
# 2. 콘코드 추출 함수
# ===============================
def extract_concordance(df, w1, w2, gov, n_samples=5):
    subset = df[
        (df["정부"] == gov) &
        (df["text"].str.contains(w1, na=False)) &
        (df["text"].str.contains(w2, na=False))
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

    results = []

    for w1, w2, gov in TARGET_PAIRS:
        texts = extract_concordance(df, w1, w2, gov, SAMPLES_PER_PAIR)
        for t in texts:
            results.append({
                "정부": gov,
                "키워드1": w1,
                "키워드2": w2,
                "문장": t
            })

    result_df = pd.DataFrame(results)

    save_path = "../result/concordance_samples.csv"
    result_df.to_csv(save_path, index=False, encoding="utf-8-sig")

    print(f"콘코드 샘플 저장 완료: {save_path}")
