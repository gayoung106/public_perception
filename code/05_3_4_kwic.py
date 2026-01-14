import pandas as pd
import random
import re

DATA_PATH = "../datas/preprocessed_2013_2022.csv"
TARGET_WORDS = {
    "문재인정부": ["플랫폼", "뉴딜"],
    "박근혜정부": ["연금", "공무원"]
}

SAMPLES = 7
random.seed(42)

def split_sentences(text):
    return [s.strip() for s in re.split(r"[.!?。]", text) if s.strip()]

def extract_kwic(df, word, gov):
    pattern = rf"\b{re.escape(word)}\b"
    gov_df = df[df["정부"] == gov]

    sents = []
    for t in gov_df["text"]:
        for s in split_sentences(t):
            if re.search(pattern, s):
                sents.append(s)

    return random.sample(sents, min(SAMPLES, len(sents)))

if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    results = []

    for gov, words in TARGET_WORDS.items():
        for w in words:
            for sent in extract_kwic(df, w, gov):
                results.append({
                    "정부": gov,
                    "키워드": w,
                    "문장": sent
                })

    pd.DataFrame(results).to_csv(
        "../result/kwic_samples.csv",
        index=False,
        encoding="utf-8-sig"
    )
