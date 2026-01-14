import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_PATH = "../datas/preprocessed_2013_2022.csv"
TOP_N = 50

def compute_tfidf(df, gov):
    corpus = df[df["정부"] == gov]["text"].dropna().tolist()

    vectorizer = TfidfVectorizer(
        tokenizer=str.split,
        token_pattern=None,   # 🔹 경고 제거용
        min_df=5,
        max_df=0.8
    )

    tfidf = vectorizer.fit_transform(corpus)
    scores = tfidf.mean(axis=0).A1
    words = vectorizer.get_feature_names_out()

    result = pd.DataFrame({
        "단어": words,
        "importance": scores
    }).sort_values("importance", ascending=False)

    return result.head(TOP_N)

if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

    tfidf_moon = compute_tfidf(df, "문재인정부")
    tfidf_park = compute_tfidf(df, "박근혜정부")

    tfidf_moon.to_csv("../result/tfidf_moon_top50.csv", index=False, encoding="utf-8-sig")
    tfidf_park.to_csv("../result/tfidf_park_top50.csv", index=False, encoding="utf-8-sig")
