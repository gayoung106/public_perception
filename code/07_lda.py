# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import os

# ===============================
# 1. 데이터 로드
# ===============================
df = pd.read_csv("../datas/preprocessed_2013_2022.csv", encoding="utf-8-sig")

output_dir = "../result"
os.makedirs(output_dir, exist_ok=True)

# ===============================
# 2. 정부별 토픽모델링 함수
# ===============================
def run_lda_by_government(
    df,
    gov_name,
    n_topics=10,
    max_features=3000
):
    print(f"\n📌 토픽모델링 시작: {gov_name}")

    texts = df[df["정부"] == gov_name]["text"].dropna().tolist()
    print(f" - 문서 수: {len(texts)}")

    vectorizer = CountVectorizer(
        max_df=0.9,
        min_df=20,
        max_features=max_features
    )

    X = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        learning_method="batch"
    )

    lda.fit(X)

    # ---------------------------
    # 토픽별 상위 단어 저장
    # ---------------------------
    topic_words = []

    for topic_idx, topic in enumerate(lda.components_):
        top_words = [
            feature_names[i]
            for i in topic.argsort()[:-16:-1]
        ]
        topic_words.append({
            "정부": gov_name,
            "토픽": f"Topic {topic_idx+1}",
            "상위단어": ", ".join(top_words)
        })

    result_df = pd.DataFrame(topic_words)
    save_path = os.path.join(output_dir, f"lda_topics_{gov_name}.csv")
    result_df.to_csv(save_path, index=False, encoding="utf-8-sig")

    print(f"✅ 토픽 결과 저장: {save_path}")

# ===============================
# 3. 실행
# ===============================
if __name__ == "__main__":
    run_lda_by_government(df, "박근혜정부", n_topics=10)
    run_lda_by_government(df, "문재인정부", n_topics=10)
