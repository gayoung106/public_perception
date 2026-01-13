# -*- coding: utf-8 -*-

import pandas as pd
import random
from gensim import corpora
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from multiprocessing import freeze_support

def main():
    print("📌 데이터 로딩 중...")
    df = pd.read_csv("../datas/preprocessed_2013_2022.csv", encoding="utf-8-sig")

    texts = [
        str(doc).split()
        for doc in df["text"].dropna()
    ]

    print(f"📌 전체 문서 수: {len(texts)}")

    # 🔹 샘플링
    sample_size = 50000
    texts = random.sample(texts, sample_size)
    print(f"📌 샘플링 문서 수: {len(texts)}")

    # 🔹 사전 & 코퍼스
    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=20, no_above=0.5)
    corpus = [dictionary.doc2bow(text) for text in texts]

    results = []

    for k in range(6, 13):
        print(f"🔄 토픽 수 {k} 계산 중...")

        lda = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=k,
            random_state=42,
            passes=10,
            iterations=200,
            alpha="auto",
            eta="auto"
        )

        coherence_model = CoherenceModel(
            model=lda,
            texts=texts,
            dictionary=dictionary,
            coherence="c_v",
            processes=1   # 🔥 핵심: 멀티프로세스 차단
        )

        coherence = coherence_model.get_coherence()
        print(f"   → Coherence: {coherence:.4f}")

        results.append({"num_topics": k, "coherence": coherence})

    result_df = pd.DataFrame(results)
    result_df.to_csv("../result/topic_coherence.csv", index=False)
    print("✅ 토픽 수 평가 완료: ../result/topic_coherence.csv")

if __name__ == "__main__":
    freeze_support()   # 🔥 Windows 필수
    main()
