import os
import pandas as pd
import random
from gensim import corpora
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from multiprocessing import freeze_support
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform

# 폰트 설정
if platform.system() == "Windows":
    font_path = "C:/Windows/Fonts/malgun.ttf"
    font_name = fm.FontProperties(fname=font_path).get_name()
    plt.rc("font", family=font_name)
else:
    plt.rc("font", family="NanumGothic")
plt.rcParams["axes.unicode_minus"] = False

def main():
    print(" 데이터 로딩 중...")
    df = pd.read_csv("../datas/preprocessed_2013_2022.csv", encoding="utf-8-sig")

    texts = [
        str(doc).split()
        for doc in df["text"].dropna()
    ]

    print(f" 전체 문서 수: {len(texts)}")

    # 🔹 샘플링
    sample_size = 50000
    # Seed 42 for reproducibility as mentioned in the paper
    random.seed(42)
    texts = random.sample(texts, sample_size)
    print(f" 샘플링 문서 수: {len(texts)}")

    # 🔹 사전 & 코퍼스
    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=20, no_above=0.5)
    corpus = [dictionary.doc2bow(text) for text in texts]

    results = []

    for k in range(6, 13):
        print(f" 토픽 수 {k} 계산 중...")

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

        cm_cv = CoherenceModel(
            model=lda,
            texts=texts,
            dictionary=dictionary,
            coherence="c_v",
            processes=1   
        )
        cm_umass = CoherenceModel(
            model=lda,
            texts=texts,
            dictionary=dictionary,
            coherence="u_mass",
            processes=1   
        )

        cv_score = cm_cv.get_coherence()
        umass_score = cm_umass.get_coherence()
        print(f"   → c_v: {cv_score:.4f}, u_mass: {umass_score:.4f}")

        results.append({
            "num_topics": k, 
            "c_v": cv_score,
            "u_mass": umass_score
        })

    result_df = pd.DataFrame(results)
    result_df.to_csv("../result/topic_coherence_dual.csv", index=False)
    print(" 토픽 수 평가 완료: ../result/topic_coherence_dual.csv")

    # 🔹 차트 그리기
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('토픽 수 (k)')
    ax1.set_ylabel('c_v (Coherence)', color=color)
    ax1.plot(result_df["num_topics"], result_df["c_v"], marker='o', color=color, label='c_v')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(range(6, 13))

    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('u_mass (Coherence)', color=color)  
    ax2.plot(result_df["num_topics"], result_df["u_mass"], marker='s', color=color, label='u_mass')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  
    plt.title("<그림 3> 토픽 수(k)에 따른 coherence 점수 비교 (c_v, u_mass)")
    
    # Legend combining both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.savefig("../result/figure3_coherence_k.png", dpi=300)
    plt.close()
    print(" 차트 저장 완료: ../result/figure3_coherence_k.png")

if __name__ == "__main__":
    freeze_support() 
    main()

