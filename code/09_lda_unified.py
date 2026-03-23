# ============================================================
# 09_lda_unified.py
# 비교가능성 확보를 위한 통합 LDA
#
# 기존 문제: 정부별 독립 LDA → 토픽 공간이 달라 비중 비교 불가
#
# 해결 방법:
#   전체 코퍼스(박근혜+문재인)로 단일 LDA 모델을 학습(통합 LDA)하고,
#   동일 토픽 공간 위에서 정부별 토픽 비중(θ)을 추출하여 비교.
#   → 토픽 공간이 동일하므로 통계적으로 비교 정합성 확보.
#
# 출력:
#   - unified_lda_topics.csv      : 토픽별 상위 15 키워드
#   - unified_topic_dist_by_gov.csv: 정부별 토픽 비중 (평균 θ)
#   - fig_unified_topic_dist.png  : 정부별 토픽 비중 막대 비교
#   - fig_unified_topic_keywords.png: 토픽별 키워드 바 차트
# ============================================================

import os
import pandas as pd
import numpy as np
from gensim import corpora
from gensim.models import LdaMulticore
from gensim.models.coherencemodel import CoherenceModel
from multiprocessing import cpu_count, freeze_support
from stopwords import STOPWORDS
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform
import warnings
warnings.filterwarnings("ignore")

# -------------------------------------------------------
# 0. 설정
# -------------------------------------------------------
if platform.system() == "Windows":
    font_path = "C:/Windows/Fonts/malgun.ttf"
    font_name = fm.FontProperties(fname=font_path).get_name()
    plt.rc("font", family=font_name)
else:
    plt.rc("font", family="NanumGothic")
plt.rcParams["axes.unicode_minus"] = False

NUM_TOPICS   = 10
PASSES       = 3
RANDOM_STATE = 42
WORKERS      = max(cpu_count() - 1, 1)
OUTPUT_DIR   = "../result/lda_unified"
DATA_PATH    = "../datas/preprocessed_2013_2022.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

STOPWORDS_SET = set(STOPWORDS(version="base"))

# -------------------------------------------------------
# 1. 데이터 로딩 및 전처리
# -------------------------------------------------------
def main():
    print("데이터 로딩 중...")
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df = df.dropna(subset=["text", "정부"])
    df = df[df["정부"].isin(["박근혜정부", "문재인정부"])]
    
    # 💡 속도 최적화를 위한 20만건 샘플링
    if len(df) > 200000:
        df = df.sample(200000, random_state=42)
        print(f"속도 최적화를 위해 200,000건 샘플링 완료")
        
    print(f"분석 대상 문서 수: {len(df):,}건")

    def tokenize(text):
        return [
            w for w in str(text).split()
            if len(w) > 1 and w not in STOPWORDS_SET
        ]

    df["tokens"] = df["text"].apply(tokenize)
    df = df[df["tokens"].apply(len) > 0]

    # -------------------------------------------------------
    # 2. 전체 코퍼스로 단일 사전 및 LDA 학습
    # -------------------------------------------------------
    print("\n사전 구축 중...")
    all_texts = df["tokens"].tolist()
    dictionary = corpora.Dictionary(all_texts)
    # 💡 사전 필터링 최적화
    dictionary.filter_extremes(no_below=20, no_above=0.3)
    print(f"사전 크기: {len(dictionary):,} 단어")

    all_corpus = [dictionary.doc2bow(t) for t in all_texts]
    df["corpus"] = all_corpus

    print(f"\n통합 LDA 학습 중... (K={NUM_TOPICS}, passes={PASSES})")
    lda = LdaMulticore(
        corpus=all_corpus,
        id2word=dictionary,
        num_topics=NUM_TOPICS,
        random_state=RANDOM_STATE,
        passes=PASSES,
        workers=WORKERS,
        alpha="symmetric",
        eta="auto"
    )

    # -------------------------------------------------------
    # 3. Coherence 측정 (c_v + u_mass)
    # -------------------------------------------------------
    print("\nCoherence 계산 중...")
    coherence_rows = []
    for coh_type in ["c_v", "u_mass"]:
        cm = CoherenceModel(
            model=lda, texts=all_texts,
            dictionary=dictionary,
            coherence=coh_type,
            processes=1
        )
        val = cm.get_coherence()
        per_topic = cm.get_coherence_per_topic()
        print(f"  {coh_type}: {val:.4f}")
        for i, v in enumerate(per_topic):
            coherence_rows.append({"모델": "통합LDA", "coherence_type": coh_type,
                                   "토픽": i, "coherence": round(v, 4)})

    coherence_df = pd.DataFrame(coherence_rows)
    coherence_df.to_csv(
        os.path.join(OUTPUT_DIR, "unified_lda_coherence.csv"),
        index=False, encoding="utf-8-sig"
    )
    print("저장: unified_lda_coherence.csv")

    # -------------------------------------------------------
    # 4. 토픽별 상위 15 키워드 저장
    # -------------------------------------------------------
    topic_words = []
    for tid in range(NUM_TOPICS):
        for word, prob in lda.show_topic(tid, topn=15):
            topic_words.append({"토픽": tid, "키워드": word, "확률": round(prob, 5)})

    topic_df = pd.DataFrame(topic_words)
    topic_df.to_csv(
        os.path.join(OUTPUT_DIR, "unified_lda_topics.csv"),
        index=False, encoding="utf-8-sig"
    )
    print("저장: unified_lda_topics.csv")

    # -------------------------------------------------------
    # 5. 정부별 토픽 비중 (θ) 계산 — 동일 공간이므로 비교 정합
    # -------------------------------------------------------
    print("\n정부별 토픽 비중 계산 중...")

    def get_doc_topic_dist(bow):
        dist = np.zeros(NUM_TOPICS)
        for tid, prob in lda.get_document_topics(bow, minimum_probability=0):
            dist[tid] = prob
        return dist

    df["topic_dist"] = df["corpus"].apply(get_doc_topic_dist)

    results = []
    for gov in ["박근혜정부", "문재인정부"]:
        sub = df[df["정부"] == gov]
        mat = np.vstack(sub["topic_dist"].values)
        mean_theta = mat.mean(axis=0)
        for tid in range(NUM_TOPICS):
            results.append({
                "정부": gov,
                "토픽": tid,
                "평균_토픽비중(θ)": round(mean_theta[tid], 5)
            })

    dist_df = pd.DataFrame(results)
    dist_df.to_csv(
        os.path.join(OUTPUT_DIR, "unified_topic_dist_by_gov.csv"),
        index=False, encoding="utf-8-sig"
    )
    print("저장: unified_topic_dist_by_gov.csv")

    # -------------------------------------------------------
    # 6. 시각화: 정부별 토픽 비중 비교 막대 차트
    # -------------------------------------------------------
    pivot = dist_df.pivot(index="토픽", columns="정부", values="평균_토픽비중(θ)").fillna(0)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(NUM_TOPICS)
    width = 0.35
    ax.bar(x - width/2, pivot.get("박근혜정부", 0), width, label="박근혜정부", color="#4C72B0")
    ax.bar(x + width/2, pivot.get("문재인정부", 0), width, label="문재인정부", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels([f"T{i}" for i in range(NUM_TOPICS)])
    ax.set_xlabel("토픽 (통합 LDA)")
    ax.set_ylabel("평균 토픽 비중 (θ)")
    ax.set_title("정부별 토픽 비중 비교 (통합 LDA 기반, 비교 정합)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_unified_topic_dist.png"), dpi=300)
    plt.close()
    print("저장: fig_unified_topic_dist.png")

    # -------------------------------------------------------
    # 7. 토픽별 키워드 바 차트
    # -------------------------------------------------------
    fig, axes = plt.subplots(2, 5, figsize=(20, 10))
    for tid, ax in enumerate(axes.flatten()):
        sub = topic_df[topic_df["토픽"] == tid].head(10)
        ax.barh(sub["키워드"][::-1], sub["확률"][::-1])
        ax.set_title(f"Topic {tid}")
        ax.set_xlabel("P(w|t)")
    plt.suptitle("통합 LDA 토픽별 상위 10 키워드")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_unified_topic_keywords.png"), dpi=300)
    plt.close()
    print("저장: fig_unified_topic_keywords.png")

    print("\n[완료] 09_lda_unified.py 실행 완료")
    print(f"결과 디렉토리: {OUTPUT_DIR}")


if __name__ == "__main__":
    freeze_support()
    main()
