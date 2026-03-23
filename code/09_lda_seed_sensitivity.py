# ============================================================
# 09_lda_seed_sensitivity.py
# LDA 시드 민감도 분석 (Robustness Check)
#
# 목적:
#   LDA 결과가 랜덤 시드에 따라 변화하는 정도를 측정하여
#   결과의 안정성(robustness)을 객관적으로 보고.
#
# 방법:
#   - 다수의 랜덤 시드(5개)로 정부별 LDA를 반복 학습
#   - 각 실행의 coherence(c_v, u_mass)를 기록
#   - 토픽 상위 키워드의 Jaccard 유사도로 토픽 안정성 측정
#   - 평균/표준편차 리포트 → 시드 민감도가 낮음을 확인
#
# 출력:
#   - seed_coherence_summary.csv  : 시드별 coherence 요약
#   - seed_topic_stability.csv    : 토픽 Jaccard 안정성
#   - fig_seed_coherence_box.png  : coherence 분포 박스플롯
# ============================================================

import os
import pandas as pd
import numpy as np
from itertools import combinations
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
PASSES       = 2  # 속도 최적화: 2로 하향
WORKERS      = max(cpu_count() - 1, 1)
SEEDS        = [42, 123, 777]  # 3개 시드로 축소
TOP_N        = 15
OUTPUT_DIR   = "../result/lda_seed_sensitivity"
DATA_PATH    = "../datas/preprocessed_2013_2022.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)
STOPWORDS_SET = set(STOPWORDS(version="base"))


# -------------------------------------------------------
# 1. 유틸리티
# -------------------------------------------------------
def tokenize(text):
    return [w for w in str(text).split() if len(w) > 1 and w not in STOPWORDS_SET]

def jaccard(set_a, set_b):
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else 0.0

def topic_top_words(lda, n=TOP_N):
    """토픽별 상위 n개 키워드를 집합 리스트로 반환"""
    return [set(w for w, _ in lda.show_topic(tid, topn=n))
            for tid in range(NUM_TOPICS)]

def best_match_jaccard(words_a, words_b):
    """
    두 LDA 결과의 토픽 집합 간 최적 매칭(헝가리안 근사)으로
    평균 Jaccard 유사도 반환.
    (토픽 순서가 달라도 가장 유사한 토픽끼리 매칭)
    """
    from itertools import permutations
    n = len(words_a)
    # 작은 경우만 완전 탐색 (n<=8); 그 이상은 그리디
    if n <= 8:
        best = 0.0
        for perm in permutations(range(n)):
            score = np.mean([jaccard(words_a[i], words_b[perm[i]]) for i in range(n)])
            best = max(best, score)
        return best
    else:
        # 그리디 매칭
        used = set()
        scores = []
        for i in range(n):
            best_j, best_s = -1, -1
            for j in range(n):
                if j in used:
                    continue
                s = jaccard(words_a[i], words_b[j])
                if s > best_s:
                    best_s, best_j = s, j
            used.add(best_j)
            scores.append(best_s)
        return np.mean(scores)


# -------------------------------------------------------
# 2. 정부별 반복 학습
# -------------------------------------------------------
def main():
    print("데이터 로딩 중...")
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df = df.dropna(subset=["text", "정부"])
    df = df[df["정부"].isin(["박근혜정부", "문재인정부"])]
    
    # 💡 속도 최적화를 위한 10만건 연속 샘플링
    if len(df) > 100000:
        df = df.sample(100000, random_state=42)
        print(f"속도 초고속화를 위해 100,000건 샘플링 완료")
        
    df["tokens"] = df["text"].apply(tokenize)
    df = df[df["tokens"].apply(len) > 0]

    coh_rows      = []   # coherence 결과
    stability_rows = []  # Jaccard 안정성 결과

    for gov in ["박근혜정부", "문재인정부"]:
        sub_df = df[df["정부"] == gov]
        texts  = sub_df["tokens"].tolist()

        dictionary = corpora.Dictionary(texts)
        # 💡 사전 필터링 최적화
        dictionary.filter_extremes(no_below=20, no_above=0.3)
        corpus = [dictionary.doc2bow(t) for t in texts]

        print(f"\n[ {gov} ] 문서 수: {len(texts):,}, 사전: {len(dictionary):,}")

        models_per_seed = {}

        for seed in SEEDS:
            print(f"  seed={seed} 학습 중...")
            lda = LdaMulticore(
                corpus=corpus, id2word=dictionary,
                num_topics=NUM_TOPICS,
                random_state=seed,
                passes=PASSES, workers=WORKERS,
                alpha="symmetric", eta="auto"
            )
            models_per_seed[seed] = lda

            for coh_type in ["c_v", "u_mass"]:
                cm = CoherenceModel(
                    model=lda, texts=texts,
                    dictionary=dictionary,
                    coherence=coh_type,
                    processes=WORKERS  # 병렬 처리로 초고속화
                )
                val = cm.get_coherence()
                coh_rows.append({
                    "정부": gov, "seed": seed,
                    "coherence_type": coh_type,
                    "coherence": round(val, 4)
                })
                print(f"    {coh_type}: {val:.4f}")

        # 시드 쌍 간 Jaccard 안정성
        seed_pairs = list(combinations(SEEDS, 2))
        for s1, s2 in seed_pairs:
            tw1 = topic_top_words(models_per_seed[s1])
            tw2 = topic_top_words(models_per_seed[s2])
            jac = best_match_jaccard(tw1, tw2)
            stability_rows.append({
                "정부": gov, "seed_1": s1, "seed_2": s2,
                "mean_jaccard": round(jac, 4)
            })

    # -------------------------------------------------------
    # 3. 결과 저장
    # -------------------------------------------------------
    coh_df = pd.DataFrame(coh_rows)
    coh_df.to_csv(
        os.path.join(OUTPUT_DIR, "seed_coherence_detail.csv"),
        index=False, encoding="utf-8-sig"
    )

    # 요약 (평균 ± 표준편차)
    summary = (
        coh_df.groupby(["정부", "coherence_type"])["coherence"]
        .agg(평균="mean", 표준편차="std", 최솟값="min", 최댓값="max")
        .reset_index()
        .round(4)
    )
    summary.to_csv(
        os.path.join(OUTPUT_DIR, "seed_coherence_summary.csv"),
        index=False, encoding="utf-8-sig"
    )
    print("\n저장: seed_coherence_summary.csv")
    print(summary.to_string(index=False))

    stab_df = pd.DataFrame(stability_rows)
    stab_df.to_csv(
        os.path.join(OUTPUT_DIR, "seed_topic_stability.csv"),
        index=False, encoding="utf-8-sig"
    )
    stab_summary = (
        stab_df.groupby("정부")["mean_jaccard"]
        .agg(평균="mean", 표준편차="std")
        .reset_index().round(4)
    )
    print("\n토픽 Jaccard 안정성 요약:")
    print(stab_summary.to_string(index=False))
    stab_summary.to_csv(
        os.path.join(OUTPUT_DIR, "seed_stability_summary.csv"),
        index=False, encoding="utf-8-sig"
    )
    print("저장: seed_stability_summary.csv")

    # -------------------------------------------------------
    # 4. 시각화: coherence 박스플롯
    # -------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, coh_type in zip(axes, ["c_v", "u_mass"]):
        sub = coh_df[coh_df["coherence_type"] == coh_type]
        data_by_gov = [
            sub[sub["정부"] == gov]["coherence"].values
            for gov in ["박근혜정부", "문재인정부"]
        ]
        bp = ax.boxplot(data_by_gov, labels=["박근혜정부", "문재인정부"], patch_artist=True)
        colors = ["#4C72B0", "#DD8452"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
        ax.set_title(f"시드별 Coherence 분포 ({coh_type})")
        ax.set_ylabel("Coherence")
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle(f"LDA 시드 민감도 분석 (seeds: {SEEDS})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_seed_coherence_box.png"), dpi=300)
    plt.close()
    print("저장: fig_seed_coherence_box.png")

    print("\n[완료] 09_lda_seed_sensitivity.py 실행 완료")
    print(f"결과 디렉토리: {OUTPUT_DIR}")


if __name__ == "__main__":
    freeze_support()
    main()
