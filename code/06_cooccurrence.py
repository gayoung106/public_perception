import pandas as pd
import os
import math
from itertools import combinations
from collections import Counter

# ===============================
# 1. 기본 설정
# ===============================
OUTPUT_DIR = "../result"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===============================
# 2. PMI 계산 함수
# ===============================
def compute_pmi(documents, keywords, min_co_cnt=10):
    """
    documents : list[set[str]]  # 문서별 키워드 집합
    keywords  : set[str]
    min_co_cnt: 최소 공동출현 문서 수
    """
    total_docs = len(documents)
    word_doc_freq = Counter()
    pair_counter = Counter()

    # 문서 단위 빈도 계산
    for words in documents:
        filtered = words & keywords
        word_doc_freq.update(filtered)

        if len(filtered) >= 2:
            pair_counter.update(combinations(sorted(filtered), 2))

    pmi_result = {}

    for (w1, w2), co_cnt in pair_counter.items():
        if co_cnt < min_co_cnt:
            continue

        p_xy = co_cnt / total_docs
        p_x = word_doc_freq[w1] / total_docs
        p_y = word_doc_freq[w2] / total_docs

        if p_xy > 0 and p_x > 0 and p_y > 0:
            pmi_value = math.log2(p_xy / (p_x * p_y))
            pmi_result[(w1, w2)] = {
                "pmi": pmi_value,
                "co_cnt": co_cnt
            }

    return pmi_result

# ===============================
# 3. 차별 공출현 테이블 생성
# ===============================
def build_differential_cooccurrence_table(
    text_df,
    keywords,
    min_delta_pmi=0.3,
    min_co_cnt=10,
    top_n=30
):
    print("\n차별 공출현(PMI) 분석 시작")

    # 정부별 문서 구성
    docs = {}
    for gov in ["박근혜정부", "문재인정부"]:
        gov_docs = []

        for doc in text_df[text_df["정부"] == gov]["text"].dropna():
            words = set(str(doc).split()) & keywords
            if len(words) >= 2:
                gov_docs.append(words)

        docs[gov] = gov_docs
        print(f" - {gov}: 문서 수 {len(gov_docs)}")

    # PMI 계산
    pmi_pk = compute_pmi(docs["박근혜정부"], keywords, min_co_cnt)
    pmi_mj = compute_pmi(docs["문재인정부"], keywords, min_co_cnt)

    rows = []
    all_pairs = set(pmi_pk.keys()) | set(pmi_mj.keys())

    for (w1, w2) in all_pairs:
        pk = pmi_pk.get((w1, w2), {"pmi": 0, "co_cnt": 0})
        mj = pmi_mj.get((w1, w2), {"pmi": 0, "co_cnt": 0})

        delta = mj["pmi"] - pk["pmi"]

        if abs(delta) >= min_delta_pmi:
            rows.append({
                "키워드1": w1,
                "키워드2": w2,
                "PMI_박근혜": round(pk["pmi"], 3),
                "PMI_문재인": round(mj["pmi"], 3),
                "ΔPMI": round(delta, 3),
                "공동출현_박근혜": pk["co_cnt"],
                "공동출현_문재인": mj["co_cnt"],
                "강화_시기": "문재인정부" if delta > 0 else "박근혜정부"
            })

    df = pd.DataFrame(rows).sort_values("ΔPMI", ascending=False)

    # 상·하위 프레임 동시 제시
    df_final = pd.concat([
        df.head(top_n),
        df.tail(top_n)
    ])

    save_path = os.path.join(
        OUTPUT_DIR,
        "differential_cooccurrence_table.csv"
    )
    df_final.to_csv(save_path, index=False, encoding="utf-8-sig")

    print(f"저장 완료: {save_path}")
    return df_final

# ===============================
# 4. 실행부
# ===============================
if __name__ == "__main__":
    text_df = pd.read_csv(
        "../datas/preprocessed_2013_2022.csv",
        encoding="utf-8-sig"
    )

    # (A) 공통 제도·조직 프레임
    CORE_FRAME = {
        "공무원", "조직", "성과", "인사", "근무",
        "체계", "연금", "노동", "개혁", "혁신"
    }

    # (B) 문재인정부 강화 프레임
    MJ_UP = {
        "공정", "인권", "소통", "참여", "규제",
        "청년", "여성", "공공", "데이터", "디지털"
    }

    # (C) 박근혜정부 강화 프레임
    PK_UP = {
        "연금", "노조", "경영", "승진", "조직",
        "공무원", "개혁", "노동"
    }

    KEYWORDS = CORE_FRAME | MJ_UP | PK_UP

    df_result = build_differential_cooccurrence_table(
        text_df=text_df,
        keywords=KEYWORDS,
        min_delta_pmi=0.3,
        min_co_cnt=10,
        top_n=30
    )

    print("\n[문재인정부 강화 공출현 프레임]")
    print(df_result[df_result["강화_시기"] == "문재인정부"].head(10))

    print("\n[박근혜정부 강화 공출현 프레임]")
    print(df_result[df_result["강화_시기"] == "박근혜정부"].head(10))
