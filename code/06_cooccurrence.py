# -*- coding: utf-8 -*-

import pandas as pd
import os
import math
from itertools import combinations
from collections import Counter
from stopwords import STOPWORDS

# ===============================
# 1. 설정
# ===============================
STOPWORDS_LIST = set(STOPWORDS())
output_dir = "../result"
os.makedirs(output_dir, exist_ok=True)

# ===============================
# 2. PMI 계산 함수
# ===============================
def compute_pmi(documents, keywords):
    total_docs = len(documents)
    word_doc_freq = Counter()
    pair_counter = Counter()

    for words in documents:
        filtered = words & keywords
        word_doc_freq.update(filtered)
        if len(filtered) >= 2:
            pair_counter.update(combinations(sorted(filtered), 2))

    pmi = {}
    for (w1, w2), co_cnt in pair_counter.items():
        p_xy = co_cnt / total_docs
        p_x = word_doc_freq[w1] / total_docs
        p_y = word_doc_freq[w2] / total_docs

        if p_xy > 0 and p_x > 0 and p_y > 0:
            pmi[(w1, w2)] = math.log2(p_xy / (p_x * p_y))

    return pmi

# ===============================
# 3. 차별 공출현 테이블 생성
# ===============================
def build_differential_cooccurrence_table(
    text_df,
    keywords,
    min_delta_pmi=0.3,
    top_n=30
):
    print("\n📌 차별 공출현 테이블 생성 시작")

    docs = {}
    for gov in ["박근혜정부", "문재인정부"]:
        gov_docs = []
        for doc in text_df[text_df["정부"] == gov]["text"].dropna():
            words = {
                w for w in str(doc).split()
                if w in keywords and w not in STOPWORDS_LIST
            }
            if words:
                gov_docs.append(words)
        docs[gov] = gov_docs
        print(f" - {gov}: 문서 수 {len(gov_docs)}")

    pmi_pk = compute_pmi(docs["박근혜정부"], keywords)
    pmi_mj = compute_pmi(docs["문재인정부"], keywords)

    rows = []
    all_pairs = set(pmi_pk.keys()) | set(pmi_mj.keys())

    for w1, w2 in all_pairs:
        delta = pmi_mj.get((w1, w2), 0) - pmi_pk.get((w1, w2), 0)
        if abs(delta) >= min_delta_pmi:
            rows.append({
                "키워드1": w1,
                "키워드2": w2,
                "PMI_박근혜": round(pmi_pk.get((w1, w2), 0), 3),
                "PMI_문재인": round(pmi_mj.get((w1, w2), 0), 3),
                "ΔPMI": round(delta, 3),
                "강화_시기": "문재인정부" if delta > 0 else "박근혜정부"
            })

    df = pd.DataFrame(rows).sort_values("ΔPMI", ascending=False)

    # 상·하위 고정
    df_final = pd.concat([
        df.head(top_n),
        df.tail(top_n)
    ])

    save_path = os.path.join(output_dir, "differential_cooccurrence_table.csv")
    df_final.to_csv(save_path, index=False, encoding="utf-8-sig")

    print(f"✅ 저장 완료: {save_path}")
    return df_final

# ===============================
# 4. 실행부
# ===============================
if __name__ == "__main__":
    text_df = pd.read_csv(
        "../datas/preprocessed_2013_2022.csv",
        encoding="utf-8-sig"
    )

    # 🔑 TF-IDF 증감률 상위 키워드 (논문 기준 고정)
    KEYWORDS = {
        "공정", "규제", "세대", "공공", "근무", "체계", "인재", "인력",
        "청년", "여성", "인권", "소통", "참여", "조직", "개혁",
        "공무원", "연금", "노동", "성과", "혁신",
        "임금", "인사", "방침", "갈등", "경쟁력", "중심"
    }

    df_result = build_differential_cooccurrence_table(
        text_df=text_df,
        keywords=KEYWORDS
    )

    print("\n🔎 상위 ΔPMI (문재인 ↑)")
    print(df_result.head(10))

    print("\n🔎 하위 ΔPMI (박근혜 ↑)")
    print(df_result.tail(10))
