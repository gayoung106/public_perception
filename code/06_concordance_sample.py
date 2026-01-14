import pandas as pd
import random
import re

# ===============================
# 1. 설정
# ===============================
DATA_PATH = "../datas/preprocessed_2013_2022.csv"

# log-odds 분석 결과를 토대로 선정된 대표 키워드 쌍
TARGET_PAIRS = [
    ("개혁", "인권", "문재인정부"),
    ("공무원", "연금", "박근혜정부")
]

SAMPLES_PER_PAIR = 5
RANDOM_SEED = 42
MIN_SENT_LEN = 20     # 너무 짧은 문장 제거
MAX_SENT_LEN = 200    # 너무 긴 문장 제거

random.seed(RANDOM_SEED)

# ===============================
# 2. 문장 분리 함수
# ===============================
def split_sentences(text: str):
    """
    뉴스 기사 텍스트를 문장 단위로 분리
    """
    if not isinstance(text, str):
        return []
    return [
        s.strip()
        for s in re.split(r"[.!?。]", text)
        if s.strip()
    ]

# ===============================
# 3. 키워드 강조 함수
# ===============================
def highlight_keywords(sentence: str, w1: str, w2: str):
    """
    문장 내 키워드 강조 (논문 표/부록 활용 목적)
    """
    sentence = re.sub(rf"\b{re.escape(w1)}\b", f"[{w1}]", sentence)
    sentence = re.sub(rf"\b{re.escape(w2)}\b", f"[{w2}]", sentence)
    return sentence

# ===============================
# 4. 콘코드(KWIC) 추출 함수
# ===============================
def extract_concordance(df, w1, w2, gov, n_samples=5):
    """
    특정 정부 시기에서 두 키워드가
    동일 문장 내에 함께 등장하는 문장 추출
    """

    pattern1 = rf"\b{re.escape(w1)}\b"
    pattern2 = rf"\b{re.escape(w2)}\b"

    gov_df = df[df["정부"] == gov]

    matched_sentences = []

    for text in gov_df["text"]:
        for sent in split_sentences(text):
            if (
                MIN_SENT_LEN <= len(sent) <= MAX_SENT_LEN and
                re.search(pattern1, sent) and
                re.search(pattern2, sent)
            ):
                matched_sentences.append(sent)

    total_count = len(matched_sentences)

    if total_count == 0:
        return [], 0

    sampled = random.sample(
        matched_sentences,
        k=min(n_samples, total_count)
    )

    return sampled, total_count

# ===============================
# 5. 실행부
# ===============================
if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

    results = []

    for w1, w2, gov in TARGET_PAIRS:
        samples, total = extract_concordance(
            df=df,
            w1=w1,
            w2=w2,
            gov=gov,
            n_samples=SAMPLES_PER_PAIR
        )

        for sent in samples:
            results.append({
                "정부": gov,
                "키워드1": w1,
                "키워드2": w2,
                "총_문장수": total,
                "문장": highlight_keywords(sent, w1, w2)
            })

    result_df = pd.DataFrame(results)

    save_path = "../result/concordance_samples_final.csv"
    result_df.to_csv(save_path, index=False, encoding="utf-8-sig")

    print(f"콘코드(KWIC) 샘플 저장 완료: {save_path}")
