import pandas as pd
from gensim import corpora
from gensim.models import LdaMulticore
from multiprocessing import cpu_count
from stopwords import STOPWORDS
import os

# ===============================
# 1. 설정
# ===============================
NUM_TOPICS = 10
PASSES = 5
ITERATIONS = 100
WORKERS = max(cpu_count() - 1, 1)

DATA_PATH = "../datas/preprocessed_2013_2022.csv"
OUTPUT_DIR = "../result/lda"
os.makedirs(OUTPUT_DIR, exist_ok=True)

STOPWORDS_SET = set(STOPWORDS(version="base"))

# ===============================
# 2. 정부별 LDA 함수
# ===============================
def run_lda_by_government(df, gov_name):
    print(f"\nLDA 시작: {gov_name}")

    texts = []

    for doc in df[df["정부"] == gov_name]["text"].dropna():
        words = [
            w for w in str(doc).split()
            if len(w) > 1 and w not in STOPWORDS_SET
        ]
        if words:
            texts.append(words)

    print(f" - 문서 수: {len(texts)}")

    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=10, no_above=0.5)

    corpus = [dictionary.doc2bow(text) for text in texts]


    lda_model = LdaMulticore(
        corpus=corpus,
        id2word=dictionary,
        num_topics=10,          
        random_state=42,
        passes=5,
        workers=6,             
        alpha="symmetric",     
        eta="auto"
    )

    # ---------------------------
    # 토픽 키워드 저장
    # ---------------------------
    topic_words = []

    for topic_id in range(NUM_TOPICS):
        words = lda_model.show_topic(topic_id, topn=15)
        for word, prob in words:
            topic_words.append({
                "정부": gov_name,
                "토픽": topic_id,
                "키워드": word,
                "확률": prob
            })

    topic_df = pd.DataFrame(topic_words)
    topic_df.to_csv(
        os.path.join(OUTPUT_DIR, f"lda_topics_{gov_name}.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    print(f"토픽 결과 저장 완료: {gov_name}")

# ===============================
# 3. 실행부
# ===============================
if __name__ == "__main__":
    print("데이터 로딩 중...")
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

    run_lda_by_government(df, "박근혜정부")
    run_lda_by_government(df, "문재인정부")
