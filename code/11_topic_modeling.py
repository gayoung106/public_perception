# ============================================================
#   파일명: 11_topic_modeling.py
#   목적: 시기별 LDA 토픽 모델링 (공직사회 인식 주제 도출)
# ============================================================

import pandas as pd
from konlpy.tag import Okt
from gensim import corpora, models
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import platform
from stopwords import STOPWORDS


if platform.system() == "Windows":
    plt.rc('font', family='Malgun Gothic')
    font_path = "C:/Windows/Fonts/malgun.ttf"
elif platform.system() == "Darwin":  # macOS
    plt.rc('font', family='AppleGothic')
    font_path = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
else:  # Linux
    plt.rc('font', family='NanumGothic')
    font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"

plt.rcParams['axes.unicode_minus'] = False

okt = Okt()

# ------------------------------------------------------------
# 🔹 명사 추출 + 전처리
# ------------------------------------------------------------
def tokenize(text):
    words = okt.nouns(str(text))
    return [w for w in words if len(w) > 1 and w not in STOPWORDS]

# ------------------------------------------------------------
# 🔹 LDA 토픽 모델링 함수
# ------------------------------------------------------------
def lda_topic_modeling(file_path, title, num_topics=3, num_words=10):
    print(f"▶ {title} 데이터 토픽 분석 중...")

    df = pd.read_csv(file_path)
    df = df.dropna(subset=['clean_text'])

    # 토큰화
    tokenized_docs = df['clean_text'].apply(tokenize).tolist()

    # 단어 사전 & 코퍼스 생성
    dictionary = corpora.Dictionary(tokenized_docs)
    corpus = [dictionary.doc2bow(text) for text in tokenized_docs]

    # LDA 모델 학습
    lda_model = models.LdaModel(
        corpus,
        num_topics=num_topics,
        id2word=dictionary,
        passes=10,
        random_state=42
    )

    # 토픽별 단어 출력
    topics = lda_model.print_topics(num_words=num_words)
    print(f"\n📘 [{title}] 토픽별 주요 단어\n")
    for i, topic in enumerate(topics):
        print(f" 토픽 {i+1}: {topic[1]}")

    # ----------  WordCloud 시각화 ----------
    for i in range(num_topics):
        plt.figure(figsize=(6, 6))
        topic_words = dict(lda_model.show_topic(i, topn=30))
        wc = WordCloud(
            font_path=font_path,  #  자동 설정된 폰트 사용
            background_color='white',
            colormap='tab10',
            width=800, height=800
        ).generate_from_frequencies(topic_words)
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title(f"{title} - 토픽 {i+1}", fontsize=14)
        plt.tight_layout()
        plt.savefig(f"../datas/{title}_topic{i+1}.png", dpi=300)
        plt.close()  # 🔹 메모리 절약

    # CSV 저장
    topic_data = []
    for i, topic in enumerate(topics):
        words = [w.split('*')[1].replace('"', '').strip() for w in topic[1].split('+')]
        topic_data.append({'시기': title, '토픽': f'Topic {i+1}', '단어': ', '.join(words)})

    pd.DataFrame(topic_data).to_csv(f"../datas/{title}_topics.csv", index=False, encoding='utf-8-sig')
    print(f" 완료: {title} ({num_topics}개 토픽 저장됨)\n")

# ------------------------------------------------------------
# 🔹 실행 구간
# ------------------------------------------------------------
file_2015_2019 = "../datas/clean_2015_2019.csv"
file_2020_2024 = "../datas/clean_2020_2024.csv"

lda_topic_modeling(file_2015_2019, "2015–2019")
lda_topic_modeling(file_2020_2024, "2020–2024")