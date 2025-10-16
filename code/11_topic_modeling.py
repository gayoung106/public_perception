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

# ---------- 1. 폰트 설정 ----------
if platform.system() == "Windows":
    font_path = "C:/Windows/Fonts/malgun.ttf"  #  Windows용 한글 폰트
elif platform.system() == "Darwin":
    font_path = "/System/Library/Fonts/AppleSDGothicNeo.ttc"  #  macOS
else:
    font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"  # Linux

plt.rcParams["axes.unicode_minus"] = False

okt = Okt()

# ------------------------------------------------------------
# 🔹 불용어 설정 (이전 단계 동일)
# ------------------------------------------------------------
EXCLUDE_WORDS = [
 # 정치 및 기관명
    '정부', '공직자', '공무원', '정책', '국가', '제도',
    '위원회', '부처', '국회', '장관', '대통령', '청와대', '서울시',
    '의원', '기자', '위원', '위원장', '관계자', '단체', '기관', '본부',
    '발표', '회의', '예산', '조사', '보고서', '언론', '나라', '시장',

    # 정치인 및 정당
    '윤석열', '윤석렬', '김건희', '이재명', '조국', '문재인', '박근혜',
    '이명박', '국민의힘', '민주당', '더불어민주당', '국힘', '혁신당', '민주',
    '진보', '보수', '더불어',

    # 지명 및 지역
    '한국', '대한민국', '서울', '일본', '대만', '지방', '강진군', '전국', '인천',

    # 불필요 연결어 / 기능어
    '관련', '통해', '의견', '논의', '추진', '방안', '지적',
    '확대', '요구', '대상', '중요', '사실', '입장', '시작', '처리','위해', '대한', '이번',

    # 사건·사고 및 뉴스용어
    '사건', '사안', '피해자', '성폭력', '성희롱', '비서', '비서실',
    '투표', '선거', '대선', '탄핵', '촛불', '법안', '검찰', '심판',

    # 역사/의례/국가보훈
    '전쟁', '호국', '현충일', '독립운동', '독립운동가', '태극기', '유공', '보훈', '희생',

    # 일반 사회어
    '학교', '학생', '아이', '유치원', '부모', '친구', '급식',
    '출산율', '가족', '노인', '주택', '지역', '부담',

    # 기사 작성용 표현
    '대표', '기획', '기분', '결정', '사실', '지난달', '시간', '내년', '사정', '선임', '발생', '신고',

    # 기타 불필요 단어
    '효능', '연구원', '연구소', '사회학', '독자', '평가', '대책', '악성', '피크', '지난해', '인상', '지난', '추념', '주목', '최근', '이상', '사람', '여론조사'
]

# ------------------------------------------------------------
# 🔹 명사 추출 + 전처리
# ------------------------------------------------------------
def tokenize(text):
    words = okt.nouns(str(text))
    return [w for w in words if len(w) > 1 and w not in EXCLUDE_WORDS]

# ------------------------------------------------------------
# 🔹 LDA 토픽 모델링 함수
# ------------------------------------------------------------
def lda_topic_modeling(file_path, title, num_topics=5, num_words=10):
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
file_2020_2025 = "../datas/clean_2020_2025.csv"

lda_topic_modeling(file_2015_2019, "2015–2019")
lda_topic_modeling(file_2020_2025, "2020–2025")