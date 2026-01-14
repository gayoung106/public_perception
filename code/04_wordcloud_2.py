import pandas as pd
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re

# 워드클라우드는 단어의 상대적 빈도를 시각적으로 탐색하기 위한 도구이며,
# 단어 간의 의미적 거리나 구조를 직접적으로 반영하지 않는다.
# 이후 분석에서는 TF-IDF, log-odds, 공출현 분석을 통해 정량적 차별성을 검증하였다.

# 1. 불용어 로드
try:
    from stopwords import STOPWORDS
    STOPWORDS_SET = set(STOPWORDS())
except ImportError:
    print("stopwords.py를 찾을 수 없습니다.")
    STOPWORDS_SET = set()

# 2. 데이터 불러오기 (정제된 전체 코퍼스)
file_path = "../datas/preprocessed_2013_2022.csv"
df = pd.read_csv(file_path, usecols=["정부", "text"], encoding="utf-8-sig")

print(f"전체 분석 대상 데이터: {len(df)}건")

# 3. 정부별 워드클라우드 생성
governments = ["박근혜정부", "문재인정부"]

plt.rcParams['font.family'] = 'Malgun Gothic'
fig, axes = plt.subplots(1, 2, figsize=(22, 10))

# 결과 저장용 (재현성 및 추가 분석 대비)
wc_results = {}

for i, gov in enumerate(governments):
    gov_df = df[df["정부"] == gov]

    if len(gov_df) == 0:
        print(f"주의: {gov}에 해당하는 데이터가 없습니다.")
        continue

    counter = Counter()

    for text in gov_df["text"]:
        words = [
            w for w in str(text).split()
            if len(w) > 1
            and w not in STOPWORDS_SET
            and re.fullmatch(r"[가-힣]{2,}", w)
        ]
        counter.update(words)

    wc_results[gov] = counter.most_common(100)

    pd.DataFrame(
        wc_results[gov],
        columns=["단어", "빈도"]
    ).to_csv(
        f"../result/wordcloud_top100_{gov}.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 워드클라우드 생성
    word_counts = dict(wc_results[gov])
    wc = WordCloud(
        font_path='malgun',
        background_color='white',
        width=800,
        height=800,
        max_words=100
    ).generate_from_frequencies(word_counts)

    axes[i].imshow(wc, interpolation='bilinear')
    axes[i].set_title(f"<{gov}> 상위 빈출 단어 (TOP 100)", fontsize=25)
    axes[i].axis('off')

    print(f"\n[{gov}] 상위 10개 단어:")
    print([w[0] for w in wc_results[gov][:10]])

plt.tight_layout()
plt.savefig("../result/final_wordcloud_comparison.png", dpi=300)
print("\n이미지 저장 완료: ../result/final_wordcloud_comparison.png")
plt.show()

# 전처리 후 산출한 워드클라우드 분석 결과, 
# 두 정권 집합에서 공통적으로 ‘평가’, ‘정책’, ‘운영’, ‘추진’, ‘인사’, ‘참여’, ‘여성’과 같은 행정·정책 일반어가 최상위 빈출어로 반복적으로 등장하였다. 
# 이들 단어는 공공정책 및 행정 담론 전반에서 상시적으로 사용되는 범용 행정어(generic administrative terms)로, 특정 정권이나 시기의 담론적 차별성을 직접적으로 반영하기보다는 비교분석 시 공통 배경어로 작동할 가능성이 높다.

# 이에 본 연구에서는 해당 단어들을 즉시 불용어로 제거하지 않고, 
# 워드클라우드 분석을 통해 공통 핵심어로 확인한 이후 ‘조건부 불용어(conditional stopwords)’로 분류하였다. 
# 이는 초기 탐색 단계에서는 전체 담론 구조를 파악하기 위해 해당 단어들을 유지하되, 
# 이후 TF-IDF, log-odds, 공출현 분석과 같은 차별화 중심 분석 단계에서 필요에 따라 제외하거나 포함하여 결과의 안정성과 해석 가능성을 검증하기 위함이다.

# 일부 ‘인사’와 ‘여성’,' ‘정책’ 등은 정책 담론에서 핵심 개념어에 해당하나, 
# 두 정권 집합 모두에서 최상위 빈출어로 반복적으로 등장하여 차별적 중요도를 판별하는 초기 분석 단계(TF-IDF, log-odds)에서는 노이즈로 작용할 가능성이 있다. 
# 이에 본 연구에서는 해당 단어들을 완전한 불용어로 제거하지 않고 조건부 불용어로 분류하여, 차별화 분석 단계에서는 통제하되 공출현 분석과 토픽 모델링 단계에서는 다시 포함하여 의미 구조를 해석하였다.

# 즉, 조건부 불용어는 분석 과정 전반에서 자동적으로 제거되는 불용어가 아니라, 
# 워드클라우드 결과를 근거로 설정된 비교 분석용 조정 변수로서, 공통 행정어로 인한 노이즈를 통제하면서도 분석자의 자의적 단어 제거를 최소화하기 위한 절차적 장치이다.