import pandas as pd
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 1. 불용어 로드
try:
    from stopwords import STOPWORDS
    STOPWORDS_SET = set(STOPWORDS)
except ImportError:
    print("stopwords.py를 찾을 수 없습니다.")
    STOPWORDS_SET = set()

# 2. 데이터 불러오기
file_path = "../datas/preprocessed_2013_2022.csv"
df = pd.read_csv(file_path, usecols=["정부", "text"], encoding="utf-8-sig")

# 3. 핵심 키워드 필터링 (NPM vs Governance)
core_filter = [
    '공직', '공무원', '행정', '관료', '인사혁신', 
    '연금', '워라밸', 'MZ', '세대', '조직문화', 
    '성과주의', '근무환경', '성과급', '지방자치', '불공정','공정','불만','만족','갈등','위계','수직','수평','자율','책임'
]
df = df[df['text'].str.contains('|'.join(core_filter), na=False)].reset_index(drop=True)
print(f"필터링 완료! 분석 대상 데이터: {len(df)}건")

# 4. 정부별 빈도 계산 및 워드클라우드 생성
governments = ["박근혜정부", "문재인정부"]
plt.rcParams['font.family'] = 'Malgun Gothic'
fig, axes = plt.subplots(1, 2, figsize=(22, 10))

# 결과를 담을 딕셔너리 (나중에 TF-IDF 등에서 활용 가능)
wc_results = {}

for i, gov in enumerate(governments):
    # 해당 정부 데이터만 추출
    gov_df = df[df["정부"] == gov]
    
    if len(gov_df) == 0:
        print(f"주의: {gov}에 해당하는 데이터가 없습니다. 컬럼의 값을 확인하세요.")
        continue

    # 단어 빈도 계산
    counter = Counter()
    for text in gov_df["text"]:

        words = [w for w in str(text).split() if len(w) > 1 and w not in STOPWORDS_SET]
        counter.update(words)
    
    wc_results[gov] = counter.most_common(100)
    
    # 워드클라우드 생성
    word_counts = dict(counter.most_common(100))
    wc = WordCloud(
        font_path='malgun', 
        background_color='white',
        width=800, height=800,
        max_words=100
    ).generate_from_frequencies(word_counts)
    
    axes[i].imshow(wc, interpolation='bilinear')
    axes[i].set_title(f"<{gov}> TOP 100", fontsize=25)
    axes[i].axis('off')
    
    print(f"\n[{gov}] 분석 완료 (상위 10개): {[w[0] for w in wc_results[gov][:10]]}")

plt.tight_layout()
plt.savefig("../datas/final_wordcloud_comparison.png")
print("\n이미지 저장 완료: ../datas/final_wordcloud_comparison.png")
plt.show()