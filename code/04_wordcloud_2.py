import pandas as pd
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

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
            if len(w) > 1 and w not in STOPWORDS_SET
        ]
        counter.update(words)

    wc_results[gov] = counter.most_common(100)

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
