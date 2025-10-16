import pandas as pd
from konlpy.tag import Okt
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---------- 1. 데이터 불러오기 ----------
df1 = pd.read_csv("../datas/preprocessed_2015_2019.csv", encoding="utf-8-sig")
df2 = pd.read_csv("../datas/preprocessed_2020_2025.csv", encoding="utf-8-sig")

# ---------- 2. 형태소 분석 ----------
okt = Okt()

def get_nouns(texts):
    nouns = []
    for t in tqdm(texts, desc="형태소 분석 중"):
        n = okt.nouns(str(t))
        # 불필요한 한 글자 제외 (의미 없는 조사/단어 제거)
        n = [word for word in n if len(word) > 1]
        nouns.extend(n)
    return nouns

nouns_15_19 = get_nouns(df1["text"])
nouns_20_25 = get_nouns(df2["text"])

# ---------- 3. 단어 빈도 계산 ----------
count_15_19 = Counter(nouns_15_19)
count_20_25 = Counter(nouns_20_25)

# ---------- 4. 상위 단어 30개 출력 ----------
print("\n [2015–2019 상위 단어 30]")
print(count_15_19.most_common(30))

print("\n[2020–2025 상위 단어 30]")
print(count_20_25.most_common(30))

# ---------- 5. 워드클라우드 생성 ----------
font_path = "/System/Library/Fonts/AppleSDGothicNeo.ttc"  # macOS 기본 한글 폰트

wc1 = WordCloud(
    font_path=font_path,
    background_color="white",
    width=800,
    height=600
).generate_from_frequencies(count_15_19)

wc2 = WordCloud(
    font_path=font_path,
    background_color="white",
    width=800,
    height=600
).generate_from_frequencies(count_20_25)

# ---------- 6. 시각화 ----------
plt.figure(figsize=(15, 7))

plt.subplot(1, 2, 1)
plt.imshow(wc1, interpolation="bilinear")
plt.axis("off")
plt.title("2015–2019 공직사회 관련 주요 단어", fontsize=14)

plt.subplot(1, 2, 2)
plt.imshow(wc2, interpolation="bilinear")
plt.axis("off")
plt.title("2020–2025 공직사회 관련 주요 단어", fontsize=14)

plt.tight_layout()
plt.show()

# ---------- 7. CSV로 빈도 저장 ----------
pd.DataFrame(count_15_19.most_common(100), columns=["단어", "빈도"]).to_csv("../datas/freq_2015_2019.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(count_20_25.most_common(100), columns=["단어", "빈도"]).to_csv("../datas/freq_2020_2025.csv", index=False, encoding="utf-8-sig")

print("\n 분석 완료: 워드클라우드 표시 + freq_*.csv 저장됨")
