import pandas as pd
from konlpy.tag import Okt
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from tqdm import tqdm
import platform
from stopwords import STOPWORDS

# =====================================================
#  1️⃣ 폰트 설정
# =====================================================
if platform.system() == "Windows":
    font_path = "C:/Windows/Fonts/malgun.ttf"  # Windows
elif platform.system() == "Darwin":
    font_path = "/System/Library/Fonts/AppleSDGothicNeo.ttc"  # macOS
else:
    font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"  # Linux

# =====================================================
#  2️⃣ 데이터 불러오기
# =====================================================
df1 = pd.read_csv("../datas/preprocessed_2015_2019.csv", encoding="utf-8-sig")
df2 = pd.read_csv("../datas/preprocessed_2020_2024.csv", encoding="utf-8-sig")

# =====================================================
#  4️⃣ 형태소 분석 함수
# =====================================================
okt = Okt()

def get_nouns(texts):
    nouns = []
    for t in tqdm(texts, desc="형태소 분석 중"):
        n = okt.nouns(str(t))
        n = [word for word in n if len(word) > 1 and word not in STOPWORDS]
        nouns.extend(n)
    return nouns

# =====================================================
#  5️⃣ 명사 추출
# =====================================================
nouns_15_19 = get_nouns(df1["text"])
nouns_20_24 = get_nouns(df2["text"])

# =====================================================
#  6️⃣ 단어 빈도 계산
# =====================================================
count_15_19 = Counter(nouns_15_19)
count_20_24 = Counter(nouns_20_24)

# =====================================================
#  7️⃣ 상위 단어 출력
# =====================================================
print("\n [2015–2019 상위 단어 30]")
print(count_15_19.most_common(30))

print("\n[2020–2024 상위 단어 30]")
print(count_20_24.most_common(30))

# =====================================================
#  8️⃣ 워드클라우드 생성
# =====================================================
wc1 = WordCloud(
    font_path=font_path,
    background_color="white",
    width=800,
    height=600,
    colormap="viridis"
).generate_from_frequencies(count_15_19)

wc2 = WordCloud(
    font_path=font_path,
    background_color="white",
    width=800,
    height=600,
    colormap="viridis"
).generate_from_frequencies(count_20_24)

# =====================================================
#  9️⃣ 시각화 (비교)
# =====================================================
plt.figure(figsize=(15, 7))
plt.subplot(1, 2, 1)
plt.imshow(wc1, interpolation="bilinear")
plt.axis("off")
plt.title("2015–2019 공직사회 주요 키워드 (불용어 제거)", fontsize=14)

plt.subplot(1, 2, 2)
plt.imshow(wc2, interpolation="bilinear")
plt.axis("off")
plt.title("2020–2024 공직사회 주요 키워드 (불용어 제거)", fontsize=14)

plt.tight_layout()
plt.show()

# =====================================================
#  🔟 결과 저장
# =====================================================
pd.DataFrame(count_15_19.most_common(100), columns=["단어", "빈도"]).to_csv("../datas/freq_2015_2019_clean.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(count_20_24.most_common(100), columns=["단어", "빈도"]).to_csv("../datas/freq_2020_2024_clean.csv", index=False, encoding="utf-8-sig")

plt.figure(figsize=(15, 7))
plt.subplot(1, 2, 1)
plt.imshow(wc1, interpolation="bilinear")
plt.axis("off")
plt.title("2015–2019", fontsize=14)

plt.subplot(1, 2, 2)
plt.imshow(wc2, interpolation="bilinear")
plt.axis("off")
plt.title("2020–2024", fontsize=14)

plt.tight_layout()
plt.savefig("../datas/wordcloud_compare_clean.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
plt.close()

print("\n 완료: 불용어 처리된 워드클라우드 및 CSV 저장됨.")
