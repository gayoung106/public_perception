import pandas as pd
from konlpy.tag import Okt
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from tqdm import tqdm
import platform

# ---------- 1. 폰트 설정 ----------
if platform.system() == "Windows":
    font_path = "C:/Windows/Fonts/malgun.ttf"  #  Windows용 한글 폰트
elif platform.system() == "Darwin":
    font_path = "/System/Library/Fonts/AppleSDGothicNeo.ttc"  # macOS
else:
    font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"  # Linux

# ---------- 2. 데이터 불러오기 ----------
df1 = pd.read_csv("../datas/preprocessed_2015_2019.csv", encoding="utf-8-sig")
df2 = pd.read_csv("../datas/preprocessed_2020_2025.csv", encoding="utf-8-sig")

# ---------- 3. 형태소 분석 ----------
okt = Okt()

def get_nouns(texts):
    nouns = []
    for t in tqdm(texts, desc="형태소 분석 중"):
        n = okt.nouns(str(t))
        n = [word for word in n if len(word) > 1]
        nouns.extend(n)
    return nouns

nouns_15_19 = get_nouns(df1["text"])
nouns_20_25 = get_nouns(df2["text"])

# ---------- 4. 단어 빈도 계산 ----------
count_15_19 = Counter(nouns_15_19)
count_20_25 = Counter(nouns_20_25)

# ---------- 5. 상위 단어 출력 ----------
print("\n [2015–2019 상위 단어 30]")
print(count_15_19.most_common(30))

print("\n[2020–2025 상위 단어 30]")
print(count_20_25.most_common(30))

# ---------- 6. 워드클라우드 생성 ----------
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

# ---------- 7. 시각화 ----------
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

# ---------- 8. CSV 저장 ----------
pd.DataFrame(count_15_19.most_common(100), columns=["단어", "빈도"]).to_csv("../datas/freq_2015_2019.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(count_20_25.most_common(100), columns=["단어", "빈도"]).to_csv("../datas/freq_2020_2025.csv", index=False, encoding="utf-8-sig")

print("분석 완료: 워드클라우드 표시 + freq_*.csv 저장됨")

# ---------- 9. 고해상도 이미지 저장 ----------
output_path = "../datas/wordcloud_compare.png"

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
plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
plt.close()

print(f"\n 워드클라우드 이미지 저장 완료 → {output_path}")

