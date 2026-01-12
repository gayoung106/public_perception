# ============================================================
#  파일명: 11_12_topic_compare_visual_csv.py
#  목적: CSV 파일에서 LDA 결과 불러와 시각화 및 비교 분석
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from wordcloud import WordCloud
import seaborn as sns
import os
import platform

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

# ------------------------------------------------------
# 🔹 1. CSV 파일 불러오기
# ------------------------------------------------------
# 예시 파일 구조:
#   topics_2015_2019.csv → columns=["토픽", "단어"]
#   topics_2020_2025.csv → columns=["토픽", "단어"]

# 파일 경로 설정
file_2015 = "../datas/2015–2019_topics.csv"
file_2020 = "../datas/2020–2024_topics.csv"


if not os.path.exists(file_2015) or not os.path.exists(file_2020):
    raise FileNotFoundError("CSV 파일이 존재하지 않습니다. 파일명을 확인하세요.")

# CSV 불러오기
df_2015 = pd.read_csv(file_2015)
df_2020 = pd.read_csv(file_2020)

print(" 파일 불러오기 완료")
print(f"2015–2019 shape: {df_2015.shape}, 2020–2024 shape: {df_2020.shape}")

# ------------------------------------------------------
# 🔹 2. 토픽별 단어 리스트 구성
# ------------------------------------------------------
topics_2015_2019 = (
    df_2015.groupby("토픽")["단어"]
    .apply(lambda x: x.dropna().tolist())
    .to_dict()
)

topics_2020_2024 = (
    df_2020.groupby("토픽")["단어"]
    .apply(lambda x: x.dropna().tolist())
    .to_dict()
)

# ------------------------------------------------------
# 🔹 3. 워드클라우드 생성
# ------------------------------------------------------
def make_wordcloud(words, title):
    text = " ".join(words)
    wc = WordCloud(
        font_path=font_path,
        width=600, height=400,
        background_color="white", colormap="viridis"
    ).generate(text)

    plt.figure(figsize=(6, 4))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title, fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.show()

print(" [Step 1] 시기별 워드클라우드 생성 중...")

for i, words in topics_2015_2019.items():
    make_wordcloud(words, f"2015–2019 토픽 {i}")

for i, words in topics_2020_2024.items():
    make_wordcloud(words, f"2020–2024 토픽 {i}")

# ------------------------------------------------------
# 🔹 4. 키워드 빈도 비교
# ------------------------------------------------------
print(" [Step 2] 시기별 토픽 키워드 변화율 계산 중...")

flat_2015 = [w for topic in topics_2015_2019.values() for w in topic]
flat_2020 = [w for topic in topics_2020_2024.values() for w in topic]

freq_2015 = pd.Series(flat_2015).value_counts().rename_axis("단어").reset_index(name="빈도_15_19")
freq_2020 = pd.Series(flat_2020).value_counts().rename_axis("단어").reset_index(name="빈도_20_24")

trend_df = pd.merge(freq_2015, freq_2020, on="단어", how="outer").fillna(0)
trend_df["변화율(%)"] = ((trend_df["빈도_20_24"] - trend_df["빈도_15_19"]) / (trend_df["빈도_15_19"] + 1)) * 100

# 상위 변화 키워드 추출
top_change = trend_df.sort_values("변화율(%)", ascending=False).head(10)

# ------------------------------------------------------
# 🔹 5. 키워드 변화율 시각화
# ------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.barplot(y="단어", x="변화율(%)", data=top_change, palette="crest")
plt.title(" 공직 담론 주요 키워드 변화 (2015–2019 → 2020–2024)", fontsize=13, fontweight="bold")
plt.xlabel("변화율(%)")
plt.ylabel("단어")
plt.tight_layout()
plt.show()

# ------------------------------------------------------
# 🔹 6. 논문용 요약표 생성
# ------------------------------------------------------
print(" [Step 3] 토픽별 대표 단어표 생성 중...")

table_data = []
for topic_id in sorted(set(df_2015["토픽"]).union(df_2020["토픽"])):
    words_2015 = ", ".join(topics_2015_2019.get(topic_id, []))
    words_2020 = ", ".join(topics_2020_2024.get(topic_id, []))
    table_data.append([f"토픽 {topic_id}", words_2015, words_2020])

topic_table = pd.DataFrame(table_data, columns=["토픽", "2015–2019 주요 단어", "2020–2024 주요 단어"])
topic_table.to_csv("../datas/topic_summary_table.csv", index=False, encoding="utf-8-sig")

print("완료: topic_summary_table.csv 저장됨")
print(topic_table)
