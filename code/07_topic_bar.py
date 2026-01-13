# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.font_manager as fm
import platform

# ===============================
# 한글 폰트 설정
# ===============================
if platform.system() == "Windows":
    font_path = "C:/Windows/Fonts/malgun.ttf"
    font_name = fm.FontProperties(fname=font_path).get_name()
    plt.rc("font", family=font_name)
elif platform.system() == "Darwin":  # macOS
    plt.rc("font", family="AppleGothic")
else:  # Linux
    plt.rc("font", family="NanumGothic")

plt.rcParams["axes.unicode_minus"] = False


# ===============================
# 1. 데이터 로드 (정부별 CSV 병합)
# ===============================
base_dir = "../result/lda"

files = [
    "lda_topics_박근혜정부.csv",
    "lda_topics_문재인정부.csv"
]

dfs = []
for f in files:
    path = os.path.join(base_dir, f)
    dfs.append(pd.read_csv(path, encoding="utf-8-sig"))

df = pd.concat(dfs, ignore_index=True)
print("📌 로딩 완료:", df.shape)

# ===============================
# 2. 토픽별 키워드 바차트
# ===============================
output_dir = "../result/topic_barplots"
os.makedirs(output_dir, exist_ok=True)

def plot_topic_keywords(df, top_n=10):
    for gov in df["정부"].unique():
        gov_df = df[df["정부"] == gov]

        for topic in sorted(gov_df["토픽"].unique(), key=int):
            topic_df = (
                gov_df[gov_df["토픽"] == topic]
                .sort_values("확률", ascending=False)
                .head(top_n)
            )

            plt.figure(figsize=(8, 5))
            plt.barh(
                topic_df["키워드"][::-1],
                topic_df["확률"][::-1]
            )
            plt.title(f"{gov} - Topic {topic}")
            plt.xlabel("P(word | topic)")
            plt.tight_layout()

            save_path = os.path.join(
                output_dir,
                f"{gov}_topic_{topic}.png"
            )
            plt.savefig(save_path, dpi=300)
            plt.close()

            print(f"✅ 저장 완료: {save_path}")

plot_topic_keywords(df)

# ===============================
# 3. 정부별 토픽 분포 비교
# ===============================
topic_dist = (
    df.groupby(["정부", "토픽"])["확률"]
    .sum()
    .reset_index()
)

pivot_df = topic_dist.pivot(
    index="토픽",
    columns="정부",
    values="확률"
).fillna(0)

ax = pivot_df.plot(kind="bar", figsize=(10, 6))
ax.set_title("정부별 토픽 분포 비교")
ax.set_ylabel("토픽 비중")
ax.set_xlabel("토픽")
plt.xticks(rotation=0)
plt.tight_layout()

save_path = "../result/topic_distribution_by_government.png"
plt.savefig(save_path, dpi=300)
plt.close()

print(f"✅ 저장 완료: {save_path}")

