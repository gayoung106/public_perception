# ============================================================
#  파일명: 12_topic_summary_analysis.py
#  목적: topic_summary_table.csv를 불러와 자동 요약 및 변화 분석
# ============================================================
import matplotlib.pyplot as plt
import pandas as pd
import platform

# ================================
#  폰트 설정 (macOS 한글 깨짐 방지)
# ================================
if platform.system() == 'Darwin':  # macOS
    plt.rc('font', family='AppleGothic')
else:  # Windows / Linux 대비
    plt.rc('font', family='Malgun Gothic')

plt.rcParams['axes.unicode_minus'] = False  # 마이너스 깨짐 방지

# ---------------------------------------------
# 🔹 1. CSV 불러오기
# ---------------------------------------------
file_path = "../datas/topic_summary_table.csv"

df = pd.read_csv(file_path)
print("불러온 데이터:")
print(df.head())

# ---------------------------------------------
# 🔹 2. 시기별 공통 단어 / 신규 단어 / 소멸 단어 비교
# ---------------------------------------------
analysis = []
for _, row in df.iterrows():
    topic = row["토픽"]
    words_2015 = set(str(row["2015–2019 주요 단어"]).split(", "))
    words_2020 = set(str(row["2020–2025 주요 단어"]).split(", "))

    common = words_2015 & words_2020
    new = words_2020 - words_2015
    disappeared = words_2015 - words_2020

    analysis.append({
        "토픽": topic,
        "공통 단어": ", ".join(common) if common else "-",
        "신규 등장 단어": ", ".join(new) if new else "-",
        "사라진 단어": ", ".join(disappeared) if disappeared else "-"
    })

trend_df = pd.DataFrame(analysis)
trend_df.to_csv("../datas/topic_keyword_change_summary.csv", index=False, encoding="utf-8-sig")
print(" 키워드 변화 요약 저장 완료: topic_keyword_change_summary.csv")

# ---------------------------------------------
# 🔹 3. 자동 요약문 생성
# ---------------------------------------------
summaries = []
for _, row in trend_df.iterrows():
    topic = row["토픽"]
    common = row["공통 단어"]
    new = row["신규 등장 단어"]
    disappeared = row["사라진 단어"]

    summary = f"""
[{topic}]
2015–2019년에는 '{disappeared}' 등의 단어가 주로 나타났으나,
2020–2025년에는 '{new}' 등의 새로운 키워드가 등장하였다.
이는 {common if common != '-' else '기존 주제와의 연속성보다는 새로운 담론의 확산'}을 시사한다.
"""
    summaries.append(summary)

print(" [시기별 토픽 변화 요약]")
print("\n".join(summaries))

# ---------------------------------------------
# 🔹 4. 전체적 변화 방향 요약
# ---------------------------------------------
all_new = set()
all_old = set()

for _, row in trend_df.iterrows():
    all_new.update(row["신규 등장 단어"].split(", "))
    all_old.update(row["사라진 단어"].split(", "))

overall_new = list(all_new - all_old)
overall_old = list(all_old - all_new)

print(" [전반적 변화 경향]")
print(f"- 신규 등장 핵심어: {', '.join(overall_new[:20])}")
print(f"- 감소/소멸된 핵심어: {', '.join(overall_old[:20])}")
