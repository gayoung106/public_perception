# ============================================================
#  파일명: 14_topic_narrative_summary.py
#  목적: 토픽 변화 데이터를 기반으로 자동 서술문 생성
# ============================================================

import pandas as pd

# 🔹 1. 데이터 불러오기
file_path = "../datas/topic_keyword_change_summary.csv"
df = pd.read_csv(file_path)

print("✅ 불러온 데이터:")
print(df.head(), "\n")

# 🔹 2. 자동 문장 생성
paragraphs = []

for _, row in df.iterrows():
    topic = row["토픽"]
    common = row["공통 단어"]
    new_words = row["신규 등장 단어"]
    old_words = row["사라진 단어"]

    # 자연스러운 문장 구조로 변환
    summary = f"""
### {topic} 분석 요약

2015–2019년에는 '{old_words}'와 같은 키워드가 중심을 이루며,
당시 공직 담론이 복지, 세대, 고용 구조 등 사회적 요인에 초점을 맞추고 있었다.

반면 2020–2025년에는 '{new_words}' 등의 단어가 새롭게 등장하여
조직 내부의 구조적 문제, 노동 조건, 세대 간 형평성, 직무 만족도 등
보다 **실질적·조직 중심적 담론**으로 이동한 양상을 보였다.

이는 '{common}'이라는 공통 주제 위에서, 
공직사회의 논의가 단순한 제도·정책적 수준을 넘어 
직무 환경과 세대 간 관계에 대한 논의로 심화되고 있음을 시사한다.
"""
    paragraphs.append(summary.strip())

# 🔹 3. 결과 파일 저장
output_path = "../datas/topic_narrative_summary.txt"
with open(output_path, "w", encoding="utf-8") as f:
    f.write("\n\n".join(paragraphs))

print("✅ 논문용 요약문 저장 완료:", output_path)

# 🔹 4. 콘솔 미리보기
print("\n📘 [논문용 자동 요약 미리보기]\n")
for p in paragraphs:
    print(p, "\n")
