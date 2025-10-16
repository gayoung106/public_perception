import pandas as pd
from collections import Counter

# 🔹 파일 불러오기
df1 = pd.read_csv("../datas/clean_2015_2019.csv", encoding="utf-8-sig")
df2 = pd.read_csv("../datas/clean_2020_2025.csv", encoding="utf-8-sig")

# 🔹 첫 번째 문서 단어 분포 확인
sample_text1 = df1.loc[0, "clean_text"]
sample_text2 = df2.loc[0, "clean_text"]

print(" 원문 샘플 비교:")
print(" [2015–2019] sample1:")
print(sample_text1[:400], "...\n")
print(" [2020–2025] sample2:")
print(sample_text2[:400], "...\n")

# 🔹 중복 단어 수 세기
words1 = sample_text1.split()
words2 = sample_text2.split()

word_counts1 = Counter(words1)
word_counts2 = Counter(words2)

duplicates1 = {w: c for w, c in word_counts1.items() if c > 2}
duplicates2 = {w: c for w, c in word_counts2.items() if c > 2}

print(" [2015–2019] 반복 횟수가 많은 단어들:")
for w, c in sorted(duplicates1.items(), key=lambda x: x[1], reverse=True):
    print(f"{w}: {c}")

print(" [2020–2025] 반복 횟수가 많은 단어들:")
for w, c in sorted(duplicates2.items(), key=lambda x: x[1], reverse=True):
    print(f"{w}: {c}")
