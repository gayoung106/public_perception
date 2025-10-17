import pandas as pd
from konlpy.tag import Okt
from collections import Counter
from tqdm import tqdm
import re
import os

# -----------------------------------------
# 1️⃣ 데이터 불러오기
# -----------------------------------------
files = [
    ("../datas/preprocessed_2015_2019.csv", "2015_2019"),
    ("../datas/preprocessed_2020_2025.csv", "2020_2025"),
]

okt = Okt()

def extract_year(date_str):
    """일자(YYYYMMDD)에서 연도만 추출"""
    match = re.search(r"\d{4}", str(date_str))
    return int(match.group()) if match else None

def get_nouns(text):
    """명사만 추출 (불필요한 한 글자 제외)"""
    nouns = okt.nouns(str(text))
    return [n for n in nouns if len(n) > 1]

# -----------------------------------------
# 2️⃣ 파일별 처리
# -----------------------------------------
for file_path, label in files:
    print(f"\n📂 {label} 데이터 처리 중...")

    df = pd.read_csv(file_path, encoding="utf-8-sig")
    if "일자" not in df.columns:
        raise ValueError(f"❌ {file_path} 파일에 '일자' 컬럼이 없습니다.")

    # 연도 추출
    df["year"] = df["일자"].apply(extract_year)
    df = df.dropna(subset=["year"])

    # 형태소 분석
    all_nouns = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{label} 형태소 분석"):
        nouns = get_nouns(row["text"])
        all_nouns.append(nouns)
    df["nouns"] = all_nouns

    # 연도별 단어 빈도 계산
    year_freq = []
    for year, group in df.groupby("year"):
        counter = Counter([word for words in group["nouns"] for word in words])
        year_df = pd.DataFrame(counter.items(), columns=["word", "freq"])
        year_df["year"] = year
        year_freq.append(year_df)

    result = pd.concat(year_freq, ignore_index=True)
    output_path = f"../datas/freq_{label}.csv"
    result.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"✅ 저장 완료: {output_path} ({len(result)}개 단어)")
