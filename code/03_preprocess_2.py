import pandas as pd
import re

# 1. 데이터 불러오기
df = pd.read_csv("../datas/news_2013_2022_merged.csv", encoding="utf-8-sig")
print(f"[STEP 6] 병합 데이터 로드: {len(df)} 건")

date_col = "날짜"

# 2. 날짜 재확인
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
before_date_filter = len(df)
df = df.dropna(subset=[date_col])
after_date_filter = len(df)

print(f"[STEP 7] 날짜 재검증 제거 전: {before_date_filter}")
print(f"[STEP 7] 날짜 재검증 제거 후: {after_date_filter}")

# 정부 시기 구분
def get_administration(x):
    if pd.Timestamp("2013-02-25") <= x <= pd.Timestamp("2017-03-10"):
        return "박근혜정부"
    elif pd.Timestamp("2017-05-10") <= x <= pd.Timestamp("2022-05-09"):
        return "문재인정부"
    else:
        return "기타"

df["정부"] = df[date_col].apply(get_administration)
df["year"] = df[date_col].dt.year

before_gov_filter = len(df)
df = df[df["정부"] != "기타"]
after_gov_filter = len(df)

print(f"[STEP 8] 정부 분류 전 기사 수: {before_gov_filter}")
print(f"[STEP 8] 박/문 정부 기사 수: {after_gov_filter}")
print(f"[STEP 8] 기타 정부 제거 기사 수: {before_gov_filter - after_gov_filter}")

# 3. 텍스트 결합
df["text"] = (
    df["제목"].fillna("") + " " +
    df["본문"].fillna("") + " " +
    df["키워드"].fillna("") + " " +
    df["특성추출(가중치순 상위 50개)"].fillna("")
)

# 특수문자 제거
df["text"] = df["text"].str.replace(r"[^가-힣\s]", " ", regex=True)
df["text"] = df["text"].str.replace(r"\s+", " ", regex=True).str.strip()

before_text_filter = len(df)
df = df[df["text"].str.len() > 0]
after_text_filter = len(df)

print(f"[STEP 9] 텍스트 전처리 전 기사 수: {before_text_filter}")
print(f"[STEP 9] 빈 텍스트 제거 후 기사 수: {after_text_filter}")
print(f"[STEP 9] 텍스트 기준 제거 기사 수: {before_text_filter - after_text_filter}")

# 최종 데이터셋
final_df = df[["날짜", "year", "정부", "언론사", "text"]]

# 저장
final_df.to_csv("../datas/preprocessed_2013_2022.csv", index=False, encoding="utf-8-sig")
print(f"[FINAL] 최종 분석용 데이터셋 크기: {len(final_df)} 건")
