import pandas as pd
import re

# 1. 데이터 불러오기
df = pd.read_csv("../datas/news_2013_2022_merged.csv", encoding="utf-8-sig")
print(f"초기 데이터 개수: {len(df)}")

# 컬럼명 확인 (merge에서 바꾼 '날짜' 사용)
date_col = '날짜' if '날짜' in df.columns else '일자'

# 2. 날짜 변환 및 필터링
print("날짜 변환 및 정부 시기 분류 중...")
df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
df = df.dropna(subset=[date_col])

# 정부 시기 구분 함수 (날짜가 이미 datetime이므로 바로 비교 가능)
def get_administration(x):
    if pd.Timestamp("2013-02-25") <= x <= pd.Timestamp("2017-03-10"):
        return "박근혜정부"
    elif pd.Timestamp("2017-05-10") <= x <= pd.Timestamp("2022-05-09"):
        return "문재인정부"
    else:
        return "기타"

df["정부"] = df[date_col].apply(get_administration)
df["year"] = df[date_col].dt.year

# 박/문 정부 데이터만 추출
df = df[df["정부"] != "기타"]
print(f"정부 분류 완료 (박/문 남은 데이터): {len(df)}")

# 3. 텍스트 결합 및 전처리
print("텍스트 결합 및 특수문자 제거 중... (대용량이라 시간이 소요됩니다)")
df["text"] = (
    df["제목"].fillna("") + " " +
    df["본문"].fillna("") + " " +
    df["키워드"].fillna("") + " " +
    df["특성추출(가중치순 상위 50개)"].fillna("")
)

# 불필요한 공백 및 한글 제외 문자 제거 (벡터화 연산으로 속도 향상)
df["text"] = df["text"].str.replace(r"[^가-힣\s]", " ", regex=True)
df["text"] = df["text"].str.replace(r"\s+", " ", regex=True).str.strip()

# 4. 최종 데이터 정리
final_df = df[[date_col, "year", "정부", "언론사", "text"]].rename(columns={date_col: "날짜"})
final_df = final_df[final_df["text"].str.len() > 0] # 빈 텍스트 제거

# --- 결과 미리보기 ---
print("\n" + "="*50)
print("🔎 전처리 결과 미리보기")
print("-" * 50)
preview = final_df.head(10).copy()
preview['text_preview'] = preview['text'].str[:50] + "..."
print(preview[["날짜", "정부", "언론사", "text_preview"]])
print("="*50 + "\n")

# 5. 저장
final_df.to_csv("../datas/preprocessed_2013_2022.csv", index=False, encoding="utf-8-sig")
print(f"최종 전처리 완료: {len(final_df)} 건 저장됨.")