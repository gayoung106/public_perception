import pandas as pd
import glob
import os

# CSV 파일 불러오기
csv_files = glob.glob("../datas/news_*.csv")
print(f"[STEP 0] 원본 CSV 파일 개수: {len(csv_files)}")

df_list = []
raw_row_count = 0

for file in csv_files:
    tmp = pd.read_csv(file, encoding="utf-8-sig")
    tmp["source_file"] = os.path.basename(file)
    raw_row_count += len(tmp)
    df_list.append(tmp)

print(f"[STEP 1] 병합 전 전체 기사 수(단순 합): {raw_row_count}")

# 병합
df_all = pd.concat(df_list, ignore_index=True)
print(f"[STEP 2] 병합 후 데이터 크기: {df_all.shape}")

# 컬럼 확인
print("=== 병합된 컬럼 목록 ===")
print(df_all.columns.tolist())

# 🔹 컬럼명 통일
if "일자" not in df_all.columns:
    raise ValueError("일자 컬럼이 없습니다.")
df_all = df_all.rename(columns={"일자": "날짜"})

# 🔹 필수 컬럼 체크
for col in ["언론사", "제목", "날짜"]:
    if col not in df_all.columns:
        raise ValueError(f"필수 컬럼 누락: {col}")

# 🔹 중복 제거
before_dedup = len(df_all)
df_all = df_all.drop_duplicates(subset=["언론사", "제목", "날짜"])
after_dedup = len(df_all)

print(f"[STEP 3] 중복 제거 전: {before_dedup}")
print(f"[STEP 3] 중복 제거 후: {after_dedup}")
print(f"[STEP 3] 중복 제거로 삭제된 기사 수: {before_dedup - after_dedup}")

# 🔹 날짜 변환
df_all["날짜"] = pd.to_datetime(df_all["날짜"], format="%Y%m%d", errors="coerce")
before_date_clean = len(df_all)
df_all = df_all.dropna(subset=["날짜"])
after_date_clean = len(df_all)

print(f"[STEP 4] 날짜 변환 실패 제거 전: {before_date_clean}")
print(f"[STEP 4] 날짜 변환 실패 제거 후: {after_date_clean}")
print(f"[STEP 4] 날짜 오류로 제거된 기사 수: {before_date_clean - after_date_clean}")

# 연도 컬럼
df_all["year"] = df_all["날짜"].dt.year

# 저장
df_all.to_csv(
    "../datas/news_2013_2022_merged.csv",
    index=False,
    encoding="utf-8-sig"
)

print(f"[STEP 5] 병합·정제 완료 데이터 저장: {len(df_all)} 건")
