# 파일명: 16_dataset_overview_full.py
import pandas as pd
from collections import Counter

# ================================
# 1️ 파일 경로 설정
# ================================
raw_2015 = "../datas/public_perception_2015_2019.csv"
raw_2020 = "../datas/public_perception_2020_2024.csv"
clean_2015 = "../datas/clean_2015_2019.csv"
clean_2020 = "../datas/clean_2020_2024.csv"

# ================================
# 2️ 파일 로드
# ================================
df_raw_15 = pd.read_csv(raw_2015, encoding="utf-8-sig")
df_raw_20 = pd.read_csv(raw_2020, encoding="utf-8-sig")
df_clean_15 = pd.read_csv(clean_2015, encoding="utf-8-sig")
df_clean_20 = pd.read_csv(clean_2020, encoding="utf-8-sig")

# ================================
# 3️ 기사 수 계산
# ================================
raw_15_count = len(df_raw_15)
raw_20_count = len(df_raw_20)
clean_15_count = len(df_clean_15)
clean_20_count = len(df_clean_20)

raw_total = raw_15_count + raw_20_count
clean_total = clean_15_count + clean_20_count

print(" [기사 개수 요약]")
print(f"• 원본 기사 수 (2015–2019): {raw_15_count}건")
print(f"• 원본 기사 수 (2020–2024): {raw_20_count}건")
print(f"→ 총 원본 기사 수: {raw_total}건")

print(f"\n• 정제 후 기사 수 (2015–2019): {clean_15_count}건")
print(f"• 정제 후 기사 수 (2020–2024): {clean_20_count}건")
print(f" 최종 정제 기사 수 합계: {clean_total}건")

# ================================
# 4️ 언론사별 기사 분포
# ================================
provider_col = None
for col in df_raw_15.columns:
    if "언론사" in col or "provider" in col:
        provider_col = col
        break

if provider_col:
    provider_counts = Counter(
        pd.concat([df_raw_15[provider_col], df_raw_20[provider_col]], axis=0)
    )
    top10 = provider_counts.most_common(10)

    print("\n📰 [언론사별 기사 분포 상위 10개]")
    for media, cnt in top10:
        ratio = (cnt / raw_total) * 100
        print(f" - {media:<10}: {cnt}건 ({ratio:.1f}%)")
else:
    print(" '언론사' 또는 'provider' 컬럼이 존재하지 않습니다.")

# ================================
# 5️ 결과 요약 CSV 저장
# ================================
summary_df = pd.DataFrame({
    "구분": [
        "원본 기사 수 (2015–2019)", "원본 기사 수 (2020–2024)", "총 원본 기사",
        "정제 기사 수 (2015–2019)", "정제 기사 수 (2020–2024)", "총 정제 기사"
    ],
    "기사 수": [
        raw_15_count, raw_20_count, raw_total,
        clean_15_count, clean_20_count, clean_total
    ]
})

summary_df.to_csv("../datas/dataset_summary_full.csv", index=False, encoding="utf-8-sig")
print(" dataset_summary_full.csv 저장 완료 ")
