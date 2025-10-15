import pandas as pd

# 파일 경로
df1 = pd.read_csv("../datas/public_perception_2015_2019.csv", encoding="utf-8-sig")
df2 = pd.read_csv("../datas/public_perception_2020_2025.csv", encoding="utf-8-sig")

# 데이터 크기 확인
print("2015–2019 데이터:", df1.shape)
print("2020–2025 데이터:", df2.shape)

# 컬럼명 확인
print("\n[2015–2019 컬럼]")
print(df1.columns.tolist())

print("\n[2020–2025 컬럼]")
print(df2.columns.tolist())

# 예시 5행 확인
print("\n샘플 미리보기 (2015–2019)")
print(df1.head())
