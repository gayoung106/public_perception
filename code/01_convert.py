import pandas as pd
import glob
import os

# datas 폴더 안의 모든 엑셀 파일 찾기
excel_files = glob.glob("../datas/news_*.xlsx")

print(f"엑셀 파일 개수: {len(excel_files)}")

for file in excel_files:
    base_name = os.path.basename(file)
    csv_file = base_name.replace(".xlsx", ".csv")

    df = pd.read_excel(file, engine="openpyxl")

    df.to_csv(
        f"../datas/{csv_file}",
        index=False,
        encoding="utf-8-sig"
    )

    print(f"변환 완료: {csv_file}")
