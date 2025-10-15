import pandas as pd
import glob
import os

# datas 폴더 안의 모든 엑셀 파일 찾기
excel_files = glob.glob("../datas/*.xlsx")

for file in excel_files:
    # 파일명만 추출해서 CSV 이름 지정
    base_name = os.path.basename(file)
    csv_file = base_name.replace(".xlsx", ".csv")

    # 엑셀 파일 읽기
    df = pd.read_excel(file, engine="openpyxl")

    # CSV로 저장 (현재 code 폴더 바로 위 PUBLIC_PERCEPTION 폴더에 저장됨)
    df.to_csv(f"../datas/{csv_file}", index=False, encoding="utf-8-sig")

    print(f"변환 완료: {csv_file}")
