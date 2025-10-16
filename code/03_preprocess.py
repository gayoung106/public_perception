import pandas as pd
import re

# CSV 불러오기
df1 = pd.read_csv("../datas/public_perception_2015_2019.csv", encoding="utf-8-sig")
df2 = pd.read_csv("../datas/public_perception_2020_2025.csv", encoding="utf-8-sig")

# 제목 + 본문 + 키워드 합치기
def combine_text(df):
    df["text"] = (
        df["제목"].fillna("") + " " +
        df["본문"].fillna("") + " " +
        df["키워드"].fillna("") + " " +
        df["특성추출(가중치순 상위 50개)"].fillna("")
    )
    # 한글, 공백만 남기기
    df["text"] = df["text"].apply(lambda x: re.sub(r"[^가-힣\s]", " ", str(x)))
    # 불필요한 공백 정리
    df["text"] = df["text"].str.replace("\s+", " ", regex=True).str.strip()
    return df[["일자", "언론사", "text"]]

# 전처리 실행
clean1 = combine_text(df1)
clean2 = combine_text(df2)

# 저장
clean1.to_csv("../datas/preprocessed_2015_2019.csv", index=False, encoding="utf-8-sig")
clean2.to_csv("../datas/preprocessed_2020_2025.csv", index=False, encoding="utf-8-sig")

print("전처리 완료: preprocessed_2015_2019.csv / preprocessed_2020_2025.csv 생성")
