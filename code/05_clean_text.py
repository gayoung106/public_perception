from konlpy.tag import Okt
import pandas as pd
import re
from stopwords import STOPWORDS

okt = Okt()

def clean_text(text):
    text = re.sub(r"[^가-힣\s]", " ", str(text))
    nouns = okt.nouns(text)
    nouns = [n for n in nouns if len(n) > 1 and n not in STOPWORDS]
    return " ".join(nouns)

for period in ["2015_2019", "2020_2024"]:
    df = pd.read_csv(f"../datas/preprocessed_{period}.csv", encoding="utf-8-sig")
    df["clean_text"] = df["text"].apply(clean_text)
    df.to_csv(f"../datas/clean_{period}.csv", index=False, encoding="utf-8-sig")
    print(f"clean_{period}.csv 생성 완료")
