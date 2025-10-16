from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# 데이터 불러오기
df1 = pd.read_csv("../datas/clean_2015_2019.csv")
df2 = pd.read_csv("../datas/clean_2020_2025.csv")

# TF-IDF 계산
vectorizer = TfidfVectorizer(max_features=1000)
tfidf1 = vectorizer.fit_transform(df1["clean_text"])
tfidf2 = vectorizer.fit_transform(df2["clean_text"])

vocab = vectorizer.get_feature_names_out()
df_tfidf1 = pd.DataFrame(tfidf1.toarray(), columns=vocab)
df_tfidf2 = pd.DataFrame(tfidf2.toarray(), columns=vocab)

# 평균 TF-IDF
mean1 = df_tfidf1.mean().sort_values(ascending=False)
mean2 = df_tfidf2.mean().sort_values(ascending=False)

# 단어별 증감 계산
change = pd.concat([mean1, mean2], axis=1, keys=["2015_2019", "2020_2025"]).fillna(0)
change["증감률(%)"] = ((change["2020_2025"] - change["2015_2019"]) / change["2015_2019"].replace(0, 0.0001)) * 100
change_sorted = change.sort_values("증감률(%)", ascending=False)

change_sorted.head(30).to_csv("../datas/keyword_change.csv", encoding="utf-8-sig")
print("keyword_change.csv 생성 완료")
