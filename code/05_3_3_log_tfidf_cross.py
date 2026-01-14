import pandas as pd

TOP_N = 50

tfidf_moon = pd.read_csv("../result/tfidf_moon_top50.csv")
tfidf_park = pd.read_csv("../result/tfidf_park_top50.csv")
log_moon = pd.read_csv("../result/log_odds_top_moon.csv")
log_park = pd.read_csv("../result/log_odds_top_park.csv")

moon_shared = set(tfidf_moon["단어"]) & set(log_moon["단어"])
park_shared = set(tfidf_park["단어"]) & set(log_park["단어"])

pd.DataFrame({"단어": sorted(moon_shared)})\
  .to_csv("../result/shared_keywords_moon.csv", index=False, encoding="utf-8-sig")

pd.DataFrame({"단어": sorted(park_shared)})\
  .to_csv("../result/shared_keywords_park.csv", index=False, encoding="utf-8-sig")
