import pandas as pd

# ===============================
# 1. 데이터 로드
# ===============================
tfidf_diff = pd.read_csv("../result/keyword_change_verified_conditional.csv")
log_odds = pd.read_csv("../result/log_odds_keywords_with_conditional.csv")

# ===============================
# 2. 정부별 분리
# ===============================
moon_log = log_odds[log_odds["강화_정부"] == "문재인정부"]
park_log = log_odds[log_odds["강화_정부"] == "박근혜정부"]

# ✅ 여기 핵심 수정
tfidf_words = set(tfidf_diff["Unnamed: 0"])

# ===============================
# 3. 교차
# ===============================
moon_final = tfidf_words & set(moon_log["단어"])
park_final = tfidf_words & set(park_log["단어"])

# ===============================
# 4. 저장
# ===============================
pd.DataFrame({"단어": sorted(moon_final)}).to_csv(
    "../result/final_keywords_moon.csv",
    index=False,
    encoding="utf-8-sig"
)

pd.DataFrame({"단어": sorted(park_final)}).to_csv(
    "../result/final_keywords_park.csv",
    index=False,
    encoding="utf-8-sig"
)

print("최종 차별 담론 키워드 생성 완료")
