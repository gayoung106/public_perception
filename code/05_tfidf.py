import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
from stopwords import STOPWORDS

# 1. 환경 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
STOPWORDS_LIST = STOPWORDS()  # 함수 호출 괄호 확인

# 2. 데이터 로드
df = pd.read_csv("../datas/preprocessed_2013_2022.csv", encoding="utf-8-sig")

# 3. 정부별 데이터 분리
park_df = df[df["정부"] == "박근혜정부"]
moon_df = df[df["정부"] == "문재인정부"]

# 4. TF-IDF 분석 (정석적 방법)
vectorizer = TfidfVectorizer(
    max_features=3000,
    min_df=10,
    max_df=0.7,
    stop_words=STOPWORDS_LIST,
    token_pattern=r"[가-힣]{2,}"
)

# 박근혜 정부 기준으로 학습 후 계산
tfidf_park = vectorizer.fit_transform(park_df["text"])
# 문재인 정부는 동일 기준(vocab)으로 계산
tfidf_moon = vectorizer.transform(moon_df["text"])

mean_park = pd.Series(tfidf_park.mean(axis=0).A1, index=vectorizer.get_feature_names_out())
mean_moon = pd.Series(tfidf_moon.mean(axis=0).A1, index=vectorizer.get_feature_names_out())

# 5. 결과 통합 및 필터링
change = pd.concat([mean_park, mean_moon], axis=1)
change.columns = ["박근혜정부", "문재인정부"]
change.fillna(0, inplace=True)

# 중요도 필터: 노이즈 제거를 위해 상위 100개 체급 단어만 선정
change["importance"] = change["박근혜정부"] + change["문재인정부"]
change_filtered = change.sort_values("importance", ascending=False).head(100).copy()

# 증감률 계산
change_filtered["증감률(%)"] = (
    (change_filtered["문재인정부"] - change_filtered["박근혜정부"])
    / change_filtered["박근혜정부"].replace(0, 0.0001)
) * 100

change_sorted = change_filtered.sort_values("증감률(%)", ascending=False)
change_sorted.to_csv("../datas/keyword_change_verified.csv", encoding="utf-8-sig")
print("분석 완료: keyword_change_verified.csv 파일이 생성되었습니다.")

# 6. 시각화 1: 좌우 분할 바 차트
top10_moon = change_sorted.head(10)
top10_park = change_sorted.tail(10).sort_values("증감률(%)") 

fig, ax = plt.subplots(1, 2, figsize=(16, 8))

sns.barplot(x='증감률(%)', y=top10_moon.index, data=top10_moon, ax=ax[0], palette='Blues_r', hue=top10_moon.index, legend=False)
ax[0].set_title('문재인 정부 비중 급증 담론 (TOP 10)', fontsize=15)

sns.barplot(x='증감률(%)', y=top10_park.index, data=top10_park, ax=ax[1], palette='Reds', hue=top10_park.index, legend=False)
ax[1].set_title('박근혜 정부 우세 담론 (TOP 10)', fontsize=15)

plt.tight_layout()
plt.show()

# 7. 시각화 2: 0을 기준으로 양옆으로 뻗어나가는 Diverging Bar Chart

top_10 = change_sorted.head(10)
bottom_10 = change_sorted.tail(10)
plot_df = pd.concat([top_10, bottom_10]).sort_values('증감률(%)')

plt.figure(figsize=(12, 10))
# 막대 색상 설정
colors = ['red' if x < 0 else 'blue' for x in plot_df['증감률(%)']]

# 인덱스(키워드)를 y축으로 사용
plt.barh(plot_df.index, plot_df['증감률(%)'], color=colors)
plt.axvline(0, color='black', linewidth=1)
plt.title('정부 간 담론 변화율 (박근혜 vs 문재인)', fontsize=15)
plt.xlabel('증감률 (%)')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()