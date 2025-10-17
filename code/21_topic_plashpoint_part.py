import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager, rc
import platform
import numpy as np

# =========================================
# 🔤 한글 폰트 설정
# =========================================
if platform.system() == "Windows":
    font_path = "C:/Windows/Fonts/malgun.ttf"
elif platform.system() == "Darwin":
    font_path = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
else:
    font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"

font_name = font_manager.FontProperties(fname=font_path).get_name()
rc("font", family=font_name)
plt.rcParams["axes.unicode_minus"] = False

# =========================================
# 🧹 불용어 사전 정의
# =========================================
STOPWORDS = set([    # 정치 및 기관명
    '정부', '공직자', '공무원', '정책', '국가', '제도',
    '위원회', '부처', '국회', '장관', '대통령', '청와대', '서울시',
    '의원', '기자', '위원', '위원장', '관계자', '단체', '기관', '본부',
    '발표', '회의', '예산', '조사', '보고서', '언론', '나라', '시장', '의장',

    # 정치인 및 정당
    '윤석열', '윤석렬', '김건희', '이재명', '조국', '문재인', '박근혜', "박원순",
    '이명박', '국민의힘', '민주당', '더불어민주당', '국힘', '혁신당', '민주',
    '진보', '보수', '더불어',

    # 지명 및 지역
    '한국', '대한민국', '서울', '일본', '대만', '지방', '강진군', '전국', '인천',

    # 불필요 연결어 / 기능어
    '관련', '통해', '의견', '논의', '추진', '방안', '지적',
    '확대', '요구', '대상', '중요', '사실', '입장', '시작', '처리','위해', '대한', '이번',

    # 사건·사고 및 뉴스용어
    '사건', '사안', '피해자', '성폭력', '성희롱', '비서', '비서실',
    '투표', '선거', '대선', '탄핵', '촛불', '법안', '검찰', '심판',

    # 역사/의례/국가보훈
    '애국', '전쟁', '호국', '현충일', '독립운동', '독립운동가', '태극기', '유공', '보훈', '희생',

    # 일반 사회어
    '학교', '학생', '아이', '유치원', '부모', '친구', '급식',
    '출산율', '가족', '노인', '주택', '지역', '부담', '교수', '회사', '시민', '보도',

    # 기사 작성용 표현
    '대표', '기획', '기분', '결정', '사실', '지난달', '시간', '내년', '사정', '선임', '발생', '신고',

    # 기타 불필요 단어
    '효능', '연구원', '연구소', '사회학', '독자', '평가', '대책', '악성', '피크', '지난해', '인상', '지난', '추념', '주목', '최근', '이상', '사람', '여론조사', '생각'])

# =========================================
# 1️⃣ 데이터 병합 및 시기 구분
# =========================================
files = [
    "../datas/freq_2015_2019.csv",
    "../datas/freq_2020_2025.csv"
]

dfs = []
for file in files:
    if os.path.exists(file):
        df_temp = pd.read_csv(file, encoding="utf-8-sig")

        if "year" not in df_temp.columns:
            print(f"⚠️ {file} 파일에 year 컬럼이 없습니다. year 추출 불가.")
            continue

        # 시기 자동 분류
        def classify_period(y):
            if pd.isna(y):
                return "미상"
            elif y <= 2016:
                return "1기(2015–2016)"
            elif 2017 <= y <= 2021:
                return "2기(2017–2021)"
            else:
                return "3기(2022–2025)"

        df_temp["period"] = df_temp["year"].apply(classify_period)

        # ✅ 불용어 제거
        df_temp = df_temp[~df_temp["word"].isin(STOPWORDS)]

        dfs.append(df_temp)
    else:
        print(f"⚠️ 파일 없음: {file}")

df = pd.concat(dfs, ignore_index=True)
print(f"✅ 데이터 병합 완료! 총 {len(df)}건 단어-연도 데이터 로드됨")
print(df["period"].value_counts())

# =========================================
# 2️⃣ 시기별 상위 20개 단어 계산
# =========================================
top_words_by_period = (
    df.groupby(["period", "word"])["freq"]
      .sum()
      .reset_index()
      .sort_values(["period", "freq"], ascending=[True, False])
      .groupby("period")
      .head(20)
)

# =========================================
# 🔎 터미널 출력: 시기별 상위 10개 단어
# =========================================
for period, group in top_words_by_period.groupby("period"):
    print(f"\n📊 {period} 주요 키워드 TOP 10")
    print(group.head(10)[["word", "freq"]].to_string(index=False))

# =========================================
# 3️⃣ 시각화
# =========================================
plt.figure(figsize=(12, 8))
sns.barplot(
    data=top_words_by_period,
    x="freq", y="word",
    hue="period",
    palette=["#8EC5FC", "#E0C3FC", "#F9D423"]
)
plt.title("시기별 주요 키워드 Top 20 비교 (불용어 제거)", fontsize=16)
plt.xlabel("빈도수", fontsize=13)
plt.ylabel("단어", fontsize=13)
plt.legend(title="시기", loc="upper right")
plt.tight_layout()
plt.show()


trend = (
    df.groupby(["year", "word"])["freq"]
      .sum()
      .reset_index()
)

# 상위 10개 단어만 필터링
top10_words = top_words_by_period["word"].unique()[:10]
trend_top = trend[trend["word"].isin(top10_words)]

plt.figure(figsize=(12,6))
sns.lineplot(data=trend_top, x="year", y="freq", hue="word", marker="o")
plt.title("2015–2025 주요 키워드 시계열 변화 추세", fontsize=14)
plt.xlabel("연도")
plt.ylabel("빈도수")
plt.legend(title="키워드", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()