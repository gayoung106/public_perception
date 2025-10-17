import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from matplotlib import font_manager, rc
import platform

# ✅ 1. 한글 폰트 설정 (Windows용)
if platform.system() == "Windows":
    font_path = "C:/Windows/Fonts/malgun.ttf"  # 맑은 고딕
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    rc("font", family=font_name)

# ✅ 2. 마이너스 기호 깨짐 방지
plt.rcParams["axes.unicode_minus"] = False



# 파일 구조 예시:
#  year | word | freq
#  2015 | 청렴 | 123

stopwords = set([    # 정치 및 기관명
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
    '전쟁', '호국', '현충일', '독립운동', '독립운동가', '태극기', '유공', '보훈', '희생',

    # 일반 사회어
    '학교', '학생', '아이', '유치원', '부모', '친구', '급식',
    '출산율', '가족', '노인', '주택', '지역', '부담', '교수', '회사', '시민', '보도',

    # 기사 작성용 표현
    '대표', '기획', '기분', '결정', '사실', '지난달', '시간', '내년', '사정', '선임', '발생', '신고',

    # 기타 불필요 단어
    '효능', '연구원', '연구소', '사회학', '독자', '평가', '대책', '악성', '피크', '지난해', '인상', '지난', '추념', '주목', '최근', '이상', '사람', '여론조사'])

# -------------------------------
# 1️⃣ 데이터 불러오기 및 병합
# -------------------------------
freq1 = pd.read_csv("../datas/freq_2015_2019.csv", encoding="utf-8-sig")
freq2 = pd.read_csv("../datas/freq_2020_2025.csv", encoding="utf-8-sig")

# 두 파일 합치기
df = pd.concat([freq1, freq2], ignore_index=True)

# ✅ 불용어 제거
df = df[~df["word"].isin(stopwords)]

# year 정수형 변환
df["year"] = df["year"].astype(int)
df = df.groupby(["year", "word"], as_index=False)["freq"].sum()

# -------------------------------
# 2️⃣ 연도별 단어 빈도 pivot 변환
# -------------------------------
pivot = df.pivot(index="year", columns="word", values="freq").fillna(0)

# 스케일 정규화 (변화율 비교를 위해)
scaler = MinMaxScaler()
pivot_scaled = pd.DataFrame(scaler.fit_transform(pivot), columns=pivot.columns, index=pivot.index)

# -------------------------------
# 3️⃣ 단어별 변화율 계산
# -------------------------------
change_rate = pivot_scaled.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)

# 단어별 평균 변화율 계산
mean_change = change_rate.mean().sort_values(ascending=False)

# 상위 15개 급상승 단어 추출 (자동)
top_keywords = mean_change.head(15).index.tolist()
print("📈 자동 탐색된 주요 급상승 키워드:")
print(top_keywords)

# -------------------------------
# 4️⃣ 플래시포인트(변곡점) 탐지
# -------------------------------
# 연도별 전체 변화량의 평균 절댓값 계산
yearly_change_strength = change_rate.abs().mean(axis=1)

# 급격히 변화한 연도 (상위 20% 이상)
threshold = yearly_change_strength.quantile(0.8)
flash_points = yearly_change_strength[yearly_change_strength >= threshold].index.tolist()

print("\n⚡ 탐지된 플래시포인트(급격한 변화 시점):", flash_points)

# -------------------------------
# 5️⃣ 시각화
# -------------------------------
plt.figure(figsize=(12,6))
sns.lineplot(data=pivot_scaled[top_keywords])
plt.title("2015–2025 자동 탐색 주요 단어 시계열 추세 (변화율 기반)", fontsize=13)
plt.ylabel("정규화 빈도수")
plt.xlabel("연도")

# 플래시포인트 표시
for fp in flash_points:
    plt.axvline(fp, color='red', linestyle='--', alpha=0.6)
    plt.text(fp+0.1, 1.02, f"Flash Point {fp}", rotation=90, color='red', fontsize=9)

plt.legend(title="자동 탐색 키워드", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
