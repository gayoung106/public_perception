import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from konlpy.tag import Okt
import seaborn as sns
import os
from matplotlib import font_manager, rc
import platform

# ==========================
# ✅ 한글 폰트 설정
# ==========================
if platform.system() == 'Windows':
    font_path = "C:/Windows/Fonts/malgun.ttf"  # 맑은 고딕
elif platform.system() == 'Darwin':  # macOS
    font_path = "/System/Library/Fonts/AppleGothic.ttf"
else:
    font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"  # 리눅스

font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 깨짐 방지

# ===============================
# 1️⃣ 데이터 병합
# ===============================
files = [
    "../datas/public_perception_2015_2019.csv",
    "../datas/public_perception_2020_2025.csv"
]

dfs = []
for file in files:
    if os.path.exists(file):
        df_temp = pd.read_csv(file, encoding="utf-8-sig")
        if "2015_2019" in file:
            df_temp["period"] = "2015–2019"
        elif "2020_2025" in file:
            df_temp["period"] = "2020–2025"
        dfs.append(df_temp)
    else:
        print(f"⚠️ 파일 없음: {file}")

df = pd.concat(dfs, ignore_index=True)
print(f"✅ 데이터 병합 완료! 총 {len(df)}건 기사 로드됨")

# ===============================
# 2️⃣ 본문 컬럼 자동 탐색
# ===============================
content_col_candidates = ["content", "본문", "summary", "요약", "내용", "text"]
available_cols = df.columns.tolist()

content_col = None
for col in content_col_candidates:
    if col in available_cols:
        content_col = col
        break

if not content_col:
    raise KeyError(f"⚠️ 본문에 해당하는 컬럼을 찾을 수 없습니다. 현재 컬럼: {available_cols}")

print(f"📰 본문 컬럼 사용: '{content_col}'")

# ===============================
# 3️⃣ 형태소 분석 (명사 추출)
# ===============================
okt = Okt()

def extract_nouns(text):
    if pd.isna(text):
        return []
    nouns = okt.nouns(str(text))
    stopwords = [    # 정치 및 기관명
    '정부', '공직자', '공무원', '정책', '국가', '제도',
    '위원회', '부처', '국회', '장관', '대통령', '청와대', '서울시',
    '의원', '기자', '위원', '위원장', '관계자', '단체', '기관', '본부',
    '발표', '회의', '예산', '조사', '보고서', '언론', '나라', '시장',

    # 정치인 및 정당
    '윤석열', '윤석렬', '김건희', '이재명', '조국', '문재인', '박근혜',
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
    '출산율', '가족', '노인', '주택', '지역', '부담',

    # 기사 작성용 표현
    '대표', '기획', '기분', '결정', '사실', '지난달', '시간', '내년', '사정', '선임', '발생', '신고',

    # 기타 불필요 단어
    '효능', '연구원', '연구소', '사회학', '독자', '평가', '대책', '악성', '피크', '지난해', '인상', '지난', '추념', '주목', '최근', '이상', '사람', '여론조사']
    return [n for n in nouns if len(n) > 1 and n not in stopwords]

df["nouns"] = df[content_col].apply(extract_nouns)

# ===============================
# 4️⃣ 시기별 단어 빈도 계산
# ===============================
word_counts = {}
for period in ["2015–2019", "2020–2025"]:
    all_words = sum(df[df["period"] == period]["nouns"], [])
    freq = Counter(all_words).most_common(20)
    word_counts[period] = pd.DataFrame(freq, columns=["단어", "빈도"])

# ===============================
# 5️⃣ 시각화
# ===============================
plt.figure(figsize=(10, 6))
sns.barplot(
    data=pd.concat([
        word_counts["2015–2019"].assign(시기="2015–2019"),
        word_counts["2020–2025"].assign(시기="2020–2025")
    ]),
    x="빈도", y="단어", hue="시기"
)
plt.title("시기별 주요 키워드 빈도 비교 (Top 20)", fontsize=14)
plt.xlabel("빈도수", fontsize=12)
plt.ylabel("단어", fontsize=12)
plt.legend(title="시기")
plt.tight_layout()

output_path = "../datas/keyword_trend_compare.png"
plt.savefig(output_path, dpi=300)
plt.show()

print("✅ 시기별 주요 키워드 비교 완료!")
print(f"📊 결과 저장: {output_path}")

# ===============================
# 6️⃣ 결과 요약 CSV 저장
# ===============================
combined_df = pd.concat([
    word_counts["2015–2019"].rename(columns={"빈도": "2015–2019"}),
    word_counts["2020–2025"].rename(columns={"빈도": "2020–2025"})
], axis=1)

combined_df.to_csv("../datas/period_keyword_summary.csv", index=False, encoding="utf-8-sig")
print("✅ 시기별 키워드 요약 저장 완료: period_keyword_summary.csv")
