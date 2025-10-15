# ============================================================
#   파일명: 09_keyword_trend_visualization.py
#   기능: 시기별 주요 키워드 및 변화율 시각화
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from konlpy.tag import Okt
import platform

# ================================
# ⚙️ 폰트 설정 (macOS 한글 깨짐 방지)
# ================================
if platform.system() == 'Darwin':  # macOS
    plt.rc('font', family='AppleGothic')
else:  # Windows / Linux 대비
    plt.rc('font', family='Malgun Gothic')

plt.rcParams['axes.unicode_minus'] = False  # 마이너스 깨짐 방지

# 🔹 형태소 분석기
okt = Okt()

# 🔹 불용어 설정
EXCLUDE_WORDS = [
    # 정치 및 기관명
    '정부', '공직자', '공무원', '정책', '국가', '제도',
    '위원회', '부처', '국회', '장관', '대통령', '청와대', '서울시',
    '의원', '기자', '위원', '위원장', '관계자', '단체', '기관', '본부',
    '발표', '회의', '예산', '조사', '보고서', '언론',

    # 정치인 및 정당
    '윤석열', '윤석렬', '김건희', '이재명', '조국', '문재인', '박근혜',
    '이명박', '국민의힘', '민주당', '더불어민주당', '국힘', '혁신당', '민주',
    '진보', '보수', '더불어',

    # 지명 및 지역
    '한국', '대한민국', '서울', '일본', '대만', '지방', '강진군', '전국', '인천',

    # 불필요 연결어 / 기능어
    '관련', '통해', '의견', '논의', '추진', '방안', '지적',
    '확대', '요구', '대상', '중요', '사실', '입장', '시작', '처리',

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
    '효능', '연구원', '연구소', '사회학', '독자', '평가', '대책', '악성'
]

# 🔹 명사 추출 함수
def extract_nouns(text):
    words = okt.nouns(str(text))
    words = [w for w in words if len(w) > 1 and w not in EXCLUDE_WORDS]
    return words

# 🔹 CSV 파일 처리 함수
def process_csv(file_path, output_name):
    df = pd.read_csv(file_path)
    text = ' '.join(df['clean_text'].astype(str))  # ✅ '본문' 대신 'clean_text'
    words = extract_nouns(text)
    counter = Counter(words)
    result = pd.DataFrame(counter.most_common(50), columns=['단어', '빈도'])
    result.to_csv(output_name, index=False, encoding='utf-8-sig')
    return result

# 🔹 파일 경로
file_2015_2019 = '../datas/clean_2015_2019.csv'
file_2020_2025 = '../datas/clean_2020_2025.csv'

# 🔹 데이터 처리
dedup_2015_2019 = process_csv(file_2015_2019, '../datas/dedup_2015_2019.csv')
dedup_2020_2025 = process_csv(file_2020_2025, '../datas/dedup_2020_2025.csv')

# ============================================================
#   (1) 2015–2019 주요 키워드 상위 20 시각화
# ============================================================
plt.figure(figsize=(10, 6))
plt.barh(dedup_2015_2019['단어'].head(20)[::-1], dedup_2015_2019['빈도'].head(20)[::-1])
plt.title('① 2015–2019 주요 키워드 상위 20', fontsize=14)
plt.xlabel('빈도')
plt.ylabel('단어')
plt.tight_layout()
plt.savefig('../datas/keyword_top_2015_2019.png', dpi=300)
plt.show()

# ============================================================
#   (2) 2020–2025 주요 키워드 상위 20 시각화
# ============================================================
plt.figure(figsize=(10, 6))
plt.barh(dedup_2020_2025['단어'].head(20)[::-1], dedup_2020_2025['빈도'].head(20)[::-1])
plt.title('② 2020–2025 주요 키워드 상위 20', fontsize=14)
plt.xlabel('빈도')
plt.ylabel('단어')
plt.tight_layout()
plt.savefig('../datas/keyword_top_2020_2025.png', dpi=300)
plt.show()

# ============================================================
#   (3) 두 시기 비교 변화율 시각화
# ============================================================
trend_df = pd.merge(
    dedup_2015_2019, dedup_2020_2025, on='단어', how='outer', suffixes=('_15_19', '_20_25')
).fillna(0)

trend_df['변화율(%)'] = ((trend_df['빈도_20_25'] - trend_df['빈도_15_19']) / (trend_df['빈도_15_19'] + 1)) * 100
trend_df.to_csv('../datas/keyword_trend.csv', index=False, encoding='utf-8-sig')

top_change = trend_df.sort_values('변화율(%)', ascending=False).head(15)

plt.figure(figsize=(10, 6))
plt.barh(top_change['단어'], top_change['변화율(%)'])
plt.title('③ 공직 인식 주요 키워드 변화 (2015–2019 → 2020–2025)', fontsize=14)
plt.xlabel('빈도 변화율 (%)')
plt.ylabel('단어')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('../datas/keyword_trend_change.png', dpi=300)
plt.show()
