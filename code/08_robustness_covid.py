# ============================================================
# 08_robustness_covid.py
# 코로나 충격 분리 강건성 체크
#
# [방법 A] 시기 분할 비교: 2017-2019(코로나 전) vs 2020-2022(코로나 중)
#   - 문재인 정부 내에서 코로나 前/中 구간을 분리하여
#     TF-IDF 상위 키워드, Log-Odds 차이를 재산출
#   - 두 구간 공통·차별 키워드를 CSV로 저장
#
# [방법 B] 코로나 관련 토큰 제외 후 TF-IDF/Log-Odds 재분석
#   - 코로나, 바이러스, 확진, 방역, 마스크 등 키워드를 불용어 처리하고
#     동일 분석 파이프라인 재실행
#   - 원결과(방법 A) 대비 상위 50개 키워드 변화율을 비교표로 저장
# ============================================================

import os
import math
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform
from stopwords import STOPWORDS

# -------------------------------------------------------
# 0. 환경 설정
# -------------------------------------------------------
if platform.system() == "Windows":
    font_path = "C:/Windows/Fonts/malgun.ttf"
    font_name = fm.FontProperties(fname=font_path).get_name()
    plt.rc("font", family=font_name)
else:
    plt.rc("font", family="NanumGothic")
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = "../result/robustness"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_PATH = "../datas/preprocessed_2013_2022.csv"

# 코로나 관련 제거 토큰 (방법 B)
COVID_TOKENS = {
    "코로나", "코로나19", "바이러스", "확진", "확진자", "감염",
    "방역", "마스크", "백신", "접종", "격리", "자가격리",
    "거리두기", "사회적거리두기", "집합금지", "봉쇄", "팬데믹",
    "신종감염병", "중앙방역대책본부", "질병관리청", "방역당국",
    "코로나바이러스", "감염병", "변이", "오미크론", "델타",
}

STOPWORDS_SET = set(STOPWORDS(version="base")) | COVID_TOKENS

# -------------------------------------------------------
# 1. 데이터 로딩 및 기간 구분
# -------------------------------------------------------
print("데이터 로딩 중...")
df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
df = df.dropna(subset=["날짜", "text"])
df["year"] = df["날짜"].dt.year

# 문재인 정부 데이터만 추출 후 코로나 전/후 분할
mj_df = df[df["정부"] == "문재인정부"].copy()
pre_covid  = mj_df[mj_df["year"].between(2017, 2019)]   # 코로나 前
post_covid = mj_df[mj_df["year"].between(2020, 2022)]   # 코로나 中

print(f"문재인 정부 전체: {len(mj_df):,}건")
print(f"  코로나 前 (2017-2019): {len(pre_covid):,}건")
print(f"  코로나 中 (2020-2022): {len(post_covid):,}건")

# -------------------------------------------------------
# 2-A. TF-IDF 상위 키워드 비교 (방법 A: 시기 분할)
# -------------------------------------------------------
def tokenize_filter(text, extra_stopwords=None):
    sw = set(STOPWORDS(version="base"))
    if extra_stopwords:
        sw |= extra_stopwords
    return " ".join(
        w for w in str(text).split()
        if len(w) > 1 and w not in sw
    )

def get_tfidf_top(texts, n=50, extra_stopwords=None):
    cleaned = [tokenize_filter(t, extra_stopwords) for t in texts]
    vec = TfidfVectorizer(max_features=5000)
    mat = vec.fit_transform(cleaned)
    scores = mat.mean(axis=0).A1
    words = vec.get_feature_names_out()
    return pd.DataFrame({"키워드": words, "TF-IDF": scores}).sort_values(
        "TF-IDF", ascending=False
    ).head(n).reset_index(drop=True)

print("\n[방법 A] 코로나 전/후 TF-IDF 계산 중...")
pre_tfidf  = get_tfidf_top(pre_covid["text"],  n=50)
post_tfidf = get_tfidf_top(post_covid["text"], n=50)

pre_tfidf["시기"]  = "2017-2019(코로나前)"
post_tfidf["시기"] = "2020-2022(코로나中)"
pre_tfidf["순위"]  = range(1, len(pre_tfidf) + 1)
post_tfidf["순위"] = range(1, len(post_tfidf) + 1)

tfidf_split = pd.concat([pre_tfidf, post_tfidf])
tfidf_split.to_csv(
    os.path.join(OUTPUT_DIR, "tfidf_covid_split.csv"),
    index=False, encoding="utf-8-sig"
)
print("저장: tfidf_covid_split.csv")

# 공통 키워드 (안정성 확인)
pre_set  = set(pre_tfidf["키워드"])
post_set = set(post_tfidf["키워드"])
common   = pre_set & post_set
pd.DataFrame({"공통키워드": sorted(common)}).to_csv(
    os.path.join(OUTPUT_DIR, "tfidf_covid_common_keywords.csv"),
    index=False, encoding="utf-8-sig"
)
print(f"공통 키워드 수: {len(common)}")

# TF-IDF 상위 20 시각화 비교
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
for ax, (sub_df, title) in zip(axes, [
    (pre_tfidf.head(20),  "2017-2019 (코로나 前)"),
    (post_tfidf.head(20), "2020-2022 (코로나 中)"),
]):
    ax.barh(sub_df["키워드"][::-1], sub_df["TF-IDF"][::-1])
    ax.set_title(title)
    ax.set_xlabel("평균 TF-IDF")
plt.suptitle("문재인 정부 코로나 前/中 TF-IDF 상위 20 키워드 비교")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig_tfidf_covid_split.png"), dpi=300)
plt.close()
print("저장: fig_tfidf_covid_split.png")

# -------------------------------------------------------
# 2-B.  Log-Odds 비교 (코로나 前 vs 코로나 中)
# -------------------------------------------------------
def word_freq(texts, sw):
    counter = Counter()
    for t in texts:
        words = [w for w in str(t).split() if len(w) > 1 and w not in sw]
        counter.update(words)
    return counter

def log_odds(count_a, count_b, prior_weight=0.01):
    """Informative Dirichlet prior log-odds ratio"""
    vocab = set(count_a.keys()) | set(count_b.keys())
    total_a = sum(count_a.values())
    total_b = sum(count_b.values())
    rows = []
    for w in vocab:
        fa = count_a.get(w, 0)
        fb = count_b.get(w, 0)
        if fa + fb < 5:
            continue
        pa = (fa + prior_weight) / (total_a + prior_weight * len(vocab))
        pb = (fb + prior_weight) / (total_b + prior_weight * len(vocab))
        lo = math.log(pa / pb)
        rows.append({"키워드": w, "log_odds": lo,
                     "빈도_A": fa, "빈도_B": fb})
    return pd.DataFrame(rows).sort_values("log_odds", ascending=False)

print("\n[방법 A] Log-Odds 계산 중...")
BASE_SW = set(STOPWORDS(version="base"))
freq_pre  = word_freq(pre_covid["text"],  BASE_SW)
freq_post = word_freq(post_covid["text"], BASE_SW)
lo_df = log_odds(freq_post, freq_pre)   # 양수 = 코로나 中 강화

lo_top = pd.concat([lo_df.head(30), lo_df.tail(30)])
lo_top.columns = ["키워드", "Log-Odds(코로나中-前)", "빈도_코로나中", "빈도_코로나前"]
lo_top.to_csv(
    os.path.join(OUTPUT_DIR, "log_odds_covid_split.csv"),
    index=False, encoding="utf-8-sig"
)
print("저장: log_odds_covid_split.csv")

# -------------------------------------------------------
# 3. 방법 B: 코로나 토큰 제거 후 전체(박근혜 vs 문재인) 재분석
# -------------------------------------------------------
print("\n[방법 B] 코로나 토큰 제거 후 전체 정부 비교 재분석...")

pk_df = df[df["정부"] == "박근혜정부"]

freq_pk_b = word_freq(pk_df["text"],  STOPWORDS_SET)
freq_mj_b = word_freq(mj_df["text"], STOPWORDS_SET)

lo_b = log_odds(freq_mj_b, freq_pk_b)
lo_b_top = pd.concat([lo_b.head(30), lo_b.tail(30)])
lo_b_top.columns = ["키워드", "Log-Odds(문재인-박근혜)", "빈도_문재인", "빈도_박근혜"]
lo_b_top.to_csv(
    os.path.join(OUTPUT_DIR, "log_odds_covid_excluded.csv"),
    index=False, encoding="utf-8-sig"
)
print("저장: log_odds_covid_excluded.csv")

# 원결과(../result/log_odds_top_*.csv)와 상위 30 키워드 변화 비교
def compare_with_original(original_csv, new_df, label):
    try:
        orig = pd.read_csv(original_csv, encoding="utf-8-sig")
    except FileNotFoundError:
        print(f"원결과 파일 없음: {original_csv}")
        return
    
    col_name = "키워드" if "키워드" in orig.columns else "단어"
    orig_kw = set(orig[col_name].head(30))
    new_kw  = set(new_df["키워드"].head(30))
    overlap = orig_kw & new_kw
    stability = len(overlap) / 30 * 100
    print(f"\n[{label}] 상위 30 키워드 안정성: {stability:.1f}% ({len(overlap)}/30 유지)")
    pd.DataFrame({
        "원결과 키워드": sorted(orig_kw),
        "유지 여부": [("O" if k in new_kw else "X") for k in sorted(orig_kw)]
    }).to_csv(
        os.path.join(OUTPUT_DIR, f"stability_{label}.csv"),
        index=False, encoding="utf-8-sig"
    )

compare_with_original(
    "../result/log_odds_top_moon.csv",
    lo_b[lo_b["log_odds"] > 0],
    "문재인정부"
)
compare_with_original(
    "../result/log_odds_top_park.csv",
    lo_b[lo_b["log_odds"] < 0].assign(log_odds=lambda x: -x["log_odds"]),
    "박근혜정부"
)

# -------------------------------------------------------
# 4. 요약 테이블: 코로나 前/後 연도별 기사 수
# -------------------------------------------------------
yearly = (
    mj_df.groupby("year")
    .size()
    .reset_index(name="기사수")
    .assign(코로나구분=lambda x: x["year"].apply(
        lambda y: "코로나前(2017-2019)" if y <= 2019 else "코로나中(2020-2022)"
    ))
)
yearly.to_csv(
    os.path.join(OUTPUT_DIR, "yearly_article_count_mj.csv"),
    index=False, encoding="utf-8-sig"
)
print("\n저장: yearly_article_count_mj.csv")
print("\n[완료] 08_robustness_covid.py 실행 완료")
print(f"결과 디렉토리: {OUTPUT_DIR}")
