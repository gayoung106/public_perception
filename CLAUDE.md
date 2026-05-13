# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**공직사회 담론의 구조적 재편 분석** (Structural Analysis of Public Sector Discourse Reconfiguration)

This research project analyzes news discourse regarding public sector perceptions during two South Korean administrations:
- Park Geun-hye Government: 2013-02-25 ~ 2017-03-10
- Moon Jae-in Government: 2017-05-10 ~ 2022-05-09

The project uses NLP techniques to extract and compare key discourse themes, keywords, and conceptual networks between the two periods.

## Data Pipeline Architecture

The analysis follows a linear processing pipeline with numbered stages:

### Stage 1: Data Conversion & Consolidation
- **01_convert.py**: Converts raw Excel data files (`news_*.xlsx` in `/datas`) to CSV format
- **02_merge.py**: Merges all monthly CSV files, deduplicates articles, validates dates, and creates `news_2013_2022_merged.csv`

### Stage 2: Preprocessing
- **03_preprocess_2.py**: 
  - Filters articles by government period (removes non-2013-2022, non-Park/Moon articles)
  - Combines title, body, keywords, and weighted features into single text field
  - Removes special characters (keeps only Korean + whitespace)
  - Outputs `preprocessed_2013_2022.csv` (clean, government-tagged corpus)

### Stage 3: Quantitative Analysis Methods
Multiple complementary analysis approaches run on the preprocessed corpus:

**TF-IDF Analysis:**
- `05_1_tfidf_base.py`: Baseline TF-IDF comparing discourse importance between administrations
- `05_1_tfidf_conditional.py`: Conditional TF-IDF with filtered stopwords
- `05_3_1_tfidf_50.py`: Top-50 word visualization

**Log-Odds Analysis:**
- `05_2_log_odd_base.py`: Baseline log-odds ratios
- `05_2_log_odd_conditional.py`: Conditional log-odds
- `05_3_2_log_odd_diff.py`: Differential analysis
- `05_3_3_log_tfidf_cross.py`: Cross-method comparison

**Keyword-In-Context (KWIC):**
- `05_3_4_kwic.py`: Extracts sample sentences for target words
- `05_3_5_final.py`: Final KWIC processing
- `05_6_6_kwic_final.py`: KWIC finalization

### Stage 4: Visualization & Network Analysis
- **04_wordcloud_2.py**: Word frequency visualization (government-wise word clouds)
- **06_cooccurrence.py**: PMI-based word co-occurrence network analysis
- **07_lda_final.py**: Latent Dirichlet Allocation topic modeling (10 topics)
- **07_lda_topic_no.py**: Topic number optimization
- **07_topic_bar.py**: Topic keyword visualizations
- **10_centrality_comparison.py**: Network centrality metrics (degree, betweenness, closeness, eigenvector)

### Stage 5: Robustness & Sensitivity Testing
- **08_robustness_covid.py**: COVID-19 period robustness check (2017-2019 vs 2020-2022 within Moon govt)
- **09_lda_seed_sensitivity.py**: LDA random seed sensitivity analysis
- **09_lda_unified.py**: Unified LDA processing

### Stage 6: Concordance
- **06_concordance_sample.py**: Sample-based concordance analysis

## Key Dependencies & Configuration

### Python Environment
- Python 3.11 with Windows support
- Virtual environment: `.venv/` (already created)

### Core Libraries
- **Data Processing**: pandas, numpy, openpyxl
- **NLP**: konlpy (Korean morphological analysis), nltk, scikit-learn, regex, gensim
- **Visualization**: matplotlib, seaborn, wordcloud, networkx, fpdf
- **Utilities**: tqdm, beautifulsoup4, requests, fonttools

### Stopwords Management
- **stopwords.py**: Central stopwords dictionary with two versions:
  - `"base"`: Core stopwords (preserves key concept words like 불안, 신뢰, 인식, 차별, 권리)
  - `"with_conditional"`: Adds conditional filtering for domain-specific words
  - Excludes ~300+ political figures, institutions, and irrelevant terms

## Running the Analysis

### Initial Setup
```powershell
# Windows PowerShell in the project root directory
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Running Individual Analysis Stages
```powershell
cd code

# Data preparation (run sequentially - each depends on previous output)
python 01_convert.py           # .xlsx to CSV
python 02_merge.py             # Merge & deduplicate
python 03_preprocess_2.py       # Create clean corpus

# Quantitative analyses (can run in parallel after preprocessing)
python 05_1_tfidf_base.py       # TF-IDF baseline
python 05_2_log_odd_base.py     # Log-odds baseline
python 07_lda_final.py          # Topic modeling
python 04_wordcloud_2.py        # Word clouds
python 06_cooccurrence.py       # Co-occurrence networks
python 10_centrality_comparison.py  # Network centrality

# Robustness checks
python 08_robustness_covid.py   # COVID sensitivity
python 09_lda_seed_sensitivity.py  # LDA seed sensitivity

# Keyword visualization
python 05_3_4_kwic.py           # KWIC extraction
```

## Data Directory Structure

- `/datas/`: Input and intermediate data files
  - `news_*.csv`: Monthly article datasets (2013-2022)
  - `news_2013_2022_merged.csv`: Consolidated raw corpus
  - `preprocessed_2013_2022.csv`: Final clean corpus (main analysis input)
- `/result/`: Output files from analyses
  - Subdirectories: `lda/`, `robustness/`, etc.
  - CSV outputs: keyword rankings, topic distributions, centrality metrics
  - PNG outputs: visualizations, network diagrams

## Data Schema

**preprocessed_2013_2022.csv** (main analysis corpus):
- `날짜` (date): YYYY-MM-DD format
- `year`: Integer year
- `정부` (government): "박근혜정부" or "문재인정부"
- `언론사` (media outlet): Source identifier
- `text`: Combined preprocessed text (title + body + keywords + features)

## Important Design Patterns

### 1. Government-Based Segmentation
All analyses explicitly filter data by `정부` column. Comparisons use:
- Park baseline (2013-2017) as reference vocabulary for TF-IDF
- Moon government (2017-2022) transformed to same vocabulary space
- This ensures keyword differences reflect real discourse shifts, not vocabulary changes

### 2. Stopwords as Analysis Parameter
- Stopwords module is imported in most scripts: `from stopwords import STOPWORDS`
- Different analysis methods use different stopword versions
- Keep KEEP_WORDS set (신뢰, 불안, 인식, 차별, 권리 등) untouched for concept preservation

### 3. Font Handling for Korean Output
- Windows: Uses Malgun Gothic (`malgun.ttf`)
- Linux: Uses NanumGothic
- Scripts detect OS with `platform.system()` for conditional font configuration

### 4. Multiprocessing in LDA
- `07_lda_final.py` uses `LdaMulticore` with `cpu_count() - 1` workers
- Parameters: 10 topics, 5 passes, 100 iterations, symmetric alpha, auto eta

### 5. Random Seed Management
- Reproducibility: Seeds set to 42 in sampling operations
- Sensitivity testing: `09_lda_seed_sensitivity.py` tests multiple seeds

## Common Modifications

### Adding New Stopwords
Edit `stopwords.py`:
1. Add word to appropriate category comment in `STOPWORDS_LIST`
2. Or to `CONDITIONAL_STOPWORDS` for context-dependent filtering
3. Ensure word doesn't overlap with `KEEP_WORDS`
4. No code changes needed—all scripts re-import dynamically

### Adjusting Analysis Parameters
- **TF-IDF**: max_features, min_df, max_df in script variables
- **LDA**: NUM_TOPICS, PASSES, ITERATIONS at top of `07_lda_final.py`
- **Co-occurrence PMI**: min_co_cnt threshold in `06_cooccurrence.py`
- **Date filtering**: Government period dates in `03_preprocess_2.py` (lines 21-26)

### Adding New Analysis Script
1. Import preprocessed data: `pd.read_csv("../datas/preprocessed_2013_2022.csv", encoding="utf-8-sig")`
2. Filter by government: `df[df["정부"] == "박근혜정부"]`
3. Load stopwords: `from stopwords import STOPWORDS; STOPWORDS_LIST = STOPWORDS(version="base")`
4. Output to `/result/` with descriptive filename
5. Use numbering convention (e.g., `11_*.py`) for pipeline consistency

## Output Files Convention

All analysis outputs save to `/result/` with patterns:
- CSVs: `[analysis]_[metric].csv` (e.g., `kwic_samples.csv`, `centrality_by_government.csv`)
- PNGs: `fig_[analysis]_[variant].png` (e.g., `fig_wordcloud_park.png`)
- Topic models: `/result/lda/` subdirectory

## Encoding Notes

All CSV files use `utf-8-sig` (UTF-8 with BOM) encoding for Windows compatibility. When reading/writing:
- Read: `encoding="utf-8-sig"`
- Write: `encoding="utf-8-sig"`
