# ============================================================
#  파일명: 15_topic_final_report.py
#  목적: 시각자료 + 텍스트 요약을 하나의 논문형 결과 리포트로 통합
# ============================================================

import pandas as pd
from fpdf import FPDF
import os

# ✅ 한글 폰트 경로 설정 (Apple의 경우 기본 폰트 사용 가능)
FONT_PATH = "/System/Library/Fonts/AppleSDGothicNeo.ttc"  # macOS 기준

class PDF(FPDF):
    def header(self):
        self.set_font("AppleGothic", "B", 16)
        self.cell(0, 10, "공직사회 담론 변화 분석 리포트 (2015–2025)", ln=True, align="C")
        self.ln(10)

# 🔹 1. PDF 객체 생성
pdf = PDF()
pdf.set_auto_page_break(auto=True, margin=15)

# ✅ 한글 폰트 등록
pdf.add_font("AppleGothic", "", FONT_PATH, uni=True)
pdf.add_font("AppleGothic", "B", FONT_PATH, uni=True)
pdf.set_font("AppleGothic", size=12)

# 🔹 2. 데이터 불러오기
summary_path = "../datas/topic_narrative_summary.txt"
images_dir = "../datas"

# 🔹 3. 본문 추가
pdf.add_page()
with open(summary_path, "r", encoding="utf-8") as f:
    paragraphs = f.readlines()

for p in paragraphs:
    if p.strip():
        pdf.multi_cell(0, 8, p.strip())
        pdf.ln(4)

# 🔹 4. 시각자료 추가
pdf.add_page()
pdf.set_font("AppleGothic", "B", 14)
pdf.cell(0, 10, "📊 시각화 결과", ln=True)
pdf.ln(5)

image_files = [
    "../datas/2015–2019_topic1.png",
    "../datas/2020–2025_topic1.png",
    "../datas/keyword_trend_change.png",
    "../datas/cooccurrence_network.png"
]

for img in image_files:
    path = os.path.join(images_dir, img)
    if os.path.exists(path):
        pdf.image(path, w=160)
        pdf.ln(10)
    else:
        pdf.cell(0, 8, f"[이미지 없음] {img}", ln=True)

# 🔹 5. PDF 저장
output_path = "../datas/final_topic_report.pdf"
pdf.output(output_path)
print("✅ 논문용 통합 리포트 저장 완료:", output_path)
