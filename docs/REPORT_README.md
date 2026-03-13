# Helix SJM 보고서 - PDF 저장 가이드

## 보고서 파일
- `report_helix_sjm_korean.html` — 한국어 성과 보고서 (HTML)

## PDF로 저장하는 방법

### 방법 1: 브라우저에서 인쇄 (권장)
```bash
# Mac: HTML 파일을 기본 브라우저로 열기
open report_helix_sjm_korean.html

# 또는
python export_report_pdf.py --browser
```
브라우저가 열리면:
1. **Mac**: `Cmd + P` → 대상에서 **"PDF로 저장"** 선택
2. **Windows**: `Ctrl + P` → 프린터에서 **"PDF로 저장"** 또는 "Microsoft Print to PDF" 선택
3. 저장 위치 지정 후 저장

### 방법 2: weasyprint로 자동 PDF 생성
```bash
pip install weasyprint
python export_report_pdf.py
```
→ `report_helix_sjm_korean.pdf` 파일이 생성됩니다.

### 스크린샷
보고서 전체 또는 특정 섹션을 스크린샷하려면:
- **Mac**: `Cmd + Shift + 4` (영역 선택) 또는 `Cmd + Shift + 5` (전체 화면)
- **Windows**: `Win + Shift + S` (스니핑 도구)

---

이 보고서는 하이퍼리즘 CeFi Systematic Trader 포지션 지원 시 자동화된 트레이딩 봇 성과를 제시하는 자료로 활용할 수 있습니다.
