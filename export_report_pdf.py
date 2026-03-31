#!/usr/bin/env python3
"""
Export the Korean Helix SJM report to PDF.

Options:
1. If weasyprint is installed: generates PDF directly
2. Otherwise: opens HTML in browser (use Cmd+P / Ctrl+P → Save as PDF)
"""

import os
import sys
import webbrowser
from pathlib import Path

REPO = Path(__file__).resolve().parent
HTML_PATH = REPO / "report_helix_sjm_korean.html"
PDF_PATH = REPO / "report_helix_sjm_korean.pdf"


def export_with_weasyprint():
    """Export HTML to PDF using weasyprint if available."""
    try:
        from weasyprint import HTML
    except ImportError:
        return False

    if not HTML_PATH.exists():
        print(f"Error: {HTML_PATH} not found")
        return False

    print("Generating PDF with weasyprint...")
    HTML(filename=str(HTML_PATH)).write_pdf(str(PDF_PATH))
    print(f"Saved: {PDF_PATH}")
    return True


def open_in_browser():
    """Open HTML in default browser for manual PDF export."""
    url = f"file://{HTML_PATH.resolve()}"
    try:
        webbrowser.open(url)
    except Exception:
        pass
    print("=" * 60)
    print("PDF 저장 방법:")
    print("1. 브라우저가 열리면 Cmd+P (Mac) 또는 Ctrl+P (Win) 누르기")
    print("2. '대상' 또는 'Printer'에서 'PDF로 저장' / 'Save as PDF' 선택")
    print("3. 저장 버튼 클릭")
    print("=" * 60)
    print(f"\nHTML 파일 경로: {HTML_PATH.resolve()}")
    print("(직접 열기: open report_helix_sjm_korean.html)")
    return True


def main():
    os.chdir(REPO)

    if "--browser" in sys.argv:
        open_in_browser()
        return

    if export_with_weasyprint():
        return

    print("weasyprint not installed. Opening HTML in browser instead.")
    open_in_browser()


if __name__ == "__main__":
    main()
