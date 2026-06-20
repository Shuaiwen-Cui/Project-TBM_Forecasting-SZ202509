# -*- coding: utf-8 -*-
from pathlib import Path
from docx import Document
from docx.shared import RGBColor

path = Path(r"E:\PROJ\Project-TBM_Forecasting-SZ202509\DELIVERY\Paper\submission\基于 Transformer 的盾构掘进关键参数多时间尺度实时预测研究-投稿-修改稿_新.docx")
doc = Document(str(path))

red_runs, black_runs = 0, 0
red_chars, black_chars = 0, 0
for p in doc.paragraphs:
    for r in p.runs:
        if not r.text.strip():
            continue
        if r.font.color.rgb == RGBColor(0xFF, 0x00, 0x00):
            red_runs += 1
            red_chars += len(r.text)
        else:
            black_runs += 1
            black_chars += len(r.text)

print(f"red runs: {red_runs}, chars: {red_chars}")
print(f"black runs: {black_runs}, chars: {black_chars}")
print(f"red ratio: {red_chars/(red_chars+black_chars)*100:.1f}%")
