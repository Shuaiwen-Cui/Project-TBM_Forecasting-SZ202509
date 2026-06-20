# -*- coding: utf-8 -*-
from pathlib import Path
from docx import Document

base = Path(r"E:\PROJ\Project-TBM_Forecasting-SZ202509\DELIVERY\Paper\submission")
work = base / "_revision_work"
for name in [
    "基于 Transformer 的盾构掘进关键参数多时间尺度实时预测研究-投稿.docx",
    "基于 Transformer 的盾构掘进关键参数多时间尺度实时预测研究-投稿-修改稿.docx",
]:
    doc = Document(str(base / name))
    out = work / (Path(name).stem + "_lines.txt")
    lines = [f"[{i}] {p.text}" for i, p in enumerate(doc.paragraphs) if p.text.strip()]
    out.write_text("\n".join(lines), encoding="utf-8")
    print(name, len(lines))
