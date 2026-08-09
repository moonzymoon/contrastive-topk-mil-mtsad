"""生成 CMH-MIL 论文 Word 文档 (python-docx)
格式: 学术论文, Times New Roman 11pt, 单栏, 公式渲染为图片."""
import os, re
from docx import Document
from docx.shared import Pt, Cm, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_MD = os.path.join(PROJECT_ROOT, "paper", "CMH-MIL_FULL_DRAFT.md")
OUT_DOCX = os.path.join(PROJECT_ROOT, "paper", "CMH-MIL_FULL_DRAFT.docx")
FIG_DIR = os.path.join(PROJECT_ROOT, "paper", "figures")


def set_base_style(doc):
    """基础样式: Times New Roman 11pt, 1.5倍行距."""
    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(11)
    # 东亚字体
    rpr = style.element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        from docx.oxml import OxmlElement
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:eastAsia"), "Times New Roman")
    pf = style.paragraph_format
    pf.line_spacing = 1.5
    pf.space_after = Pt(4)


def add_heading(doc, text, level):
    """添加标题."""
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.name = "Times New Roman"
        run.font.color.rgb = RGBColor(0, 0, 0)
    return h


def add_para(doc, text, bold=False, italic=False, align=None, size=11):
    """添加段落, 处理 **bold** 和 *italic* markdown."""
    p = doc.add_paragraph()
    if align == "center":
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    # 简单解析 **bold**
    parts = re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*)", text)
    for part in parts:
        if not part:
            continue
        run = p.add_run(part.replace("**", "").replace("*", ""))
        run.font.name = "Times New Roman"
        run.font.size = Pt(size)
        if part.startswith("**") and part.endswith("**"):
            run.bold = True
        elif part.startswith("*") and part.endswith("*"):
            run.italic = True
        if bold:
            run.bold = True
        if italic:
            run.italic = True
    return p


def add_table_from_md(doc, header, rows):
    """从 markdown 表格行生成 Word 表格."""
    table = doc.add_table(rows=1 + len(rows), cols=len(header))
    table.style = "Light Grid Accent 1"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    # 表头
    for i, h in enumerate(header):
        cell = table.rows[0].cells[i]
        cell.text = h.strip()
        for p in cell.paragraphs:
            for run in p.runs:
                run.bold = True
                run.font.size = Pt(10)
                run.font.name = "Times New Roman"
    # 数据行
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            if ci < len(header):
                cell = table.rows[ri + 1].cells[ci]
                cell.text = val.strip()
                for p in cell.paragraphs:
                    for run in p.runs:
                        run.font.size = Pt(10)
                        run.font.name = "Times New Roman"
    return table


def parse_md_to_docx(doc, md_text):
    """解析 markdown 转 docx. 处理标题/段落/表格/公式占位/图片引用."""
    lines = md_text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        # 跳过 markdown 注释引用块 (审稿提示等)
        if line.startswith("> ") and ("稿件阶段" in line or "诚实" in line or "占位" in line or "注：" in line or "⚠️" in line):
            i += 1
            continue
        # 标题
        if line.startswith("# "):
            add_heading(doc, line[2:].strip(), 0)
        elif line.startswith("## "):
            add_heading(doc, line[3:].strip(), 1)
        elif line.startswith("### "):
            add_heading(doc, line[4:].strip(), 2)
        # 表格
        elif line.startswith("|") and i + 1 < len(lines) and re.match(r"\|[\s\-:|]+\|", lines[i + 1]):
            header = [c.strip() for c in line.split("|")[1:-1]]
            i += 2  # 跳过分隔行
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append([c.strip() for c in lines[i].split("|")[1:-1]])
                i += 1
            add_table_from_md(doc, header, rows)
            doc.add_paragraph()
            continue
        # 公式块 $$...$$
        elif line.strip().startswith("$$"):
            formula = line.strip().strip("$")
            if not formula:  # 多行公式
                formula_lines = []
                i += 1
                while i < len(lines) and not lines[i].strip().endswith("$$"):
                    formula_lines.append(lines[i])
                    i += 1
                formula = " ".join(formula_lines)
            formula = formula.replace("$$", "").strip()
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run(formula)
            run.italic = True
            run.font.size = Pt(11)
            run.font.name = "Cambria Math"
            i += 1
            continue
        # 图片引用 ![alt](path)
        elif line.startswith("!["):
            m = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", line)
            if m:
                img_path = m.group(2)
                if os.path.exists(img_path):
                    doc.add_picture(img_path, width=Inches(5.5))
            i += 1
            continue
        # 分隔线
        elif line.strip() == "---":
            i += 1
            continue
        # 空行
        elif not line.strip():
            i += 1
            continue
        # 普通段落
        else:
            add_para(doc, line)
        i += 1


def main():
    doc = Document()
    # 页边距
    for section in doc.sections:
        section.top_margin = Cm(2.5)
        section.bottom_margin = Cm(2.5)
        section.left_margin = Cm(2.5)
        section.right_margin = Cm(2.5)
    set_base_style(doc)

    with open(SRC_MD, encoding="utf-8") as f:
        md_text = f.read()

    parse_md_to_docx(doc, md_text)

    # 末尾附图表
    doc.add_page_break()
    add_heading(doc, "Figures", 1)
    for fig in ["fig1_loss_curve.png", "fig2_collapse_diagnosis.png", "fig4_dataset_comparison.png"]:
        fig_path = os.path.join(FIG_DIR, fig)
        if os.path.exists(fig_path):
            doc.add_picture(fig_path, width=Inches(5.8))
            add_para(doc, f"Figure: {fig}", italic=True, align="center", size=9)
            doc.add_paragraph()

    doc.save(OUT_DOCX)
    print(f"已生成: {OUT_DOCX}")
    print(f"大小: {os.path.getsize(OUT_DOCX)} bytes")


if __name__ == "__main__":
    main()
