"""生成 CMH-MIL 论文 PDF (fpdf2)
英文论文, 含表格/公式文本/图片. 用 Unicode 字体支持中文姓名."""
import os, re
from fpdf import FPDF

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_MD = os.path.join(PROJECT_ROOT, "paper", "CMH-MIL_FULL_DRAFT.md")
OUT_PDF = os.path.join(PROJECT_ROOT, "paper", "CMH-MIL_FULL_DRAFT.pdf")
FIG_DIR = os.path.join(PROJECT_ROOT, "paper", "figures")
# Windows 中文字体(支持中文姓名)
CN_FONT = "C:\\Windows\\Fonts\\simsun.ttc"


class PaperPDF(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Sun", "I", 8)
            self.set_text_color(120, 120, 120)
            self.cell(0, 8, "CMH-MIL: Conformal Multi-Hypersphere Learning for Weakly-Supervised MTS Anomaly Detection",
                      align="C", new_x="LMARGIN", new_y="NEXT")
            self.set_text_color(0, 0, 0)
            self.ln(2)

    def footer(self):
        self.set_y(-15)
        self.set_font("Sun", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")
        self.set_text_color(0, 0, 0)


def clean_md(text):
    """去掉审稿提示引用块(> 开头含特定关键词)."""
    out = []
    skip_block = False
    for line in text.split("\n"):
        if line.startswith("> ") and any(k in line for k in ["稿件阶段", "诚实", "占位", "注：", "⚠️", "本节结果均", "初步"]):
            skip_block = True
            continue
        if skip_block:
            if line.startswith("> "):
                continue
            skip_block = False
        if line.strip() == "---":
            continue
        out.append(line)
    return "\n".join(out)


def render_inline(pdf, text):
    """渲染含 **bold**/*italic* 的行内文本."""
    parts = re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)", text)
    for part in parts:
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            pdf.set_font("Sun", "B", 10)
            pdf.write(5, part[2:-2])
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            pdf.set_font("Sun", "I", 10)
            pdf.write(5, part[1:-1])
        elif part.startswith("`") and part.endswith("`"):
            pdf.set_font("Sun", "", 9)
            pdf.write(5, part[1:-1])
        else:
            pdf.set_font("Sun", "", 10)
            pdf.write(5, part)


def add_table(pdf, header, rows):
    """渲染表格."""
    n = len(header)
    col_w = (pdf.w - 2 * pdf.l_margin) / n
    # 表头
    pdf.set_font("Sun", "B", 8.5)
    pdf.set_fill_color(220, 230, 240)
    for h in header:
        pdf.cell(col_w, 6, h.strip()[:20], border=1, fill=True, align="C")
    pdf.ln()
    # 数据
    pdf.set_font("Sun", "", 8.5)
    for row in rows:
        for ci, val in enumerate(row):
            if ci < n:
                fill = (245, 245, 245) if any(b in val for b in ["CMH-MIL", "**0.223", "**0.469"]) else None
                if fill:
                    pdf.set_fill_color(*fill)
                    pdf.set_font("Sun", "B", 8.5)
                else:
                    pdf.set_font("Sun", "", 8.5)
                pdf.cell(col_w, 5.5, val.strip().replace("**", "")[:22], border=1,
                         fill=bool(fill), align="C")
        pdf.ln()
    pdf.ln(2)


def md_to_pdf(md_text, out_path):
    pdf = PaperPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=20)
    # 注册中文 TTF, 用多个别名覆盖脚本内所有字体引用(支持中文姓名)
    for alias in ["Sun", "Times", "Helvetica", "Courier"]:
        try:
            pdf.add_font(alias, "", CN_FONT)
            pdf.add_font(alias, "B", CN_FONT)
            pdf.add_font(alias, "I", CN_FONT)
            pdf.add_font(alias, "BI", CN_FONT)
        except Exception:
            pass
    pdf.add_page()
    pdf.set_margins(20, 20, 20)

    lines = md_text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        # 标题
        if line.startswith("# "):
            pdf.ln(3)
            pdf.set_font("Sun", "B", 15)
            pdf.multi_cell(0, 7, line[2:].strip())
            pdf.ln(2)
        elif line.startswith("## "):
            pdf.ln(2)
            pdf.set_font("Sun", "B", 12)
            pdf.multi_cell(0, 6, line[3:].strip())
            pdf.ln(1)
        elif line.startswith("### "):
            pdf.ln(1)
            pdf.set_font("Sun", "B", 11)
            pdf.multi_cell(0, 5.5, line[4:].strip())
        # 表格
        elif line.startswith("|") and i + 1 < len(lines) and re.match(r"\|[\s\-:|]+\|", lines[i + 1]):
            header = [c.strip() for c in line.split("|")[1:-1]]
            i += 2
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append([c.strip() for c in lines[i].split("|")[1:-1]])
                i += 1
            add_table(pdf, header, rows)
            continue
        # 公式 $$...$$
        elif line.strip().startswith("$$"):
            formula = line.strip().strip("$")
            if not formula:
                fl = []
                i += 1
                while i < len(lines) and "$$" not in lines[i]:
                    fl.append(lines[i])
                    i += 1
                formula = " ".join(fl)
            pdf.ln(1)
            pdf.set_font("Sun", "I", 10)
            pdf.multi_cell(0, 5, "    " + formula.replace("$$", "").strip(), align="C")
            pdf.ln(1)
            i += 1
            continue
        # 图片
        elif line.startswith("!["):
            m = re.match(r"!\[[^\]]*\]\(([^)]+)\)", line)
            if m and os.path.exists(m.group(1)):
                pdf.image(m.group(1), w=160)
                pdf.ln(2)
            i += 1
            continue
        # 列表项
        elif re.match(r"^\d+\.\s|^\-\s", line):
            render_inline(pdf, "  " + re.sub(r"^\d+\.\s|^\-\s", "", line))
            pdf.ln(4.5)
        # 空行
        elif not line.strip():
            pdf.ln(2)
        # 普通段落
        else:
            render_inline(pdf, line)
            pdf.ln(4.5)
        i += 1

    # 末尾图表
    pdf.add_page()
    pdf.set_font("Sun", "B", 14)
    pdf.cell(0, 8, "Figures", new_y="NEXT")
    pdf.ln(3)
    for fig in ["fig1_loss_curve.png", "fig2_collapse_diagnosis.png", "fig4_dataset_comparison.png"]:
        fig_path = os.path.join(FIG_DIR, fig)
        if os.path.exists(fig_path):
            pdf.image(fig_path, w=170)
            pdf.set_font("Sun", "I", 8)
            pdf.cell(0, 5, fig, align="C", new_y="NEXT")
            pdf.ln(3)

    pdf.output(out_path)
    print(f"已生成: {out_path}")
    print(f"大小: {os.path.getsize(out_path)} bytes")


if __name__ == "__main__":
    with open(SRC_MD, encoding="utf-8") as f:
        md_text = f.read()
    md_to_pdf(clean_md(md_text), OUT_PDF)
