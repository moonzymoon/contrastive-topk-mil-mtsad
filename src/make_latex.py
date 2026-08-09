"""CMH-MIL markdown → Springer Nature LaTeX (sn-article 风格, 单栏投稿)
输出 sn-manuscript.tex, 可用 pdflatex/xelatex 编译.
若无 LaTeX 环境, 也可作为投稿源文件提交(编辑系统会编译)."""
import os, re

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_MD = os.path.join(PROJECT_ROOT, "paper", "CMH-MIL_FULL_DRAFT.md")
OUT_TEX = os.path.join(PROJECT_ROOT, "paper", "sn-manuscript.tex")
FIG_DIR = os.path.join(PROJECT_ROOT, "paper", "figures")


def esc(s):
    """LaTeX 转义."""
    s = s.replace("&", r"\&").replace("%", r"\%").replace("#", r"\#").replace("_", r"\_")
    s = s.replace("{", r"\{").replace("}", r"\}").replace("~", r"\textasciitilde{}")
    return s


def md_inline(s):
    """行内 markdown → LaTeX. $...$ 数学段不转义, 其余转义 + bold/italic."""
    # 按 $ 分割: 奇数索引段是数学(不转义), 偶数段转义
    parts = s.split("$")
    out = []
    for idx, part in enumerate(parts):
        if idx % 2 == 1:
            # 数学段, 原样保留(包回$)
            out.append("$" + part + "$")
        else:
            # 文本段: 转义 + bold/italic
            t = esc(part)
            t = re.sub(r"\*\*([^*]+)\*\*", r"\\textbf{\1}", t)
            t = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"\\textit{\1}", t)
            t = re.sub(r"`([^`]+)`", r"\\texttt{\1}", t)
            out.append(t)
    return "".join(out)


def convert(md_text):
    lines = md_text.split("\n")
    out = []
    i = 0
    tab_counter = [0]
    fig_counter = [0]

    def flush_table():
        pass

    while i < len(lines):
        line = lines[i].rstrip()
        # 跳过审稿注释块
        if line.startswith("> ") and any(k in line for k in ["稿件", "诚实", "占位", "注：", "⚠️", "本节", "初步"]):
            i += 1; continue
        # 标题
        if line.startswith("# "):
            title = line[2:].strip()
            if title == "Title":
                # 跳过 "# Title" 行 + 紧跟的 **标题文本** 行 + Abstract 前
                i += 1
                while i < len(lines):
                    nl = lines[i].strip()
                    if nl.startswith("# Abstract") or nl.startswith("## "):
                        break
                    i += 1
                continue
            # 跳过 Authors/Author Details(已在 preamble)
            if title in ("Authors", "Author Details"):
                while i < len(lines) and lines[i].strip() != "---":
                    i += 1
                i += 1; continue
            if title == "Abstract":
                i += 1
                while i < len(lines) and not lines[i].startswith("# "):
                    i += 1
                continue
            if title == "References":
                # references 由 references.md 末尾追加, 跳过本行(后续内容也会被处理, 但无所谓)
                i += 1; continue
            # 一级章节: 去掉 "I." "II." 等罗马前缀, 用 \section (LaTeX自动编号1,2,3)
            clean = re.sub(r"^[IVX]+\.\s*", "", title)
            # 去可能的 ** 包裹
            clean = clean.replace("**", "")
            out.append(f"\n\\section{{{md_inline(clean)}}}\n")
        elif line.startswith("## "):
            h = line[3:].strip()
            if h in ("Authors", "Author Details"):
                while i < len(lines) and lines[i].strip() != "---":
                    i += 1
                i += 1; continue
            if h == "Abstract":
                i += 1
                while i < len(lines) and not lines[i].startswith("# "):
                    i += 1
                continue
            # 二级: 去掉 "I-A." "II-B." 等前缀, 用 \subsection (自动1.1,1.2)
            clean = re.sub(r"^[IVX]+[-][A-Z0-9]+\.\s*", "", h)
            clean = clean.replace("**", "")
            out.append(f"\n\\subsection{{{md_inline(clean)}}}\n")
        elif line.startswith("### "):
            h3 = re.sub(r"^[IVX]+[-][A-Z0-9]+\.\s*", "", line[4:].strip())
            h3 = h3.replace("**", "")
            out.append(f"\n\\subsubsection{{{md_inline(h3)}}}\n")
        # 表格
        elif line.startswith("|") and i + 1 < len(lines) and re.match(r"\|[\s\-:|]+\|", lines[i + 1]):
            tab_counter[0] += 1
            header = [c.strip() for c in line.split("|")[1:-1]]
            ncol = len(header)
            i += 2
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append([c.strip() for c in lines[i].split("|")[1:-1]])
                i += 1
            # 从前一非空行提取 caption (markdown **Table Na. xxx.** 或 **Table N. xxx.**)
            cap_text = f"Table {tab_counter[0]}"
            for j in range(len(out) - 1, max(-1, len(out) - 8), -1):
                raw = out[j].strip()
                cm = re.search(r"\*{0,2}Table\s+\d+[a-z]?\.\s*(.+?)\*{0,2}\s*$", raw)
                if cm:
                    # 取标题(到第一个句号止, 去末尾句号和**)
                    title = cm.group(1).split(".")[0].strip().rstrip("*").strip()
                    cap_text = md_inline(title) if title else f"Table {tab_counter[0]}"
                    out[j] = ""
                    break
            colspec = "l" + "c" * (ncol - 1) if ncol > 1 else "l"
            # 宽表用 footnotesize 缩字号
            size_cmd = "\\footnotesize" if ncol > 6 else "\\small"
            tbl = [f"\n\\begin{{table}}[H]\\centering\\caption{{{cap_text}}}\\label{{tab:{tab_counter[0]}}}\n",
                   f"{size_cmd}\n", f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n",
                   " & ".join(md_inline(h) for h in header) + " \\\\\n\\midrule\n"]
            for r in rows:
                tbl.append(" & ".join(md_inline(c) for c in r[:ncol]) + " \\\\\n")
            tbl.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
            out.append("".join(tbl))
            continue
        # 公式 $$...$$
        elif line.strip().startswith("$$"):
            formula = line.strip().strip("$")
            tag = ""
            m = re.search(r"\\tag\{([^}]+)\}", formula)
            if m:
                tag = m.group(1); formula = re.sub(r"\\tag\{[^}]+\}", "", formula)
            if not formula:
                fl = []; i += 1
                while i < len(lines) and "$$" not in lines[i]:
                    fl.append(lines[i]); i += 1
                formula = " ".join(fl)
            eq = formula.replace("$$", "").strip()
            if tag:
                out.append(f"\n\\begin{{equation}}\n{eq}\n\\label{{eq:{tag}}}\n\\end{{equation}}\n")
            else:
                out.append(f"\n\\begin{{equation*}}\n{eq}\n\\end{{equation*}}\n")
            i += 1; continue
        # 图片
        elif line.startswith("!["):
            m = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", line)
            if m:
                fig_counter[0] += 1
                fn = os.path.basename(m.group(2))
                out.append(f"\n\\begin{{figure}}[H]\\centering\n\\includegraphics[width=0.9\\linewidth]{{figures/{fn}}}\n")
                out.append(f"\\caption{{{m.group(1) or 'Figure ' + str(fig_counter[0])}}}\n\\end{{figure}}\n")
            i += 1; continue
        elif line.strip() == "---":
            i += 1; continue
        elif not line.strip():
            i += 1; continue
        # 列表 -> itemize 环境(连续项包裹)
        elif re.match(r"^-\s", line):
            out.append("\\begin{itemize}\n")
            while i < len(lines) and re.match(r"^-\s", lines[i]):
                item_text = re.sub(r"^-\s", "", lines[i])
                out.append("\\item " + md_inline(item_text) + "\n")
                i += 1
            out.append("\\end{itemize}\n")
            continue
        elif re.match(r"^\d+\.\s", line):
            out.append("\\begin{enumerate}\n")
            while i < len(lines) and re.match(r"^\d+\.\s", lines[i]):
                item_text = re.sub(r"^\d+\.\s", "", lines[i])
                out.append("\\item " + md_inline(item_text) + "\n")
                i += 1
            out.append("\\end{enumerate}\n")
            continue
        else:
            out.append(md_inline(line) + "\n")
        i += 1

    return "\n".join(out)


def main():
    with open(SRC_MD, encoding="utf-8") as f:
        md_text = f.read()

    body = convert(md_text)

    # 提取 title/abstract/authors 从 md
    title_m = re.search(r"\*\*(From Bags.+?)\*\*", md_text)
    title = title_m.group(1) if title_m else "CMH-MIL"

    # LaTeX preamble: 标准 article 类 + Springer 投稿视觉风格(单栏/双倍行距/11pt)
    tex = r"""\documentclass[10pt,a4paper]{article}
\usepackage{mathptmx}
\usepackage{graphicx}
\usepackage{amsmath,amssymb}
\usepackage{float}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage{authblk}
\usepackage{parskip}

\title{""" + title.replace("*", "") + r"""}
\author[1]{Yao Zhang\thanks{Corresponding author: zhangyao@bdu.edu.cn; ORCID: 0009-0008-7636-8158}}
\author[1]{Baocai Li\thanks{ORCID: 0009-0000-6127-488X}}
\author[2]{Yichen Wang\thanks{ORCID: 0009-0007-3682-7463}}
\affil[1]{Department of Artificial Intelligence, Baoding University, Baoding, Hebei 071000, China}
\affil[2]{Department of Education, Baoding University, Baoding, Hebei 071000, China}
\date{}

\begin{document}
\maketitle

\begin{abstract}
""" + extract_abstract(md_text) + r"""
\end{abstract}

\noindent\textbf{Keywords:} Anomaly detection; Multiple-instance learning; Multi-hypersphere prototype; Conformal prediction; Weak supervision; Time series.

""" + body + r"""

\end{document}
"""
    # 后处理: 把参考文献裸列表包进 thebibliography 环境(Springer标准, 自动悬挂缩进)
    # 匹配从 \section{References} 到 \end{document}
    ref_pattern = re.compile(r"\\section\{References[^\n]*\}(.*?)\\end\{document\}", re.DOTALL)
    m = ref_pattern.search(tex)
    if m:
        before_refs = tex[:m.start()]
        refs_body = m.group(1)
        items = re.findall(r"\[(\d+)\]\s*(.+?)(?=\n\[\d+\]|\Z)", refs_body, re.DOTALL)
        bib_lines = ["\\begin{thebibliography}{99}\n"]
        for num, content in items:
            content = content.strip().replace("\n", " ")
            bib_lines.append(f"\\bibitem{{{num}}} {content}\n")
        bib_lines.append("\\end{thebibliography}\n")
        tex = before_refs + "".join(bib_lines) + "\\end{document}"

    with open(OUT_TEX, "w", encoding="utf-8") as f:
        f.write(tex)
    print(f"已生成 LaTeX: {OUT_TEX}")
    print(f"大小: {os.path.getsize(OUT_TEX)} bytes")


def extract_abstract(md):
    m = re.search(r"# Abstract\s*\n(.*?)(?=\n\*\*Keywords\*\*|\n# I\.|\n---)", md, re.DOTALL)
    if m:
        abs_text = m.group(1).strip()
        # 去markdown标记
        abs_text = re.sub(r"\*\*([^*]+)\*\*", r"\1", abs_text)
        abs_text = re.sub(r"\*([^*]+)\*", r"\1", abs_text)
        return abs_text
    return "(abstract)"


if __name__ == "__main__":
    main()
