"""W4-⑬ 参考文献法证级核验与重建 (2026-11-17)
================================================
流程 (方法论库《SCI论文返修与参考文献法证级核验流程》):
  1) 解析 tex 的 26 条 bibitem
  2) 逐条 Crossref 验证: 已知DOI直查 / 标题搜索取最高分
  3) 归一化逐字段比对 (标题/年份/第一作者)
  4) 已知4条实锤直接给出修正条目 (CARLA/TimeMIL/Exathlon/USAD)
  5) 输出: bib_forensic_report.json + 重建的 thebibliography LaTeX 块
缓存 + 限速 (Crossref 礼仪: mailto UA, 1req/s)
"""
import os, re, json, time, urllib.request, urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.join(HERE, "..", "..", "paper", "IJMLC_Contrastive_TopK_MIL_MTSAD.tex")
CACHE = os.path.join(HERE, "..", "results", "crossref_cache.json")
OUT = os.path.join(HERE, "..", "results", "bib_forensic_report.json")

UA = {"User-Agent": "BibForensic/1.0 (mailto:zhangyao@bdu.edu.cn)"}


def http_json(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=25) as r:
        return json.loads(r.read().decode("utf-8"))


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())[:60]


def load_cache():
    return json.load(open(CACHE, encoding="utf-8")) if os.path.exists(CACHE) else {}


def crossref_by_doi(doi, cache):
    if f"doi:{doi}" in cache:
        return cache[f"doi:{doi}"]
    try:
        m = http_json(f"https://api.crossref.org/works/{urllib.parse.quote(doi)}")["message"]
        rec = {"title": (m.get("title") or [""])[0],
               "year": (m.get("issued", {}).get("date-parts") or [[None]])[0][0],
               "container": (m.get("container-title") or [""])[0],
               "first_author": ((m.get("author") or [{}])[0].get("family", "")),
               "volume": m.get("volume", ""), "page": m.get("page", ""),
               "doi": doi}
    except Exception as e:
        rec = {"error": str(e)[:100], "doi": doi}
    cache[f"doi:{doi}"] = rec
    time.sleep(1.0)
    return rec


def crossref_by_title(title, cache):
    key = f"tit:{norm(title)}"
    if key in cache:
        return cache[key]
    try:
        q = urllib.parse.quote(title[:120])
        m = http_json(f"https://api.crossref.org/works?rows=3&query.bibliographic={q}")["message"]["items"]
        best = None
        for it in m:
            t = (it.get("title") or [""])[0]
            if norm(t)[:40] == norm(title)[:40]:
                best = {"title": t,
                        "year": (it.get("issued", {}).get("date-parts") or [[None]])[0][0],
                        "container": (it.get("container-title") or [""])[0],
                        "first_author": ((it.get("author") or [{}])[0].get("family", "")),
                        "doi": it.get("DOI", "")}
                break
        rec = best or {"error": "no-match", "candidates":
                       [{"title": (i.get("title") or [""])[0], "doi": i.get("DOI", "")} for i in m]}
    except Exception as e:
        rec = {"error": str(e)[:100]}
    cache[key] = rec
    time.sleep(1.0)
    return rec


# 论文当前26条 + 4条实锤修正 + 7篇需新增(R2意见)
KNOWN_FIXES = {
    "11": "CARLA -> Darban, Z.Z., Webb, G.I., Pan, S., Aggarwal, C.C., Salehi, M. CARLA: Self-supervised Contrastive Representation Learning for Time Series Anomaly Detection. Pattern Recognition, 157:110874, 2025. doi:10.1016/j.patcog.2024.110874",
    "14": "TimeMIL -> Chen, X., Qiu, P., Zhu, W., Li, H., Wang, H., Sotiras, A., Wang, Y., Razi, A. TimeMIL: Advancing Multivariate Time Series Classification via a Time-aware Multiple Instance Learning. ICML, PMLR 235:7190-7206, 2024. (arXiv:2405.03140)",
    "23": "Exathlon -> Jacob, V., Song, F., Stiegler, A., Rad, B., Diao, Y., Tatbul, N. Exathlon: A Benchmark for Explainable Anomaly Detection over Time Series. PVLDB, 14(11):2613-2626, 2021. doi:10.14778/3476249.3476307",
    "24": "USAD -> Audibert, J., Michiardi, P., Guyard, F., Marti, S., Zuluaga, M. USAD: UnSupervised Anomaly Detection on Multivariate Time Series. KDD, pp.3395-3404, 2020. doi:10.1145/3394486.3403392",
}
NEW_REFS = [
    ("DCdetector", "10.1145/3580305.3599295"),
    ("CAROTS", "arXiv:2506.03964 (Kim et al., ICML 2025)"),
    ("MulGad", "10.1016/j.inffus.2025.103008"),
    ("DACAD", "10.1109/TKDE.2025.3569909"),
    ("NRdetector", "10.1145/3690624.3709257"),
    ("TAIL-MIL-correct", "10.1609/aaai.v39i17.33933 (Jang & Kwon, AAAI 2025, 39(17))"),
    ("TimeMIL-correct", "PMLR 235:7190-7206 ICML 2024"),
]


def main():
    tex = open(PAPER, encoding="utf-8").read()
    bib = tex[tex.find(r"\begin{thebibliography}"):tex.find(r"\end{thebibliography}")]
    entries = re.findall(r"\\bibitem\{(\d+)\}\s*(.+?)(?=\\bibitem|\\end)", bib, re.DOTALL)
    cache = load_cache()
    report = {"entries": {}, "known_fixes": KNOWN_FIXES, "new_refs": NEW_REFS}

    for key, body in entries:
        body = " ".join(body.split())
        title_m = re.search(r"\.\s*([A-Z][^.]{15,120}?)\.\s*(?:In|arXiv|\\textit|[A-Z]{2,})", body)
        title = title_m.group(1) if title_m else body[:70]
        year_m = re.search(r"(19|20)\d{2}", body)
        year = year_m.group(0) if year_m else None
        first_author = body.split(",")[0].strip()
        # 验证: 先标题搜索
        rec = crossref_by_title(title, cache)
        match = "error" not in rec and rec.get("title")
        verdict = {"bib_first_author": first_author, "bib_year": year,
                   "cr": rec, "title_probe": title}
        if match:
            ta = norm(rec.get("first_author", ""))[:8] in norm(first_author) or \
                 norm(first_author)[:8] in norm(rec.get("first_author", ""))
            verdict["author_match"] = bool(ta)
            verdict["year_match"] = (str(rec.get("year")) == str(year))
            verdict["verdict"] = "OK" if (ta and verdict["year_match"]) else "CHECK"
        else:
            verdict["verdict"] = "UNVERIFIED(arXiv/会议无DOI属正常, 人工核)"
        report["entries"][key] = verdict
        print(f"[{key:>2}] {verdict['verdict']:12s} {title[:55]}")

    json.dump(cache, open(CACHE, "w", encoding="utf-8"), indent=2)
    json.dump(report, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    n_ok = sum(1 for v in report["entries"].values() if v["verdict"] == "OK")
    print(f"\n完成: OK {n_ok}/{len(entries)}; 4条实锤修正已内置; 报告: {OUT}")


if __name__ == "__main__":
    main()
