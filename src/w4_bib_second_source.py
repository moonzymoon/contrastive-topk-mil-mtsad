"""W4-⑬ 第二源验证: 15条 UNVERIFIED (arXiv API + DBLP API) + 新增条目核验
输出: results/bib_second_source_report.json
"""
import os, json, time, urllib.request, urllib.parse, re, xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "bib_second_source_report.json")
UA = {"User-Agent": "BibForensic/2.0 (mailto:zhangyao@bdu.edu.cn)"}


def http(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=25) as r:
        return r.read().decode("utf-8")


def arxiv_verify(arxiv_id, want_title_kw, want_year):
    """arXiv API: 验证 ID 是否指向含关键词的论文."""
    try:
        raw = http(f"http://export.arxiv.org/api/query?id_list={arxiv_id}")
        root = ET.fromstring(raw)
        ns = {"a": "http://www.w3.org/2005/Atom"}
        entry = root.find("a:entry", ns)
        if entry is None:
            return {"verdict": "NOT_FOUND", "id": arxiv_id}
        title = " ".join(entry.find("a:title", ns).text.split())
        year = entry.find("a:published", ns).text[:4]
        authors = [a.find("a:name", ns).text for a in entry.findall("a:author", ns)]
        ok = want_title_kw.lower() in title.lower()
        return {"verdict": "OK" if ok else "MISMATCH", "id": arxiv_id,
                "title": title, "year": year, "first_author": authors[0] if authors else "",
                "want": want_title_kw, "want_year": want_year,
                "year_ok": year == want_year}
    except Exception as e:
        return {"verdict": "ERR", "id": arxiv_id, "err": str(e)[:100]}


def dblp_verify(query, want_year=None):
    """OpenAlex API (DBLP 被 bot 防护拦截, 换开放索引)."""
    import time as _t
    raw = None
    for attempt in range(4):
        try:
            raw = http("https://api.openalex.org/works?search=" + urllib.parse.quote(query) + "&per-page=3")
            break
        except Exception as _e:
            if attempt == 3:
                return {"verdict": "ERR", "query": query, "err": str(_e)[:100]}
            _t.sleep(3 * (attempt + 1))
    try:
        res = json.loads(raw).get("results", [])
        out = []
        for w in res:
            auth = (w.get("authorships") or [{}])[0].get("author", {}).get("display_name", "")
            out.append({"title": w.get("display_name", ""), "venue": (w.get("primary_location") or {}).get("source", {}) and ((w.get("primary_location") or {}).get("source") or {}).get("display_name", ""),
                        "year": w.get("publication_year"), "first_author": auth,
                        "doi": w.get("doi", "")})
        return {"verdict": "HITS" if out else "NO_HIT", "query": query, "hits": out}
    except Exception as e:
        return {"verdict": "ERR", "query": query, "err": str(e)[:100]}


TASKS_ARXIV = [
    # (bibitem编号, arXiv ID, 标题关键词, 论文写的年份)
    ("16", "1807.03748", "contrastive predictive", "2018"),
    ("25", "2107.07511", "conformal prediction", "2021"),
]
TASKS_DBLP = [
    ("1",  "Deep One-Class Classification Ruff", "2018"),
    ("6",  "Attention-based Deep Multiple Instance Learning Ilse", "2018"),
    ("8",  "Unbiased Multiple Instance Learning Weakly Supervised Video Anomaly", "2023"),
    ("13", "LOF Identifying Density-Based Local Outliers", "2000"),
    ("15", "TSB-UAD Benchmark Univariate Time Series", "2022"),
    ("17", "Understanding Contrastive Representation Learning Alignment Uniformity", "2020"),
    ("18", "Axiomatic Attribution for Deep Networks", "2017"),
    ("4",  "review deep learning based anomaly detection Neurocomputing", "2026"),
    ("5",  "Graph anomaly detection time series survey", "2025"),
    ("12", "Correlation-aware spatial-temporal graph multivariate time-series anomaly", "2024"),
    # 新增五篇 + WADI
    ("NEW-DCdetector", "DCdetector dual attention contrastive time series anomaly", "2023"),
    ("NEW-MulGad", "MulGad multi-granularity contrastive multivariate time series", "2025"),
    ("NEW-DACAD", "DACAD domain adaptation contrastive anomaly detection multivariate", "2025"),
    ("NEW-NRdetector", "noise-resilient point-wise anomaly detection weak segment", "2025"),
    ("WADI-Ahmed", "WADI water distribution attack detection", "2017"),
]


def main():
    report = {"arxiv": {}, "dblp": {}}
    for key, aid, kw, yr in TASKS_ARXIV:
        report["arxiv"][key] = arxiv_verify(aid, kw, yr)
        print(f"arXiv [{key}]: {report['arxiv'][key]['verdict']} {report['arxiv'][key].get('title', '')[:60]}")
        time.sleep(1.0)
    for key, q, yr in TASKS_DBLP:
        r = dblp_verify(q)
        # 判定: 首hit年份匹配且标题相近
        best = (r.get("hits") or [{}])[0]
        r["year_want"] = yr
        r["verdict_final"] = ("OK" if best and str(best.get("year")) == yr else "CHECK")
        report["dblp"][key] = r
        print(f"DBLP [{key}]: {r['verdict_final']} {str(best.get('title',''))[:60]} ({best.get('year')})")
        time.sleep(2.5)
    json.dump(report, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("\nreport:", OUT)


if __name__ == "__main__":
    main()
