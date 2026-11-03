"""Retrieve project-24 evidence from the supplied PDFs, without inferring arrows.

Run with the bundled PDF Python. The runtime tree executor never imports pypdf.
Full continued sections are retained, not only keyword-containing lines. Chapter
3 columns are extracted separately, to avoid pairing unrelated left/right cells.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "knowledge/paper_dtree/first2014_project24"
HANDBOOK = Path("C:/Users/ASUSA/Desktop/DSM-5原版鉴别诊断手册_First_2014.pdf")
PAPER = Path("C:/Users/ASUSA/Desktop/参考/将专家知识整合进大型语言模型，提升精神病学推理和诊断的表现.pdf")

# Primary tables, including every continued page. PTSD and ASD share one table.
PRIMARY_TABLES = [
    ("3.1.4", 168, 171, ["D001", "D002", "D003"]),
    ("3.2.1", 175, 176, ["D005"]), ("3.2.2", 177, 177, ["D006", "D007"]),
    ("3.2.3", 178, 179, ["D004"]), ("3.3.1", 182, 184, ["D008", "D010"]),
    ("3.3.2", 185, 187, ["D009"]), ("3.4.1", 189, 191, ["D013", "D015"]),
    ("3.4.2", 192, 193, ["D014"]), ("3.5.3", 202, 203, ["D012"]),
    ("3.5.7", 212, 214, ["D011"]), ("3.6.1", 215, 217, ["D016"]),
    ("3.6.2", 218, 219, ["D017"]), ("3.7.1", 225, 226, ["D018", "D019"]),
    ("3.7.2", 227, 228, ["D020"]), ("3.10.2", 246, 248, ["D021"]),
    ("3.10.3", 249, 250, ["D022"]), ("3.10.4", 251, 251, ["D023"]),
    ("3.11.1", 252, 254, ["Insomnia"]),
]

FAMILY_ALIASES = {
    "adhd": ["Attention-Deficit/Hyperactivity Disorder", "ADHD", "Combined presentation", "Predominantly inattentive presentation", "Predominantly hyperactive/impulsive presentation"],
    "delusional": ["Delusional Disorder"], "schizophrenia": ["Schizophrenia"],
    "schizoaffective": ["Schizoaffective Disorder"], "bipolar1": ["Bipolar I Disorder", "Bipolar I or", "Bipolar I and"],
    "bipolar2": ["Bipolar II Disorder", "Bipolar I or Bipolar II", "Bipolar I and Bipolar II"],
    "gad": ["Generalized Anxiety Disorder"], "phobia": ["Specific Phobia", "Specific Phobias"],
    "mdd": ["Major Depressive Disorder", "Major Depressive Episode"],
    "pdd": ["Persistent Depressive Disorder", "Dysthymia"],
    "ocd": ["Obsessive-Compulsive Disorder", "OCD"], "bdd": ["Body Dysmorphic Disorder"],
    "ptsd": ["Posttraumatic Stress Disorder", "PTSD", "PSTD"],
    "asd": ["Acute Stress Disorder"], "adjustment": ["Adjustment Disorder"],
    "anorexia": ["Anorexia Nervosa"], "bulimia": ["Bulimia Nervosa"],
    "binge": ["Binge-Eating Disorder"], "insomnia": ["Insomnia Disorder"],
}


def normalize(text):
    text = text.replace("\u00ad", "").replace("\u0002", "").replace("\u0003", "")
    # Remove PDF line-wrap hyphens, preserving lexical hyphens on the same line.
    text = re.sub(r"(?<=\w)[-‐]\s*\n\s*(?=\w)", "", text)
    return re.sub(r"\s+", " ", text).strip()


def search_form(text):
    return re.sub(r"[^a-z0-9]", "", normalize(text).lower())


def read_pages(path):
    from pypdf import PdfReader
    return [{"pdf_page": i + 1, "text": p.extract_text() or ""}
            for i, p in enumerate(PdfReader(path).pages)]


def extract(handbook=HANDBOOK, paper=PAPER, output=DATA):
    import pdfplumber
    pages = read_pages(handbook)
    paper_pages = read_pages(paper)
    source_catalog = json.loads((ROOT / "knowledge/paper_dtree/first2014_source_catalog.json").read_text(encoding="utf-8"))
    normalized_aliases = {k: [search_form(a) for a in aliases] for k, aliases in FAMILY_ALIASES.items()}
    hits = []
    for p in pages:
        compact = search_form(p["text"])
        families = [k for k, aliases in normalized_aliases.items() if any(a in compact for a in aliases)]
        if families:
            n = p["pdf_page"]
            hits.append({"pdf_page": n, "print_page": n - 17 if n >= 18 else None,
                         "families": families,
                         "kind": "front_matter" if n < 18 else "index" if n >= 336 else "classification" if n >= 308 else "clinical"})
    hit_by_page = {p["pdf_page"]: p for p in hits}
    sections = [{"id": "chapter1", "kind": "general_differential_method", "print_start": 1, "print_end": 20}]
    # All 29 source trees are searched, including those not in the previous 18.
    for t in source_catalog["trees"]:
        if any(n in hit_by_page for n in range(t["pdf_start"], t["pdf_end"] + 1)):
            sections.append({"id": t["tree_id"], "title": t["title"], "kind": "symptom_tree_and_commentary",
                             "print_start": t["print_start"], "print_end": t["print_end"]})
    for tid, start, end, labels in PRIMARY_TABLES:
        sections.append({"id": tid, "kind": "primary_differential_table", "print_start": start,
                         "print_end": end, "target_ids": labels})
    # Cross-references in other tables are evidence too; store complete pages.
    selected = {n for section in sections for n in range(section["print_start"] + 17, section["print_end"] + 18)}
    selected.update(p["pdf_page"] for p in hits)
    full_pages = [{**p, "print_page": p["pdf_page"] - 17 if p["pdf_page"] >= 18 else None}
                  for p in pages if p["pdf_page"] in selected]
    table_columns, table_rows = [], []
    with pdfplumber.open(handbook) as pdf:
        for tid, start, end, _ in PRIMARY_TABLES:
            first = pdf.pages[start + 16]
            contrast = [w for w in first.extract_words() if w["text"] == "In" and w["top"] > 80 and w["x0"] > 150]
            if not contrast:
                raise ValueError(f"Missing native right-column heading in {tid}")
            gutter = round(contrast[0]["x0"] - 1, 1)
            for printed in range(start, end + 1):
                page = pdf.pages[printed + 16]
                # Read the gutter from each table's native right-column heading.
                # Bipolar tables are asymmetric (right starts x=202.7 rather
                # than x=230.2); a fixed midpoint would corrupt their row labels.
                table_columns.append({"table_id": tid, "print_page": printed, "pdf_page": printed + 17,
                                      "column_gutter": gutter,
                                      "left_column": page.crop((36, 65, gutter, 678)).extract_text(),
                                      "right_column": page.crop((gutter, 65, 422, 678)).extract_text()})
                lines = {}
                for word in page.extract_words(x_tolerance=2, y_tolerance=2):
                    if word["x0"] < gutter and 65 < word["top"] < 677:
                        lines.setdefault(round(word["top"], 1), []).append(word)
                # Native hanging indent: first row line x=38.8/38.9,
                # continuation x=44.8/44.9; headings/footnotes x=37.3/37.4.
                footnote_tops = [c["top"] for c in page.chars if c["top"] > 100 and c["size"] < 9 and 37 < c["x0"] < 41]
                table_end = min(footnote_tops) - 1 if footnote_tops else 677
                starts = sorted(y for y, words in lines.items() if 38.3 < min(w["x0"] for w in words) < 40 and y < table_end)
                for i, top in enumerate(starts):
                    bottom = starts[i + 1] - 1 if i + 1 < len(starts) else table_end
                    if bottom <= top:
                        raise ValueError(f"Invalid native row bounds {tid}/{printed}/{top}")
                    left = page.crop((37, top - 0.5, gutter, bottom)).extract_text() or ""
                    right = page.crop((gutter, top - 0.5, 422, bottom)).extract_text() or ""
                    table_rows.append({"id": f"{tid}.p{printed}.r{i+1}", "table_id": tid,
                                       "print_page": printed, "pdf_page": printed + 17,
                                       "kind": "definition" if printed == start and i == 0 else "differential",
                                       "left": left, "right": right,
                                       "bbox": [37, round(top - 0.5, 1), 422, round(bottom, 1)],
                                       "extraction_status": "native_hanging_indent_rows; full_page_and_columns_retained_for_cross_check"})
    result = {
        "id": "First2014-project24-source-evidence-v1", "standard": "DSM-5", "date": "2026-11-02",
        "documents": [{"id": "First2014", "path": str(handbook), "pdf_pages": len(pages), "printed_page_offset": 17},
                      {"id": "Sarma2025", "path": str(paper), "pdf_pages": len(paper_pages)}],
        "method": "All 339 handbook pages searched with full names, abbreviations and line-wrap normalization; complete source-tree sections and primary continued tables retained; other matching pages retained in full. Retrieval is not clinical validation or verbatim author supplement recovery.",
        "aliases": FAMILY_ALIASES, "page_hits": hits, "sections": sections,
        "pages": full_pages, "primary_table_columns": table_columns, "primary_table_rows": table_rows,
        "paper_pages": paper_pages,
        "layout_review": {"rendered_pdf_pages": [185, 194, 199, 269], "finding": "Two aligned text columns; pypdf interleaves columns. Per-table native gutters retained (bipolar tables asymmetric); small-font footnotes kept in full-page extracts outside row segmentation. No graph edges inferred from text order."},
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "source_evidence.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"handbook_pages_searched": len(pages), "matching_pages": len(hits),
                      "retained_pages": len(full_pages), "symptom_tree_sections": sum(s["kind"] == "symptom_tree_and_commentary" for s in sections),
                      "primary_tables": len(PRIMARY_TABLES), "primary_table_pages": len(table_columns)}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handbook", type=Path, default=HANDBOOK)
    parser.add_argument("--paper", type=Path, default=PAPER)
    parser.add_argument("--output", type=Path, default=DATA)
    a = parser.parse_args()
    extract(a.handbook, a.paper, a.output)
