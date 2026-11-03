"""Read the user-supplied 2014 handbook; headings are not executable edges."""
from __future__ import annotations

from pathlib import Path


# Printed start pages, checked against the contents of the supplied First 2014 PDF.
HANDBOOK_TREES = (
    ("Poor School Performance", 22),
    ("Behavioral Problems in a Child or Adolescent", 25),
    ("Speech Disturbance", 30),
    ("Distractibility", 35),
    ("Delusions", 38),
    ("Hallucinations", 44),
    ("Catatonic Symptoms", 49),
    ("Elevated or Expansive Mood", 52),
    ("Irritable Mood", 56),
    ("Depressed Mood", 61),
    ("Suicidal Ideation or Behavior", 67),
    ("Psychomotor Retardation", 72),
    ("Anxiety", 75),
    ("Panic Attacks", 80),
    ("Avoidance Behavior", 83),
    ("Trauma or Psychosocial Stressors Involved in the Etiology", 87),
    ("Somatic Complaints or Illness/Appearance Anxiety", 91),
    ("Appetite Changes or Unusual Eating Behavior", 94),
    ("Insomnia", 99),
    ("Hypersomnolence", 104),
    ("Sexual Dysfunction in a Female", 109),
    ("Sexual Dysfunction in a Male", 113),
    ("Aggressive Behavior", 116),
    ("Impulsivity or Impulse-Control Problems", 122),
    ("Self-Injury or Self-Mutilation", 126),
    ("Excessive Substance Use", 129),
    ("Memory Loss", 135),
    ("Cognitive Impairment", 139),
    ("Etiological Medical Conditions", 149),
)


def build_source_index(pdf_path: str | Path) -> dict:
    from pypdf import PdfReader

    path = Path(pdf_path).resolve()
    reader = PdfReader(path)
    copyright_page = reader.pages[4].extract_text()
    if "2014" not in copyright_page or "978-1-58562-462-1" not in copyright_page:
        raise ValueError("This importer requires the supplied First 2014 edition")
    entries = []
    for index, (title, start) in enumerate(HANDBOOK_TREES, 1):
        tree_id = f"2.{index}"
        pdf_page = start + 17
        text = reader.pages[pdf_page - 1].extract_text()
        compact = "".join(text.split()).casefold()
        if "".join(f"{tree_id} Decision Tree for {title}".split()).casefold() not in compact:
            raise ValueError(f"Heading/page mismatch: {tree_id}")
        next_start = HANDBOOK_TREES[index][1] if index < len(HANDBOOK_TREES) else 157
        entries.append({
            "tree_id": tree_id, "title": title,
            "print_start": start, "print_end": next_start - 1,
            "pdf_start": pdf_page, "pdf_end": next_start - 1 + 17,
            "node_review_status": "awaiting_node_and_edge_transcription",
            "screening_status": "awaiting_source_based_prompt_review",
            "refinement_status": "awaiting_development_and_change_log",
        })
    return {
        "id": "First2014-source-catalog-v1", "standard": "DSM-5",
        "review_status": "catalog_only_not_executable", "source_path": str(path),
        "document_id": "First2014", "title": "DSM-5 Handbook of Differential Diagnosis",
        "edition": "First edition, 2014", "isbn": "9781585624621",
        "total_pdf_pages": len(reader.pages), "pdf_page_offset": 17,
        "book_tree_count": len(entries), "paper_reported_tree_count": 28,
        "paper_tree_mapping_status": "unresolved_do_not_drop_a_tree_by_guessing",
        "trees": entries,
    }
