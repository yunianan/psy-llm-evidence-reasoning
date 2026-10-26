"""Read only the relevant First2014 pages; do not modify medical graph content."""
import json
from pathlib import Path

import pdfplumber

HERE = Path(__file__).resolve().parent
SOURCE = Path(r"C:\Users\ASUSA\Desktop\DSM-5原版鉴别诊断手册_First_2014.pdf")
OUTPUT = HERE / "source-review"


if __name__ == "__main__":
    OUTPUT.mkdir(exist_ok=True)
    with pdfplumber.open(SOURCE) as document:
        pages = {}
        for printed in [75, 76, 77, 78, 79, 91, 92, 93, 218, 219]:
            page = document.pages[printed + 16]
            pages[str(printed)] = page.extract_text() or ""
            print(json.dumps({"printed_page": printed, "pdf_page_one_based": printed + 17,
                              "text": pages[str(printed)]}, ensure_ascii=True))
            if printed in [77, 78, 92]:
                page.to_image(resolution=125).save(OUTPUT / f"first2014-p{printed}.png")
        (OUTPUT / "relevant-pages.json").write_text(json.dumps({"source": str(SOURCE), "pages": pages}, ensure_ascii=False, indent=2), encoding="utf-8")
