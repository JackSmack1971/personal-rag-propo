from pathlib import Path
from typing import List, Dict
from pypdf import PdfReader

def read_text_file(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore")

def read_pdf(path: Path) -> str:
    text = []
    reader = PdfReader(str(path))
    for page in reader.pages:
        try:
            text.append(page.extract_text() or "")
        except Exception:
            text.append("")
    return "\n".join(text)

def read_markdown(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore")

def to_paragraphs(text: str, min_len: int = 200) -> List[str]:
    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    # Merge very short paragraphs into neighbors
    merged: List[str] = []
    buf = ""
    for p in parts:
        if len(buf) < min_len:
            buf = (buf + "\n\n" + p).strip()
        else:
            merged.append(buf)
            buf = p
    if buf:
        merged.append(buf)
    return merged

def parse_any(path: Path) -> Dict:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        raw = read_pdf(path)
    elif suffix in (".txt",".md",".markdown"):
        raw = read_text_file(path) if suffix==".txt" else read_markdown(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")
    return {"text": raw, "paragraphs": to_paragraphs(raw)}
