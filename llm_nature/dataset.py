from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

@dataclass
class QAItem:
    q: str
    a: str

def load_paragraph(path: str | None = None) -> str:
    p = Path(path) if path else (DATA / "paragraph.txt")
    return p.read_text(encoding="utf-8")

def load_qa_corpus(path: str | None = None) -> list[QAItem]:
    p = Path(path) if path else (DATA / "qa_corpus.txt")
    text = p.read_text(encoding="utf-8")

    items: list[QAItem] = []
    q = None
    a = None

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.lower().startswith("q:"):
            q = line[2:].strip()
            a = None
            continue
        if line.lower().startswith("a:"):
            a = line[2:].strip()
            if q is not None:
                items.append(QAItem(q=q, a=a or ""))
                q = None
                a = None
            continue

    return items
