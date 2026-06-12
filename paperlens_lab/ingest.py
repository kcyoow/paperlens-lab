from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree

import requests


ARXIV_ID_RE = re.compile(
    r"(?:(?:arxiv\.org/(?:abs|pdf)/)|arXiv:)?(?P<id>\d{4}\.\d{4,5}(?:v\d+)?|[a-z-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)",
    re.IGNORECASE,
)


@dataclass
class PaperSource:
    title: str
    authors: str
    source_label: str
    text: str
    pdf_url: str = ""
    warnings: tuple[str, ...] = ()


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def arxiv_id_from(value: str) -> Optional[str]:
    if not value:
        return None
    match = ARXIV_ID_RE.search(value.strip())
    return match.group("id") if match else None


def fetch_arxiv_source(value: str, timeout: int = 15) -> Optional[PaperSource]:
    arxiv_id = arxiv_id_from(value)
    if not arxiv_id:
        return None

    api_url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
    response = requests.get(api_url, timeout=timeout)
    response.raise_for_status()

    root = ElementTree.fromstring(response.text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    entry = root.find("atom:entry", ns)
    if entry is None:
        return None

    title = clean_text(entry.findtext("atom:title", default="", namespaces=ns))
    summary = clean_text(entry.findtext("atom:summary", default="", namespaces=ns))
    authors = [
        clean_text(author.findtext("atom:name", default="", namespaces=ns))
        for author in entry.findall("atom:author", ns)
    ]
    pdf_url = ""
    for link in entry.findall("atom:link", ns):
        if link.attrib.get("title") == "pdf":
            pdf_url = link.attrib.get("href", "")
            break

    return PaperSource(
        title=title or f"arXiv:{arxiv_id}",
        authors=", ".join(a for a in authors if a),
        source_label=f"arXiv:{arxiv_id}",
        text=summary,
        pdf_url=pdf_url,
    )


def extract_pdf_text(file_path: str | Path, max_pages: int = 16) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("PDF support requires pypdf. Install requirements.txt first.") from exc

    reader = PdfReader(str(file_path))
    pages = []
    for index, page in enumerate(reader.pages[:max_pages]):
        page_text = page.extract_text() or ""
        if page_text.strip():
            pages.append(f"[page {index + 1}]\n{page_text}")
    return clean_text("\n\n".join(pages))


def download_pdf_text(pdf_url: str, max_pages: int = 12, timeout: int = 30) -> str:
    response = requests.get(pdf_url, timeout=timeout)
    response.raise_for_status()
    with tempfile.NamedTemporaryFile(suffix=".pdf") as handle:
        handle.write(response.content)
        handle.flush()
        return extract_pdf_text(handle.name, max_pages=max_pages)


def build_source(
    uploaded_pdf: str | None,
    arxiv_or_url: str,
    pasted_text: str,
    max_pdf_pages: int,
) -> PaperSource:
    pasted = clean_text(pasted_text or "")
    arxiv_source = fetch_arxiv_source(arxiv_or_url) if arxiv_or_url.strip() else None

    title = arxiv_source.title if arxiv_source else "Untitled paper"
    authors = arxiv_source.authors if arxiv_source else ""
    source_label = arxiv_source.source_label if arxiv_source else "manual input"
    pdf_url = arxiv_source.pdf_url if arxiv_source else ""
    warnings: list[str] = []

    fragments = []
    if arxiv_source and arxiv_source.text:
        fragments.append(f"Title: {arxiv_source.title}\n\nAbstract: {arxiv_source.text}")

    if uploaded_pdf:
        fragments.append(extract_pdf_text(uploaded_pdf, max_pages=max_pdf_pages))
        source_label = Path(uploaded_pdf).name
    elif pdf_url and not pasted:
        try:
            fragments.append(download_pdf_text(pdf_url, max_pages=max_pdf_pages))
        except Exception as exc:
            warnings.append(f"pdf_download_or_parse_failed: {type(exc).__name__}")
            fragments.append(arxiv_source.text if arxiv_source else "")

    if pasted:
        fragments.append(pasted)

    text = clean_text("\n\n".join(fragment for fragment in fragments if fragment.strip()))
    if not text:
        raise ValueError("Add a PDF, arXiv URL/ID, or paper text.")

    return PaperSource(
        title=title,
        authors=authors,
        source_label=source_label,
        text=text,
        pdf_url=pdf_url,
        warnings=tuple(warnings),
    )
