from pypdf import PdfReader

def extract_text_from_pdf(pdf_path: str) -> list[str]:
    """Wczytuje PDF i dzieli tekst na fragmenty (chunki)."""
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"

    # Prosty podział na fragmenty po 800 znaków z nakładaniem 100 znaków
    chunk_size = 800
    overlap = 100
    chunks = []
    start = 0
    while start < len(full_text):
        end = start + chunk_size
        chunks.append(full_text[start:end])
        start += chunk_size - overlap

    return chunks
