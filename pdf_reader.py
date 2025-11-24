# src/pdf_reader.py
from pypdf import PdfReader

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file and return as a single string.
    """
    reader = PdfReader(pdf_path)
    text_parts = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text)
    return "\n".join(text_parts)
