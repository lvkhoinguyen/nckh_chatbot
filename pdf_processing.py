from functools import lru_cache

@lru_cache(maxsize=100)
def process_pdf_with_cache(pdf_content):
    """Cache PDF processing results"""
    return extract_text_from_pdf(pdf_content)