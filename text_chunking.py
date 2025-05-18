import re

def split_text_into_chunks(text, min_tokens=200, max_tokens=500):
    """Split text into chunks based on paragraphs while respecting token limits"""
    # Split text into paragraphs based on empty lines
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        if len(current_chunk.split()) + len(paragraph.split()) <= max_tokens:
            current_chunk += paragraph + "\n"
        else:
            if len(current_chunk.split()) >= min_tokens:
                chunks.append(current_chunk.strip())
                current_chunk = paragraph + "\n"
            else:
                current_chunk += paragraph + "\n"

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks
