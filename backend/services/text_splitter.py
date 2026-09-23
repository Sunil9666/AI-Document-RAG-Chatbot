import re


def split_text(text, chunk_size=800, chunk_overlap=100):

    # Clean excessive whitespace
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    # Split document into paragraphs
    paragraphs = text.split("\n\n")

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if not paragraph:
            continue

        # If adding this paragraph stays within the limit
        if len(current_chunk) + len(paragraph) <= chunk_size:

            current_chunk += paragraph + "\n\n"

        else:

            if current_chunk:
                chunks.append(current_chunk.strip())

            # Start a new chunk
            current_chunk = paragraph + "\n\n"

    # Add final chunk
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks