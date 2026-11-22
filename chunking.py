
from transformers import AutoTokenizer

from pdf_processor import extract_text_from_pdf


# Embedding model tokenizer
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Chunk configuration
CHUNK_SIZE = 200
CHUNK_OVERLAP = 40

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def chunk_pages(pages):
    """
    Split PDF pages into overlapping token chunks.

    Preserve the original PDF page number for each chunk.
    """

    chunks = []

    step_size = CHUNK_SIZE - CHUNK_OVERLAP

    for page_data in pages:

        text = page_data["text"].strip()
        page_number = page_data["page"]

        if not text:
            continue

        # Convert text into token IDs
        token_ids = tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=False
        )

        # Generate overlapping chunks
        for start in range(0, len(token_ids), step_size):

            chunk_ids = token_ids[start:start + CHUNK_SIZE]

            if not chunk_ids:
                continue

            # Convert token IDs back into text
            chunk_text = tokenizer.decode(
                chunk_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            ).strip()

            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "page": page_number
                })

            # Stop after processing the final chunk
            if start + CHUNK_SIZE >= len(token_ids):
                break

    return chunks


if __name__ == "__main__":

    pdf_path = input(
        "Enter the path to your PDF file: "
    ).strip().strip('"')

    try:
        pages = extract_text_from_pdf(pdf_path)

        if not pages:
            print("No readable text found in the PDF.")
            raise SystemExit(1)

        chunks = chunk_pages(pages)

        print(f"\nPages extracted: {len(pages)}")
        print(f"Chunks created: {len(chunks)}")

        # Check maximum chunk token length
        max_chunk_tokens = max(
            (
                len(tokenizer.encode(
                    chunk["text"],
                    add_special_tokens=True
                ))
                for chunk in chunks
            ),
            default=0
        )

        print(
            f"Maximum chunk token length: {max_chunk_tokens}"
        )

        print(
            f"Model maximum sequence length: "
            f"{tokenizer.model_max_length}"
        )

        # Display first three chunks
        for index, chunk in enumerate(chunks[:3], start=1):

            print(
                f"\n--- Chunk {index} | "
                f"Page {chunk['page']} ---"
            )

            print(chunk["text"][:500])

    except Exception as error:
        print(f"\nError during chunking: {error}")
