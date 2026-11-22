
import pymupdf
from pathlib import Path


def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file while preserving page numbers.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        list: A list of dictionaries containing page numbers and text.
    """

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("The provided file is not a PDF.")

    pages = []

    with pymupdf.open(pdf_path) as pdf:

        for page_number, page in enumerate(pdf, start=1):

            text = page.get_text("text").strip()

            if text:
                pages.append({
                    "page": page_number,
                    "text": text
                })

    return pages


if __name__ == "__main__":

    pdf_path = input(
        "Enter the path to your PDF file: "
    ).strip().strip('"')

    try:
        pages = extract_text_from_pdf(pdf_path)

        if not pages:
            print("No readable text found in the PDF.")
        else:
            print(f"\nTotal pages containing text: {len(pages)}")

            for page in pages[:2]:
                print(f"\n--- Page {page['page']} ---")
                print(page["text"][:1000])

    except Exception as error:
        print(f"\nError: {error}")
