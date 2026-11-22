
from pdf_processor import extract_text_from_pdf
from chunking import chunk_pages
from vector_store import create_vector_store, search_vector_store


PDF_PATH = (
    "sample_documents/"
    "9.-Nineteen-Eighty-Four-Author-George-Orwell.pdf"
)


def main():

    print("\nLoading PDF...")

    pages = extract_text_from_pdf(PDF_PATH)

    if not pages:
        print("No readable text found in the PDF.")
        return

    print(f"Pages extracted: {len(pages)}")

    print("\nCreating text chunks...")

    chunks = chunk_pages(pages)

    if not chunks:
        print("No chunks were created.")
        return

    print(f"Chunks created: {len(chunks)}")

    print("\nBuilding FAISS vector store...")

    index, stored_chunks = create_vector_store(chunks)

    print("\nVector store is ready!")
    print(f"Total vectors: {index.ntotal}")

    print("\nYou can now ask questions about the PDF.")
    print("Type 'exit' to stop.")

    while True:

        question = input("\nYour question: ").strip()

        if question.lower() == "exit":
            print("\nExiting retrieval system.")
            break

        if not question:
            continue

        results = search_vector_store(
            question,
            index,
            stored_chunks,
            top_k=5
        )

        if not results:
            print("No relevant passages found.")
            continue

        print("\nRetrieved passages:")

        for rank, result in enumerate(results, start=1):

            print(
                f"\n--- Result {rank} | "
                f"Page {result['page']} | "
                f"Score: {result['score']:.4f} ---"
            )

            print(result["text"][:700])


if __name__ == "__main__":
    main()
