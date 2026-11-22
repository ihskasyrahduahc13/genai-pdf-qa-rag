
import os

from dotenv import load_dotenv
from google import genai

from pdf_processor import extract_text_from_pdf
from chunking import chunk_pages
from vector_store import create_vector_store, search_vector_store


# Load API key from .env
load_dotenv()

if not os.getenv("GEMINI_API_KEY"):
    raise ValueError(
        "GEMINI_API_KEY not found. Check your .env file."
    )

# Initialize Gemini client
client = genai.Client()

MODEL_NAME = "gemini-3.8-flash"


def generate_answer(question, index, chunks, top_k=8):
    """
    Retrieve relevant PDF passages and generate
    an answer using Gemini.
    """

    # Step 1: Retrieve relevant passages
    results = search_vector_store(
        question,
        index,
        chunks,
        top_k=top_k
    )

    if not results:
        return "I could not find relevant information in the PDF.", []

    # Step 2: Prepare context with page numbers
    context = ""

    for rank, result in enumerate(results, start=1):
        context += (
            f"\nPassage {rank} "
            f"(Page {result['page']}):\n"
            f"{result['text']}\n"
        )

    # Step 3: Create the prompt
    prompt = f"""
You are a PDF question-answering assistant.

Answer the user's question using ONLY the
information provided in the context below.

Instructions:
1. Give a clear and direct answer.
2. Do not invent facts or use outside knowledge.
3. If the context does not contain the answer,
   clearly say that the information was not found.
4. Use simple, understandable language.

Context:
{context}

Question:
{question}

Answer:
"""

    # Step 4: Generate answer using Gemini
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    answer = response.text or "No answer was generated."

    # Step 5: Collect source page numbers
    source_pages = sorted(
        set(result["page"] for result in results)
    )

    return answer, source_pages


def main():

    pdf_path = input(
        "Enter the path to your PDF file: "
    ).strip().strip('"')

    try:
        print("\nLoading PDF...")
        pages = extract_text_from_pdf(pdf_path)

        if not pages:
            print("No readable text found in the PDF.")
            return

        print(f"Pages extracted: {len(pages)}")

        print("\nCreating text chunks...")
        chunks = chunk_pages(pages)

        print(f"Chunks created: {len(chunks)}")

        print("\nBuilding FAISS vector store...")
        index, stored_chunks = create_vector_store(chunks)

        print("\nRAG pipeline is ready!")
        print("You can now ask questions about the PDF.")

        while True:

            question = input(
                "\nYour question (or type 'exit'): "
            ).strip()

            if question.lower() == "exit":
                print("Exiting PDF question-answering system.")
                break

            if not question:
                continue

            print("\nGenerating answer...")

            answer, source_pages = generate_answer(
                question,
                index,
                stored_chunks
            )

            print("\nAnswer:")
            print(answer)

            if source_pages:
                print("\nSource pages:", source_pages)

    except Exception as error:
        print(f"\nError: {error}")


if __name__ == "__main__":
    main()
