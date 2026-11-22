
import os
import tempfile
import streamlit as st

from pdf_processor import extract_text_from_pdf
from chunking import chunk_pages
from vector_store import create_vector_store
from qa_chain import generate_answer


# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="PDF Question Answering",
    page_icon="📚",
    layout="wide"
)


# ---------------- SESSION STATE ----------------

if "index" not in st.session_state:
    st.session_state.index = None

if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "processed_file" not in st.session_state:
    st.session_state.processed_file = None


# ---------------- SIDEBAR ----------------

with st.sidebar:

    st.header("📄 Upload Document")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"]
    )

    if uploaded_file is not None:

        st.success(f"Selected: {uploaded_file.name}")

    process_button = st.button(
        "Process PDF",
        use_container_width=True,
        disabled=uploaded_file is None
    )

    if process_button and uploaded_file is not None:

        temp_path = None

        try:
            with st.spinner("Processing your PDF..."):

                # Save uploaded PDF temporarily
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as temp_file:

                    temp_file.write(uploaded_file.getvalue())
                    temp_path = temp_file.name

                # Extract text
                pages = extract_text_from_pdf(temp_path)

                if not pages:
                    st.error("No readable text found in the PDF.")

                else:
                    # Create chunks
                    chunks = chunk_pages(pages)

                    # Build FAISS vector store
                    index, stored_chunks = create_vector_store(chunks)

                    # Store results in session state
                    st.session_state.index = index
                    st.session_state.chunks = stored_chunks
                    st.session_state.messages = []
                    st.session_state.processed_file = uploaded_file.name

                    st.success(
                        f"PDF processed successfully! "
                        f"{len(pages)} pages extracted."
                    )

        except Exception as error:
            st.error(f"Error processing PDF: {error}")

        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    st.divider()

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ---------------- MAIN INTERFACE ----------------

st.title("📚 PDF Question Answering System")

st.write(
    "Upload a PDF document and ask questions about its content. "
    "The system uses Retrieval-Augmented Generation (RAG) "
    "to generate answers with source page references."
)

st.divider()


# ---------------- PDF STATUS ----------------

if st.session_state.index is None:

    st.info(
        "👉 Upload a PDF document from the sidebar "
        "and click Process PDF to get started."
    )

    st.subheader("How to use this application")

    st.markdown("""
    1. Upload your PDF document.
    2. Click **Process PDF**.
    3. Wait for text extraction and vector creation.
    4. Enter your question in the chat box.
    5. Receive an AI-generated answer with source pages.
    """)

else:

    st.success(
        f"Your PDF is ready! "
        f"You can start asking questions. "
        f"Document: {st.session_state.processed_file}"
    )


# ---------------- DISPLAY CHAT HISTORY ----------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message.get("source_pages"):
            st.caption(
                "Source pages: "
                + ", ".join(map(str, message["source_pages"]))
            )


# ---------------- CHAT INPUT ----------------

if st.session_state.index is not None:

    question = st.chat_input("Ask a question about your PDF...")

    if question:

        # Display user question
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        # Generate answer
        with st.chat_message("assistant"):

            with st.spinner("Generating answer..."):

                try:
                    answer, source_pages = generate_answer(
                        question,
                        st.session_state.index,
                        st.session_state.chunks
                    )

                    st.markdown(answer)

                    if source_pages:
                        st.caption(
                            "Source pages: "
                            + ", ".join(map(str, source_pages))
                        )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "source_pages": source_pages
                    })

                except Exception as error:

                    error_message = f"Error generating answer: {error}"

                    st.error(error_message)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_message
                    })
