# GenAI PDF Question Answering Using RAG

## Project Overview
A Generative AI-based application that allows users to upload PDF documents and ask questions about their content. It uses Retrieval-Augmented Generation (RAG) to retrieve relevant information and generate context-aware answers with source page references.

## Features
- Upload and process PDF documents.
- Extract text while preserving page numbers.
- Split text into overlapping chunks.
- Generate embeddings using Sentence Transformers.
- Retrieve relevant passages using FAISS.
- Generate answers using the Google Gemini API.
- Display answers with source page references through Streamlit.

## Technologies Used
- Python
- Google Gemini API
- Retrieval-Augmented Generation (RAG)
- Sentence Transformers
- FAISS
- PyMuPDF
- Streamlit

## Project Structure
```text
genai-pdf-qa-rag/
│
├── app.py
├── pdf_processor.py
├── chunking.py
├── embeddings.py
├── vector_store.py
├── qa_chain.py
├── test_retrieval.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── sample_documents/
```

## Installation and Setup

1. Clone the repository:
```bash
git clone https://github.com/ihskasyrahduahc13/genai-pdf-qa-rag.git
```

2. Navigate to the project directory:
```bash
cd genai-pdf-qa-rag
```

3. Create and activate a virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Create a `.env` file and add your Gemini API key:
```env
GEMINI_API_KEY=your_api_key_here
```

6. Run the application:
```bash
streamlit run app.py
```

## How It Works
1. Upload a PDF document.
2. Extract text using PyMuPDF.
3. Divide the extracted text into overlapping chunks.
4. Generate embeddings using Sentence Transformers.
5. Retrieve relevant passages using FAISS.
6. Generate answers using Google Gemini.
7. Display answers and source page references in Streamlit.

## Note
A valid Google Gemini API key is required to generate answers. Keep your API key private and do not upload your `.env` file to GitHub.