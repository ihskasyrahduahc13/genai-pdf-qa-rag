
import faiss
import numpy as np

from embeddings import generate_embeddings


def create_vector_store(chunks):
    """
    Generate embeddings for PDF chunks and store them
    in a FAISS vector index.
    """

    if not chunks:
        raise ValueError("No chunks provided.")

    texts = [chunk["text"] for chunk in chunks]

    print("Generating embeddings for PDF chunks...")

    embeddings = generate_embeddings(
        texts,
        batch_size=32
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    # Create FAISS index using inner product.
    # Since embeddings are normalized, this is cosine similarity.
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    print("\nVector store created successfully!")
    print("Total vectors:", index.ntotal)
    print("Embedding dimensions:", dimension)

    return index, chunks


def search_vector_store(question, index, chunks, top_k=5):
    """
    Search FAISS for the most relevant PDF passages.
    """

    if not question or not question.strip():
        return []

    if index.ntotal == 0:
        return []

    # Generate embedding for the user's question.
    query_embedding = generate_embeddings(
        [question.strip()]
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32
    )

    # Limit results to the available vectors.
    top_k = min(top_k, index.ntotal)

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx < 0:
            continue

        results.append({
            "text": chunks[int(idx)]["text"],
            "page": chunks[int(idx)]["page"],
            "score": float(score)
        })

    return results
