
import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Load the embedding model
model = SentenceTransformer(MODEL_NAME)


def generate_embeddings(texts, batch_size=32):
    """
    Convert a list of text passages into normalized embeddings.
    """

    if not texts:
        raise ValueError("No text provided for embedding.")

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return np.asarray(embeddings, dtype=np.float32)


if __name__ == "__main__":

    sample_texts = [
        "Winston Smith works at the Ministry of Truth.",
        "Julia is a character in the novel Nineteen Eighty-Four.",
        "The Thought Police monitor people's thoughts."
    ]

    print("Generating embeddings...")

    embeddings = generate_embeddings(sample_texts)

    print("\nEmbedding generation successful!")
    print("Number of text passages:", len(sample_texts))
    print("Embedding shape:", embeddings.shape)
    print("Embedding dimensions:", embeddings.shape[1])
