"""Batch-embed statements using TF-IDF vectorizer with caching."""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

from src import config

# Path for the vectorizer pickle
VECTORIZER_PATH = config.ARTIFACTS_DIR / "tfidf_vectorizer.pkl"


def load_model():
    """
    Load the TF-IDF vectorizer model.

    Returns:
        TfidfVectorizer or None if needs to be fitted
    """
    if VECTORIZER_PATH.exists():
        print(f"Loading TF-IDF vectorizer from {VECTORIZER_PATH}")
        with open(VECTORIZER_PATH, "rb") as f:
            vectorizer = pickle.load(f)
        return vectorizer
    else:
        print("No cached vectorizer found")
        return None


def should_reembed(statements_df: pd.DataFrame) -> bool:
    """
    Check if we need to re-embed statements.

    Args:
        statements_df: DataFrame of statements

    Returns:
        True if we need to embed, False if cache is valid
    """
    # Check if both files exist
    if not config.VECTORS_PATH.exists() or not VECTORIZER_PATH.exists():
        print("No cached vectors or vectorizer found, will embed")
        return True

    # Check if row count matches
    try:
        cached_vectors = np.load(config.VECTORS_PATH)
        if len(cached_vectors) != len(statements_df):
            print(f"Row count mismatch: cached={len(cached_vectors)}, current={len(statements_df)}, will re-embed")
            return True
        else:
            print(f"Cache valid: {len(cached_vectors)} vectors match statement count")
            return False
    except Exception as e:
        print(f"Error loading cached vectors: {e}, will re-embed")
        return True


def embed_statements(statements_df: pd.DataFrame, batch_size: int = 256) -> np.ndarray:
    """
    Batch-embed all statements using TF-IDF and cache to disk.

    Args:
        statements_df: DataFrame with 'text' column
        batch_size: Unused, kept for API compatibility

    Returns:
        Numpy array of shape (n_statements, max_features)
    """
    # Check cache
    if not should_reembed(statements_df):
        print(f"Loading cached vectors from {config.VECTORS_PATH}")
        vectors = np.load(config.VECTORS_PATH)
        return vectors

    # Extract texts
    texts = statements_df["text"].tolist()

    # Create and fit TF-IDF vectorizer
    print(f"Fitting TF-IDF vectorizer on {len(texts)} statements...")
    print("  max_features=8000, sublinear_tf=True")

    vectorizer = TfidfVectorizer(
        max_features=8000,
        sublinear_tf=True,
        lowercase=True,
        stop_words='english',
    )

    # Fit and transform
    tfidf_matrix = vectorizer.fit_transform(texts)

    # Convert sparse to dense float32
    print("Converting sparse matrix to dense float32...")
    vectors = tfidf_matrix.toarray().astype(np.float32)

    # L2 normalize for cosine similarity
    print("L2 normalizing vectors...")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    vectors = vectors / norms

    # Save vectorizer
    print(f"Saving vectorizer to {VECTORIZER_PATH}")
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    # Save vectors
    print(f"Saving vectors to {config.VECTORS_PATH}")
    np.save(config.VECTORS_PATH, vectors)
    print(f"✓ Saved {vectors.shape} vectors")

    return vectors


def embed_query(query: str, model=None) -> np.ndarray:
    """
    Embed a single query string.

    Args:
        query: Query text
        model: Pre-loaded vectorizer (optional, will load if not provided)

    Returns:
        Numpy array of shape (1, max_features) - note: returns 2D for sklearn compatibility
    """
    if model is None:
        model = load_model()
        if model is None:
            raise ValueError("No fitted vectorizer found. Run embed_statements first.")

    # Transform query
    vector = model.transform([query]).toarray().astype(np.float32)

    # L2 normalize
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm

    # Return as 1D array for consistency
    return vector[0]


def get_embedding_dim() -> int:
    """
    Get the embedding dimension for the configured model.

    Returns:
        Embedding dimension (8000 for TF-IDF with max_features=8000)
    """
    return 8000


def main():
    """Main entry point for M2 embedding."""
    print("Loading statements...")
    df = pd.read_parquet(config.STATEMENTS_PATH)
    print(f"Loaded {len(df)} statements")

    # Embed
    vectors = embed_statements(df)

    print("\n" + "=" * 60)
    print("EMBEDDING SUMMARY")
    print("=" * 60)
    print(f"Statements: {len(df)}")
    print(f"Vectors shape: {vectors.shape}")
    print(f"Embedding dim: {vectors.shape[1]}")
    print(f"Cached at: {config.VECTORS_PATH}")
    print(f"Vectorizer at: {VECTORIZER_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
