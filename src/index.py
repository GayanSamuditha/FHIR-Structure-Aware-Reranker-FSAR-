"""Build and query a numpy-based cosine similarity index."""

import numpy as np
import pandas as pd
from typing import Tuple

from src import config


class NumpyIndex:
    """
    Simple numpy-based cosine similarity index.

    Since embeddings are already L2-normalized, cosine similarity
    is just the dot product.
    """

    def __init__(self, vectors: np.ndarray, metadata: pd.DataFrame):
        """
        Initialize the index.

        Args:
            vectors: Numpy array of shape (n, embedding_dim), L2-normalized
            metadata: DataFrame with record metadata (same order as vectors)
        """
        self.vectors = vectors
        self.metadata = metadata
        self.n_docs = len(vectors)

        assert len(vectors) == len(metadata), "Vectors and metadata must have same length"

    def search(self, query_vector: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for top-k most similar vectors.

        Args:
            query_vector: Query embedding, shape (embedding_dim,), L2-normalized
            k: Number of results to return

        Returns:
            Tuple of (indices, scores) where indices are into self.vectors
            and scores are cosine similarities
        """
        # Compute cosine similarities (dot product since vectors are normalized)
        scores = np.dot(self.vectors, query_vector)

        # Get top-k indices (argsort returns ascending, so we reverse)
        top_k = min(k, len(scores))
        top_indices = np.argsort(scores)[::-1][:top_k]
        top_scores = scores[top_indices]

        return top_indices, top_scores

    def get_metadata(self, indices: np.ndarray) -> pd.DataFrame:
        """
        Get metadata for the given indices.

        Args:
            indices: Array of indices

        Returns:
            DataFrame of metadata
        """
        return self.metadata.iloc[indices].reset_index(drop=True)


def build_index(vectors: np.ndarray, metadata: pd.DataFrame) -> NumpyIndex:
    """
    Build a numpy index.

    Args:
        vectors: Embedding vectors
        metadata: Metadata DataFrame

    Returns:
        NumpyIndex
    """
    return NumpyIndex(vectors, metadata)


def load_index() -> NumpyIndex:
    """
    Load the index from disk.

    Returns:
        NumpyIndex with vectors and metadata
    """
    print("Loading vectors and metadata...")
    vectors = np.load(config.VECTORS_PATH)
    metadata = pd.read_parquet(config.STATEMENTS_PATH)

    print(f"Loaded {len(vectors)} vectors, {len(metadata)} metadata records")
    assert len(vectors) == len(metadata), "Vector/metadata count mismatch"

    return build_index(vectors, metadata)


def main():
    """Test the index."""
    index = load_index()

    print("\n" + "=" * 60)
    print("INDEX SUMMARY")
    print("=" * 60)
    print(f"Documents: {index.n_docs}")
    print(f"Embedding dim: {index.vectors.shape[1]}")
    print("=" * 60)


if __name__ == "__main__":
    main()
