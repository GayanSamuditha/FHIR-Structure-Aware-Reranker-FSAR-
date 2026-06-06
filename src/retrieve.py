"""Stage-1 dense top-k retrieval."""

import pandas as pd
import numpy as np
from typing import Tuple

from src import config
from src.embed import embed_query, load_model
from src.index import load_index, NumpyIndex


def retrieve(
    query: str,
    index: NumpyIndex | None = None,
    model=None,
    k: int | None = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Retrieve top-k candidates for a query using dense retrieval.

    Args:
        query: Query text
        index: Pre-loaded index (optional, will load if not provided)
        model: Pre-loaded embedding model (optional)
        k: Number of results to return (defaults to config.TOP_K_RETRIEVE)

    Returns:
        Tuple of (results_df, scores) where results_df has columns from statements
        and scores is a numpy array of cosine similarities
    """
    if k is None:
        k = config.TOP_K_RETRIEVE

    # Load index if not provided
    if index is None:
        index = load_index()

    # Embed query
    query_vector = embed_query(query, model=model)

    # Search
    indices, scores = index.search(query_vector, k=k)

    # Get metadata
    results_df = index.get_metadata(indices).copy()
    results_df["score"] = scores

    return results_df, scores


def format_results(results_df: pd.DataFrame, truncate_text: int = 80) -> pd.DataFrame:
    """
    Format results for display.

    Args:
        results_df: Results DataFrame with id, resource_type, date, text, score
        truncate_text: Maximum length for text field

    Returns:
        Formatted DataFrame
    """
    display_df = results_df[["id", "resource_type", "date", "text", "score"]].copy()

    # Truncate text
    display_df["text"] = display_df["text"].apply(
        lambda x: x[:truncate_text] + "..." if len(x) > truncate_text else x
    )

    return display_df


def main():
    """Demo retrieval with a hardcoded query."""
    query = "What lab results did the patient have before starting a medication?"

    print("=" * 80)
    print("DENSE RETRIEVAL DEMO")
    print("=" * 80)
    print(f"Query: {query}")
    print()

    # Load resources once
    print("Loading index and model...")
    index = load_index()
    model = load_model()
    print()

    # Retrieve
    print(f"Retrieving top-{config.TOP_K_RETRIEVE} candidates...")
    results_df, scores = retrieve(query, index=index, model=model)

    # Show top 5
    print()
    print("=" * 80)
    print("TOP-5 RESULTS")
    print("=" * 80)
    display_df = format_results(results_df.head(5))

    # Print as a nice table
    for idx, row in display_df.iterrows():
        print(f"\nRank {idx + 1} (score: {row['score']:.4f})")
        print(f"  ID: {row['id']}")
        print(f"  Type: {row['resource_type']}")
        print(f"  Date: {row['date']}")
        print(f"  Text: {row['text']}")

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
