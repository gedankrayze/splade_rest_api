"""
Search utilities for deduplication and result processing
"""

import logging
from typing import List, Dict, Any

# Configure logging
logger = logging.getLogger("search_utils")


def deduplicate_and_threshold_results(
        results: List[Dict[str, Any]],
        min_score_threshold: float = 0.0,
        deduplicate_by_original_id: bool = True
) -> List[Dict[str, Any]]:
    """
    Process search results to apply score thresholding and deduplication
    
    Args:
        results: List of search result documents with scores
        min_score_threshold: Minimum score to include a result (0.0 to 1.0)
        deduplicate_by_original_id: Whether to deduplicate using original_document_id
        
    Returns:
        Processed list of results after filtering and deduplication
    """
    # Apply score thresholding
    filtered_results = [
        result for result in results
        if result["score"] >= min_score_threshold
    ]

    if len(filtered_results) != len(results):
        logger.debug(
            f"Filtered out {len(results) - len(filtered_results)} results below score threshold {min_score_threshold}")
    
    # If no deduplication needed, return thresholded results
    if not deduplicate_by_original_id:
        return filtered_results

    # Track highest scoring document for each original document
    original_docs = {}
    chunks = []

    # First pass - separate original docs and chunks, find best score per original doc
    for result in filtered_results:
        metadata = result.get("metadata", {})

        if metadata and metadata.get("is_chunk") and "original_document_id" in metadata:
            # Track this as a chunk
            chunks.append(result)

            # Update highest score for original document if needed
            original_id = metadata["original_document_id"]
            current_score = result["score"]

            if original_id not in original_docs or current_score > original_docs[original_id]["score"]:
                original_docs[original_id] = result
        else:
            # Handle non-chunk documents normally
            doc_id = result["id"]
            if doc_id not in original_docs or result["score"] > original_docs[doc_id]["score"]:
                original_docs[doc_id] = result

    # Second pass - collect other chunks that belong to original documents for context
    # (We might want to merge their content or use them for context in a real application)
    related_chunks = {}
    for chunk in chunks:
        original_id = chunk["metadata"]["original_document_id"]
        if original_id in original_docs:
            if original_id not in related_chunks:
                related_chunks[original_id] = []
            related_chunks[original_id].append(chunk)

    # The final results are just the best representative for each original document
    deduplicated_results = list(original_docs.values())

    # Sort by score in descending order
    deduplicated_results.sort(key=lambda x: x["score"], reverse=True)

    logger.debug(f"Deduplicated from {len(filtered_results)} to {len(deduplicated_results)} results")

    return deduplicated_results


def merge_chunk_content(
        results: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Merge chunk content from the same original document
    
    Args:
        results: List of search result documents
        
    Returns:
        List with merged chunks from the same document
    """
    # Group chunks by original document ID
    grouped_chunks = {}
    non_chunks = []

    for result in results:
        metadata = result.get("metadata", {})

        if metadata and metadata.get("is_chunk") and "original_document_id" in metadata:
            original_id = metadata["original_document_id"]
            if original_id not in grouped_chunks:
                grouped_chunks[original_id] = []
            grouped_chunks[original_id].append(result)
        else:
            non_chunks.append(result)

    # Merge chunks from the same document
    merged_results = non_chunks.copy()

    for original_id, chunks in grouped_chunks.items():
        # Sort chunks by chunk_index
        chunks.sort(key=lambda x: x["metadata"].get("chunk_index", 0))

        # Take the highest score
        max_score = max(chunk["score"] for chunk in chunks)

        # Merge content with separators
        merged_content = "\n\n[...]\n\n".join(chunk["content"] for chunk in chunks)

        # Create a merged result
        merged_result = {
            "id": original_id,
            "content": merged_content,
            "metadata": {
                "original_document_id": original_id,
                "merged_from_chunks": len(chunks),
                "chunk_ids": [chunk["id"] for chunk in chunks]
            },
            "score": max_score
        }

        merged_results.append(merged_result)

    # Sort by score
    merged_results.sort(key=lambda x: x["score"], reverse=True)

    return merged_results
