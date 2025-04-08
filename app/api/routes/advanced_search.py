"""
API routes for advanced search functionality with deduplication and score thresholding
"""

import json
from typing import Dict, Any, Optional

from fastapi import APIRouter, HTTPException, Query, status, Depends

from app.api.dependencies import get_splade_service
from app.core.config import settings
from app.core.search_utils import deduplicate_and_threshold_results, merge_chunk_content
from app.core.splade_service import SpladeService
from app.models.schema import SearchResponse

router = APIRouter()


@router.get("/{collection_id}", response_model=SearchResponse)
@router.get("/{collection_id}/", response_model=SearchResponse)
async def advanced_search(
        collection_id: str,
        query: str = Query(..., description="Search query"),
        top_k: int = Query(settings.DEFAULT_TOP_K, description="Number of results to return"),
        min_score: float = Query(0.3, description="Minimum similarity score threshold (0-1)"),
        metadata_filter: Optional[str] = Query(None, description="JSON string of metadata filters"),
        deduplicate: bool = Query(True, description="Deduplicate results from same document"),
        merge_chunks: bool = Query(True, description="Merge chunks from the same document"),
        splade_service: SpladeService = Depends(get_splade_service)
):
    """
    Search for documents in a specific collection with advanced options
    
    - min_score: Only return results with similarity score >= this threshold (0.0-1.0)
    - deduplicate: Remove duplicate chunks from the same original document
    - merge_chunks: Merge content from chunks of the same document
    """
    # Check if collection exists
    if not splade_service.get_collection(collection_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Collection with ID {collection_id} not found"
        )

    # Parse metadata filter if provided
    filter_metadata = None
    if metadata_filter:
        try:
            filter_metadata = json.loads(metadata_filter)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid metadata filter JSON"
            )

    # Perform search with minimum score threshold 
    # But get more results than requested to allow for filtering
    effective_top_k = min(top_k * 3, 100)  # Get more but cap at 100
    results, query_time = splade_service.search(
        collection_id,
        query,
        effective_top_k,
        filter_metadata
    )

    # Apply post-processing
    if min_score > 0 or deduplicate:
        results = deduplicate_and_threshold_results(
            results,
            min_score_threshold=min_score,
            deduplicate_by_original_id=deduplicate
        )

    # Merge chunks if requested
    if merge_chunks:
        has_chunks = any(r.get("metadata", {}).get("is_chunk", False) for r in results)
        if has_chunks:
            results = merge_chunk_content(results)

    # Limit to requested number
    results = results[:top_k]

    return {
        "results": results,
        "query_time_ms": query_time
    }


@router.get("/", response_model=Dict[str, Any])
@router.get("", response_model=Dict[str, Any])
async def advanced_search_all(
        query: str = Query(..., description="Search query"),
        top_k: int = Query(settings.DEFAULT_TOP_K, description="Number of results per collection"),
        min_score: float = Query(0.3, description="Minimum similarity score threshold (0-1)"),
        metadata_filter: Optional[str] = Query(None, description="JSON string of metadata filters"),
        deduplicate: bool = Query(True, description="Deduplicate results from same document"),
        merge_chunks: bool = Query(True, description="Merge chunks from the same document"),
        splade_service: SpladeService = Depends(get_splade_service)
):
    """
    Search across all collections with advanced options
    
    - min_score: Only return results with similarity score >= this threshold (0.0-1.0)
    - deduplicate: Remove duplicate chunks from the same original document
    - merge_chunks: Merge content from chunks of the same document
    """
    # Parse metadata filter if provided
    filter_metadata = None
    if metadata_filter:
        try:
            filter_metadata = json.loads(metadata_filter)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid metadata filter JSON"
            )

    # Perform raw search
    search_results = splade_service.search_all_collections(
        query,
        top_k * 3,  # Get more results for filtering
        filter_metadata
    )

    # Process each collection's results
    all_results = search_results["results"]
    processed_results = {}

    for collection_id, results in all_results.items():
        # Apply post-processing
        if min_score > 0 or deduplicate:
            processed = deduplicate_and_threshold_results(
                results,
                min_score_threshold=min_score,
                deduplicate_by_original_id=deduplicate
            )
        else:
            processed = results

        # Merge chunks if requested
        if merge_chunks:
            has_chunks = any(r.get("metadata", {}).get("is_chunk", False) for r in processed)
            if has_chunks:
                processed = merge_chunk_content(processed)

        # Limit to requested number and add to results
        if processed:
            processed_results[collection_id] = processed[:top_k]

    return {
        "results": processed_results,
        "query_time_ms": search_results["query_time_ms"]
    }
