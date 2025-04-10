"""
API routes for advanced search functionality with deduplication and score thresholding
"""

import json
import logging
from typing import Dict, Any, Optional

from fastapi import APIRouter, HTTPException, Query, status, Depends

from app.api.dependencies import get_splade_service
from app.core.config import settings
from app.core.search_utils import deduplicate_and_threshold_results, merge_chunk_content
from app.core.splade_service import SpladeService  # For type hints only
from app.llm.client import AsyncLLMClient
from app.models.schema import SearchResponse, PaginationInfo

router = APIRouter()

logger = logging.getLogger(__name__)


async def expand_query_with_llm(query: str, model: str = None) -> str:
    """Use LLM to expand the search query with synonyms and related terms"""
    # Use default model if none provided
    model_to_use = model or settings.LLM_DEFAULT_MODEL or "gpt-4o-mini"

    # Create LLM client
    llm_client = AsyncLLMClient()

    prompt = f"""
    I need to search for information in a document about: "{query}"
    
    Please create an expanded search query that includes:
    - Synonyms for key terms
    - Related technical terms
    - Alternative ways to express the same concept
    
    Only provide the expanded search query without explanations.
    """

    try:
        response = await llm_client.response(
            user_input=prompt,
            model=model_to_use,
            temperature=0.3,
            max_tokens=200
        )

        expanded_query = response.get("text", "").strip()
        return expanded_query if expanded_query else query
    except Exception as e:
        # Log error and return original query
        logger.error(f"Query expansion failed: {e}")
        return query


@router.get("/{collection_id}", response_model=SearchResponse)
@router.get("/{collection_id}/", response_model=SearchResponse)
async def advanced_search(
        collection_id: str,
        query: str = Query(..., description="Search query"),
        top_k: int = Query(settings.DEFAULT_TOP_K, description="Number of results to return"),
        page: int = Query(1, description="Page number", ge=1),
        page_size: int = Query(settings.DEFAULT_TOP_K, description="Results per page", ge=1),
        min_score: float = Query(0.3, description="Minimum similarity score threshold (0-1)"),
        metadata_filter: Optional[str] = Query(None, description="JSON string of metadata filters"),
        deduplicate: bool = Query(True, description="Deduplicate results from same document"),
        merge_chunks: bool = Query(True, description="Merge chunks from the same document"),
        latitude: Optional[float] = Query(None, description="Latitude for geo search", ge=-90.0, le=90.0),
        longitude: Optional[float] = Query(None, description="Longitude for geo search", ge=-180.0, le=180.0),
        radius_km: Optional[float] = Query(settings.GEO_DEFAULT_RADIUS_KM, description="Search radius in kilometers",
                                           gt=0),
        query_expansion: bool = Query(False, description="Use LLM to expand the query"),
        expansion_model: Optional[str] = Query(None, description="Optional LLM model to use for query expansion"),
        splade_service: SpladeService = Depends(get_splade_service)
):
    """
    Search for documents in a specific collection with advanced options
    
    - min_score: Only return results with similarity score >= this threshold (0.0-1.0)
    - deduplicate: Remove duplicate chunks from the same original document
    - merge_chunks: Merge content from chunks of the same document
    - latitude/longitude/radius_km: Filter results by geographic location
    """
    # Check if collection exists
    if not splade_service.get_collection(collection_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Collection with ID {collection_id} not found"
        )

    # Expand query if requested
    search_query = query
    if query_expansion and settings.LLM_ENABLED:
        expanded_query = await expand_query_with_llm(query, expansion_model)
        if expanded_query and expanded_query != query:
            logger.info(f"Query expanded from '{query}' to '{expanded_query}'")
            search_query = expanded_query
    
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

    # Create geo filter if coordinates provided
    geo_filter = None
    if latitude is not None and longitude is not None:
        geo_filter = {
            "latitude": latitude,
            "longitude": longitude,
            "radius_km": radius_km
        }

    # For pagination, we need to get more results than requested
    # Calculate the effective number of results to fetch, considering pagination
    # We multiply by 3 to account for filtering, deduplication, and score thresholding
    effective_top_k = min(page * page_size * 3, 500)  # Get more but cap at 500

    # If top_k is explicitly specified and larger than our calculation, use that instead
    if top_k > effective_top_k:
        effective_top_k = top_k
    
    results, query_time = splade_service.search(
        collection_id,
        search_query,  # Use expanded query if available
        effective_top_k,
        filter_metadata,
        geo_filter
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

    # Calculate total results for pagination
    total_results = len(results)
    total_pages = (total_results + page_size - 1) // page_size  # Ceiling division

    # Apply pagination
    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, total_results)
    paginated_results = results[start_idx:end_idx]

    # Create pagination info
    pagination = PaginationInfo(
        page=page,
        page_size=page_size,
        total_results=total_results,
        total_pages=total_pages
    )

    response_data = {
        "results": paginated_results,
        "query_time_ms": query_time,
        "pagination": pagination
    }

    # Add expanded query to response if it was used
    if query_expansion and search_query != query:
        response_data["expanded_query"] = search_query
        response_data["original_query"] = query

    return response_data


@router.get("/", response_model=Dict[str, Any])
@router.get("", response_model=Dict[str, Any])
async def advanced_search_all(
        query: str = Query(..., description="Search query"),
        top_k: int = Query(settings.DEFAULT_TOP_K, description="Number of results per collection"),
        page: int = Query(1, description="Page number", ge=1),
        page_size: int = Query(settings.DEFAULT_TOP_K, description="Results per page", ge=1),
        min_score: float = Query(0.3, description="Minimum similarity score threshold (0-1)"),
        metadata_filter: Optional[str] = Query(None, description="JSON string of metadata filters"),
        deduplicate: bool = Query(True, description="Deduplicate results from same document"),
        merge_chunks: bool = Query(True, description="Merge chunks from the same document"),
        latitude: Optional[float] = Query(None, description="Latitude for geo search", ge=-90.0, le=90.0),
        longitude: Optional[float] = Query(None, description="Longitude for geo search", ge=-180.0, le=180.0),
        radius_km: Optional[float] = Query(settings.GEO_DEFAULT_RADIUS_KM, description="Search radius in kilometers",
                                           gt=0),
        query_expansion: bool = Query(False, description="Use LLM to expand the query"),
        expansion_model: Optional[str] = Query(None, description="Optional LLM model to use for query expansion"),
        splade_service: SpladeService = Depends(get_splade_service)
):
    """
    Search across all collections with advanced options
    
    - min_score: Only return results with similarity score >= this threshold (0.0-1.0)
    - deduplicate: Remove duplicate chunks from the same original document
    - merge_chunks: Merge content from chunks of the same document
    - latitude/longitude/radius_km: Filter results by geographic location
    """
    # Expand query if requested
    search_query = query
    if query_expansion and settings.LLM_ENABLED:
        expanded_query = await expand_query_with_llm(query, expansion_model)
        if expanded_query and expanded_query != query:
            logger.info(f"Query expanded from '{query}' to '{expanded_query}'")
            search_query = expanded_query
            
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

    # Create geo filter if coordinates provided
    geo_filter = None
    if latitude is not None and longitude is not None:
        geo_filter = {
            "latitude": latitude,
            "longitude": longitude,
            "radius_km": radius_km
        }

    # Calculate the effective number of results to fetch, considering pagination
    # We multiply by 3 to account for filtering, deduplication, and score thresholding
    effective_top_k = min(page * page_size * 3, 500)  # Get more but cap at 500

    # If top_k is explicitly specified and larger than our calculation, use that instead
    if top_k > effective_top_k:
        effective_top_k = top_k

    # Perform raw search
    search_results = splade_service.search_all_collections(
        search_query,  # Use expanded query if available
        effective_top_k,  # Get more results for filtering
        filter_metadata,
        geo_filter,
        min_score
    )

    # Process each collection's results
    all_results = search_results["results"]
    processed_results = {}
    pagination_info = {}

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

        # Calculate total results for pagination
        total_results = len(processed)
        total_pages = (total_results + page_size - 1) // page_size  # Ceiling division

        # Apply pagination
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, total_results)
        paginated_results = processed[start_idx:end_idx]

        # Add to results if we have any
        if paginated_results:
            processed_results[collection_id] = paginated_results

            # Add pagination info for this collection
            pagination_info[collection_id] = {
                "page": page,
                "page_size": page_size,
                "total_results": total_results,
                "total_pages": total_pages
            }

    response_data = {
        "results": processed_results,
        "pagination": pagination_info,
        "query_time_ms": search_results["query_time_ms"]
    }

    # Add expanded query to response if it was used
    if query_expansion and search_query != query:
        response_data["expanded_query"] = search_query
        response_data["original_query"] = query

    return response_data
