"""
API routes for search functionality
"""

import json
from typing import Dict, Any, Optional

from fastapi import APIRouter, HTTPException, Query, status, Depends

from app.api.dependencies import get_splade_service
from app.core.config import settings
from app.core.splade_service import SpladeService  # For type hints only
from app.models.schema import SearchResponse, PaginationInfo

router = APIRouter()


# Routes for searching in a collection
@router.get("/{collection_id}", response_model=SearchResponse)
@router.get("/{collection_id}/", response_model=SearchResponse)
async def search_collection(
        collection_id: str,
        query: str = Query(..., description="Search query"),
        top_k: int = Query(settings.DEFAULT_TOP_K, description="Number of results to return"),
        page: int = Query(1, description="Page number", ge=1),
        page_size: int = Query(settings.DEFAULT_TOP_K, description="Results per page", ge=1),
        metadata_filter: Optional[str] = Query(None, description="JSON string of metadata filters"),
        min_score: float = Query(settings.MIN_SCORE_THRESHOLD, description="Minimum score threshold for results"),
        latitude: Optional[float] = Query(None, description="Latitude for geo search", ge=-90.0, le=90.0),
        longitude: Optional[float] = Query(None, description="Longitude for geo search", ge=-180.0, le=180.0),
        radius_km: Optional[float] = Query(settings.GEO_DEFAULT_RADIUS_KM, description="Search radius in kilometers",
                                           gt=0),
        splade_service: SpladeService = Depends(get_splade_service)
):
    """Search for documents in a specific collection"""
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

    # Create geo filter if coordinates provided
    geo_filter = None
    if latitude is not None and longitude is not None:
        geo_filter = {
            "latitude": latitude,
            "longitude": longitude,
            "radius_km": radius_km
        }

    # For pagination, we need to get more results than just the current page
    # Calculate the total number of results to fetch
    total_to_fetch = page * page_size

    # Perform search with the calculated total
    results, query_time = splade_service.search(
        collection_id, query, total_to_fetch, filter_metadata, geo_filter, min_score
    )

    # Calculate pagination information
    total_results = len(results)
    total_pages = (total_results + page_size - 1) // page_size  # Ceiling division

    # Paginate the results
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

    return {
        "results": paginated_results,
        "query_time_ms": query_time,
        "pagination": pagination
    }


@router.get("/", response_model=Dict[str, Any])
@router.get("", response_model=Dict[str, Any])
async def search_all_collections(
        query: str = Query(..., description="Search query"),
        top_k: int = Query(settings.DEFAULT_TOP_K, description="Number of results per collection"),
        page: int = Query(1, description="Page number", ge=1),
        page_size: int = Query(settings.DEFAULT_TOP_K, description="Results per page", ge=1),
        metadata_filter: Optional[str] = Query(None, description="JSON string of metadata filters"),
        min_score: float = Query(settings.MIN_SCORE_THRESHOLD, description="Minimum score threshold for results"),
        latitude: Optional[float] = Query(None, description="Latitude for geo search", ge=-90.0, le=90.0),
        longitude: Optional[float] = Query(None, description="Longitude for geo search", ge=-180.0, le=180.0),
        radius_km: Optional[float] = Query(settings.GEO_DEFAULT_RADIUS_KM, description="Search radius in kilometers",
                                           gt=0),
        splade_service: SpladeService = Depends(get_splade_service)
):
    """Search across all collections"""
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

    # Calculate total results to fetch per collection for pagination
    total_to_fetch = page * page_size

    # Perform search with the calculated total
    results = splade_service.search_all_collections(
        query, total_to_fetch, filter_metadata, geo_filter, min_score
    )

    # Apply pagination to each collection's results
    all_results = results["results"]
    paginated_results = {}
    pagination_info = {}

    for collection_id, collection_results in all_results.items():
        # Calculate pagination for this collection
        total_results = len(collection_results)
        total_pages = (total_results + page_size - 1) // page_size  # Ceiling division

        # Paginate the results
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, total_results)
        collection_paginated = collection_results[start_idx:end_idx]

        if collection_paginated:  # Only include if we have results
            paginated_results[collection_id] = collection_paginated

            # Store pagination info for this collection
            pagination_info[collection_id] = {
                "page": page,
                "page_size": page_size,
                "total_results": total_results,
                "total_pages": total_pages
            }

    return {
        "results": paginated_results,
        "pagination": pagination_info,
        "query_time_ms": results["query_time_ms"]
    }
