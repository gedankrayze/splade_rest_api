"""
API routes for search functionality
"""

import json
from typing import Dict, Any, Optional

from fastapi import APIRouter, HTTPException, Query, status, Depends

from app.api.dependencies import get_splade_service
from app.core.config import settings
from app.core.splade_service import SpladeService  # For type hints only
from app.models.schema import SearchResponse

router = APIRouter()


# Routes for searching in a collection
@router.get("/{collection_id}", response_model=SearchResponse)
@router.get("/{collection_id}/", response_model=SearchResponse)
async def search_collection(
        collection_id: str,
        query: str = Query(..., description="Search query"),
        top_k: int = Query(settings.DEFAULT_TOP_K, description="Number of results to return"),
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

    # Perform search
    results, query_time = splade_service.search(
        collection_id, query, top_k, filter_metadata, geo_filter, min_score
    )

    return {
        "results": results,
        "query_time_ms": query_time
    }


@router.get("/", response_model=Dict[str, Any])
@router.get("", response_model=Dict[str, Any])
async def search_all_collections(
        query: str = Query(..., description="Search query"),
        top_k: int = Query(settings.DEFAULT_TOP_K, description="Number of results per collection"),
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

    # Perform search
    results = splade_service.search_all_collections(
        query, top_k, filter_metadata, geo_filter, min_score
    )

    return results
