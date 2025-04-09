"""
Type definitions for the MemSplora API client.
"""

from typing import List, Dict, Any, Optional, TypedDict


class GeoCoordinates(TypedDict):
    """Geographic coordinates model"""
    latitude: float  # Latitude in decimal degrees (-90.0 to 90.0)
    longitude: float  # Longitude in decimal degrees (-180.0 to 180.0)


class Document(TypedDict, total=False):
    """Document representation"""
    id: str
    content: str
    metadata: Optional[Dict[str, Any]]
    location: Optional[GeoCoordinates]  # Geographic location of the document
    source_id: Optional[str]  # Source document ID for chunked documents
    chunk_index: Optional[int]  # Chunk index for chunked documents


class SearchResult(TypedDict, total=False):
    """Single search result"""
    id: str
    content: str
    metadata: Optional[Dict[str, Any]]
    score: float
    location: Optional[GeoCoordinates]  # Geographic location of the result
    distance_km: Optional[float]  # Distance in kilometers (only present in geo searches)


class SearchResponse(TypedDict):
    """Search response from the API"""
    results: List[SearchResult]
    query_time_ms: float


class MultiCollectionSearchResponse(TypedDict):
    """Search response across multiple collections"""
    results: Dict[str, List[SearchResult]]
    query_time_ms: float


class Collection(TypedDict, total=False):
    """Collection representation"""
    id: str
    name: str
    description: Optional[str]
    model_name: Optional[str]  # Optional domain-specific model name


class CollectionList(TypedDict):
    """Response for list collections endpoint"""
    collections: List[Collection]


class CollectionStats(TypedDict):
    """Collection statistics"""
    id: str
    name: str
    document_count: int
    index_size: int
    vocab_size: int


class CollectionDetails(TypedDict):
    """Detailed collection information"""
    id: str
    name: str
    description: Optional[str]
    stats: CollectionStats
    model_name: Optional[str]  # Optional domain-specific model name


class DocumentAddResponse(TypedDict):
    """Response when adding a document"""
    id: str
    success: bool


class BatchAddResponse(TypedDict):
    """Response when adding documents in batch"""
    added_count: int
    success: bool


class GeoSearchParams(TypedDict, total=False):
    """Parameters for geo-spatial search"""
    latitude: float
    longitude: float
    radius_km: float
