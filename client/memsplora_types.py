"""
Type definitions for the MemSplora API client.
"""

from typing import List, Dict, Any, Optional, TypedDict


class Document(TypedDict):
    """Document representation"""
    id: str
    content: str
    metadata: Optional[Dict[str, Any]]


class SearchResult(TypedDict):
    """Single search result"""
    id: str
    content: str
    metadata: Optional[Dict[str, Any]]
    score: float


class SearchResponse(TypedDict):
    """Search response from the API"""
    results: List[SearchResult]
    query_time_ms: float


class MultiCollectionSearchResponse(TypedDict):
    """Search response across multiple collections"""
    results: Dict[str, List[SearchResult]]
    query_time_ms: float


class Collection(TypedDict):
    """Collection representation"""
    id: str
    name: str
    description: Optional[str]


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


class DocumentAddResponse(TypedDict):
    """Response when adding a document"""
    id: str
    success: bool


class BatchAddResponse(TypedDict):
    """Response when adding documents in batch"""
    added_count: int
    success: bool
