"""
Pydantic models for API request/response schemas
"""

from typing import List, Dict, Any, Optional, Tuple

from pydantic import BaseModel, Field, validator


class GeoCoordinates(BaseModel):
    """Geographic coordinates model"""
    latitude: float = Field(..., ge=-90.0, le=90.0, description="Latitude in decimal degrees")
    longitude: float = Field(..., ge=-180.0, le=180.0, description="Longitude in decimal degrees")

    def to_tuple(self) -> Tuple[float, float]:
        """Convert to (lat, lon) tuple"""
        return (self.latitude, self.longitude)


class Document(BaseModel):
    """Document model for indexing and retrieval"""
    id: str = Field(..., description="Unique document identifier")
    content: str = Field(..., description="Document content to be indexed")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional document metadata")
    location: Optional[GeoCoordinates] = Field(default=None, description="Geographic location of the document")
    source_id: Optional[str] = Field(default=None, description="Source document ID for chunked documents")
    chunk_index: Optional[int] = Field(default=None, description="Chunk index for chunked documents")

    @validator('metadata', pre=True)
    def handle_legacy_location(cls, v, values):
        """Handle legacy location data in metadata for backward compatibility"""
        if v and 'latitude' in v and 'longitude' in v:
            # If we have location data in metadata but no location field,
            # create the location object
            if 'location' not in values or values['location'] is None:
                values['location'] = GeoCoordinates(
                    latitude=v['latitude'],
                    longitude=v['longitude']
                )
        return v


class DocumentBatch(BaseModel):
    """Batch of documents for bulk operations"""
    documents: List[Document] = Field(..., description="List of documents")


class SearchResult(BaseModel):
    """Search result model"""
    id: str = Field(..., description="Document identifier")
    content: str = Field(..., description="Document content")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Document metadata")
    score: float = Field(..., description="Relevance score")


class GeoSearchParams(BaseModel):
    """Geo-spatial search parameters"""
    latitude: float = Field(..., ge=-90.0, le=90.0, description="Latitude for geo search")
    longitude: float = Field(..., ge=-180.0, le=180.0, description="Longitude for geo search")
    radius_km: float = Field(default=10.0, gt=0, description="Search radius in kilometers")


class PaginationInfo(BaseModel):
    """Pagination information"""
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Number of results per page")
    total_results: int = Field(..., description="Total number of results")
    total_pages: int = Field(..., description="Total number of pages")

class SearchResponse(BaseModel):
    """Search response model"""
    results: List[SearchResult] = Field(..., description="Search results")
    query_time_ms: float = Field(..., description="Query execution time in milliseconds")
    pagination: Optional[PaginationInfo] = Field(default=None, description="Pagination information")


class Collection(BaseModel):
    """Collection model for grouping documents"""
    id: str = Field(..., description="Unique collection identifier")
    name: str = Field(..., description="Collection name")
    description: Optional[str] = Field(default=None, description="Collection description")
    model_name: Optional[str] = Field(default=None, description="Optional domain-specific model name")


class CollectionList(BaseModel):
    """List of collections"""
    collections: List[Collection] = Field(..., description="List of collections")


class CollectionStats(BaseModel):
    """Collection statistics"""
    id: str = Field(..., description="Collection identifier")
    name: str = Field(..., description="Collection name")
    document_count: int = Field(..., description="Number of documents in collection")
    index_size: int = Field(..., description="Size of FAISS index")
    vocab_size: int = Field(..., description="Model vocabulary size")
