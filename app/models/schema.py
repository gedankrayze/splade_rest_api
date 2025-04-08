"""
Pydantic models for API request/response schemas
"""

from typing import List, Dict, Any, Optional

from pydantic import BaseModel, Field


class Document(BaseModel):
    """Document model for indexing and retrieval"""
    id: str = Field(..., description="Unique document identifier")
    content: str = Field(..., description="Document content to be indexed")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional document metadata")
    source_id: Optional[str] = Field(default=None, description="Source document ID for chunked documents")
    chunk_index: Optional[int] = Field(default=None, description="Chunk index for chunked documents")


class DocumentBatch(BaseModel):
    """Batch of documents for bulk operations"""
    documents: List[Document] = Field(..., description="List of documents")


class SearchResult(BaseModel):
    """Search result model"""
    id: str = Field(..., description="Document identifier")
    content: str = Field(..., description="Document content")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Document metadata")
    score: float = Field(..., description="Relevance score")


class SearchResponse(BaseModel):
    """Search response model"""
    results: List[SearchResult] = Field(..., description="Search results")
    query_time_ms: float = Field(..., description="Query execution time in milliseconds")


class Collection(BaseModel):
    """Collection model for grouping documents"""
    id: str = Field(..., description="Unique collection identifier")
    name: str = Field(..., description="Collection name")
    description: Optional[str] = Field(default=None, description="Collection description")


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
