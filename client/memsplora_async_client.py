"""
MemSplora Async REST API Client
A comprehensive async client for interacting with the MemSplora API endpoints.
"""

import json
from typing import Optional, Dict, Any, List, overload

import aiohttp

from memsplora_types import (
    Collection, CollectionList, CollectionDetails, CollectionStats,
    Document, SearchResponse, MultiCollectionSearchResponse,
    DocumentAddResponse, BatchAddResponse
)


class MemSploraAsyncClient:
    """Async client for interacting with all MemSplora API endpoints."""

    def __init__(self, base_url: str, api_key: Optional[str] = None):
        """
        Initialize the async API client.

        Args:
            base_url: Base URL of the API server (e.g., 'https://api.example.com')
            api_key: Optional API key for authentication
        """
        self.base_url = base_url.rstrip('/')
        self.headers = {'Content-Type': 'application/json'}
        if api_key:
            self.headers['Authorization'] = f'Bearer {api_key}'
        self._session = None

    async def __aenter__(self):
        """Create session on context manager enter"""
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close session on context manager exit"""
        if self._session:
            await self._session.close()
            self._session = None

    @property
    def session(self) -> aiohttp.ClientSession:
        """Get or create the aiohttp session"""
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self._session

    # Collection Management
    async def list_collections(self) -> CollectionList:
        """List all collections."""
        async with self.session.get(f'{self.base_url}/collections', headers=self.headers) as response:
            response.raise_for_status()
            return await response.json()

    async def get_collection(self, collection_id: str) -> CollectionDetails:
        """Get collection details."""
        async with self.session.get(f'{self.base_url}/collections/{collection_id}', headers=self.headers) as response:
            response.raise_for_status()
            return await response.json()

    async def get_collection_stats(self, collection_id: str) -> CollectionStats:
        """Get collection statistics."""
        async with self.session.get(f'{self.base_url}/collections/{collection_id}/stats',
                                    headers=self.headers) as response:
            response.raise_for_status()
            return await response.json()

    async def create_collection(self, collection_id: str, name: str, description: Optional[str] = None) -> Collection:
        """Create a new collection."""
        data = {"id": collection_id, "name": name, "description": description}
        async with self.session.post(f'{self.base_url}/collections', headers=self.headers, json=data) as response:
            response.raise_for_status()
            return await response.json()

    async def delete_collection(self, collection_id: str) -> None:
        """Delete a collection."""
        async with self.session.delete(f'{self.base_url}/collections/{collection_id}',
                                       headers=self.headers) as response:
            response.raise_for_status()

    # Document Management
    async def add_document(self, collection_id: str, document: Dict[str, Any]) -> DocumentAddResponse:
        """Add a document to a collection."""
        async with self.session.post(f'{self.base_url}/documents/{collection_id}', headers=self.headers,
                                     json=document) as response:
            response.raise_for_status()
            return await response.json()

    async def batch_add_documents(self, collection_id: str, documents: List[Dict[str, Any]]) -> BatchAddResponse:
        """Add multiple documents to a collection in batch."""
        async with self.session.post(
                f'{self.base_url}/documents/{collection_id}/batch',
                headers=self.headers,
                json={"documents": documents}
        ) as response:
            response.raise_for_status()
            return await response.json()

    async def get_document(self, collection_id: str, document_id: str) -> Document:
        """Get a document from a collection."""
        async with self.session.get(f'{self.base_url}/documents/{collection_id}/{document_id}',
                                    headers=self.headers) as response:
            response.raise_for_status()
            return await response.json()

    async def delete_document(self, collection_id: str, document_id: str) -> None:
        """Delete a document from a collection."""
        async with self.session.delete(f'{self.base_url}/documents/{collection_id}/{document_id}',
                                       headers=self.headers) as response:
            response.raise_for_status()

    # Search Operations
    @overload
    async def search(self, collection_id: str, query: str, top_k: int = 10,
                     metadata_filter: Optional[Dict[str, Any]] = None) -> SearchResponse:
        ...

    async def search(self, collection_id: str, query: str, top_k: int = 10,
                     metadata_filter: Optional[Dict[str, Any]] = None) -> SearchResponse:
        """Perform a basic search in a collection."""
        params = {'query': query, 'top_k': top_k}
        if metadata_filter:
            params['metadata_filter'] = json.dumps(metadata_filter)

        async with self.session.get(
                f'{self.base_url}/search/{collection_id}',
                headers=self.headers,
                params=params
        ) as response:
            response.raise_for_status()
            return await response.json()

    async def search_all(self, query: str, top_k: int = 10,
                         metadata_filter: Optional[Dict[str, Any]] = None) -> MultiCollectionSearchResponse:
        """Search across all collections."""
        params = {'query': query, 'top_k': top_k}
        if metadata_filter:
            params['metadata_filter'] = json.dumps(metadata_filter)

        async with self.session.get(
                f'{self.base_url}/search',
                headers=self.headers,
                params=params
        ) as response:
            response.raise_for_status()
            return await response.json()

    async def advanced_search(
            self,
            collection_id: str,
            query: str,
            top_k: int = 10,
            min_score: float = 0.3,
            metadata_filter: Optional[Dict[str, Any]] = None,
            deduplicate: bool = True,
            merge_chunks: bool = True
    ) -> SearchResponse:
        """Perform an advanced search with chunking and deduplication."""
        params = {
            'query': query,
            'top_k': top_k,
            'min_score': min_score,
            'deduplicate': deduplicate,
            'merge_chunks': merge_chunks
        }
        if metadata_filter:
            params['metadata_filter'] = json.dumps(metadata_filter)

        async with self.session.get(
                f'{self.base_url}/advanced-search/{collection_id}',
                headers=self.headers,
                params=params
        ) as response:
            response.raise_for_status()
            return await response.json()

    async def advanced_search_all(
            self,
            query: str,
            top_k: int = 10,
            min_score: float = 0.3,
            metadata_filter: Optional[Dict[str, Any]] = None,
            deduplicate: bool = True,
            merge_chunks: bool = True
    ) -> MultiCollectionSearchResponse:
        """Perform an advanced search across all collections."""
        params = {
            'query': query,
            'top_k': top_k,
            'min_score': min_score,
            'deduplicate': deduplicate,
            'merge_chunks': merge_chunks
        }
        if metadata_filter:
            params['metadata_filter'] = json.dumps(metadata_filter)

        async with self.session.get(
                f'{self.base_url}/advanced-search',
                headers=self.headers,
                params=params
        ) as response:
            response.raise_for_status()
            return await response.json()
