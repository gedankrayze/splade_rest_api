"""
MemSplora REST API Client
A comprehensive client for interacting with the MemSplora API endpoints.
"""

import json
from typing import Optional, Dict, Any, List

import requests

from memsplora_types import (
    Collection, CollectionList, CollectionDetails, CollectionStats,
    Document, SearchResponse, MultiCollectionSearchResponse,
    DocumentAddResponse, BatchAddResponse
)


class MemSploraClient:
    """Client for interacting with all MemSplora API endpoints."""

    def __init__(self, base_url: str, api_key: Optional[str] = None):
        """
        Initialize the API client.

        Args:
            base_url: Base URL of the API server (e.g., 'https://api.example.com')
            api_key: Optional API key for authentication
        """
        self.base_url = base_url.rstrip('/')
        self.headers = {'Content-Type': 'application/json'}
        if api_key:
            self.headers['Authorization'] = f'Bearer {api_key}'

    # Collection Management
    def list_collections(self) -> CollectionList:
        """List all collections."""
        response = requests.get(f'{self.base_url}/collections', headers=self.headers)
        response.raise_for_status()
        return response.json()

    def get_collection(self, collection_id: str) -> CollectionDetails:
        """Get collection details."""
        response = requests.get(f'{self.base_url}/collections/{collection_id}', headers=self.headers)
        response.raise_for_status()
        return response.json()

    def get_collection_stats(self, collection_id: str) -> CollectionStats:
        """Get collection statistics."""
        response = requests.get(f'{self.base_url}/collections/{collection_id}/stats', headers=self.headers)
        response.raise_for_status()
        return response.json()

    def create_collection(self, collection_id: str, name: str, description: Optional[str] = None,
                          model_name: Optional[str] = None) -> Collection:
        """
        Create a new collection.
        
        Args:
            collection_id: Unique ID for the collection
            name: Display name for the collection
            description: Optional description
            model_name: Optional domain-specific model to use for this collection
        """
        data = {
            "id": collection_id,
            "name": name,
            "description": description,
            "model_name": model_name
        }
        response = requests.post(f'{self.base_url}/collections', headers=self.headers, json=data)
        response.raise_for_status()
        return response.json()

    def delete_collection(self, collection_id: str) -> None:
        """Delete a collection."""
        response = requests.delete(f'{self.base_url}/collections/{collection_id}', headers=self.headers)
        response.raise_for_status()

    # Document Management
    def add_document(self, collection_id: str, document: Dict[str, Any]) -> DocumentAddResponse:
        """Add a document to a collection."""
        response = requests.post(f'{self.base_url}/documents/{collection_id}', headers=self.headers, json=document)
        response.raise_for_status()
        return response.json()

    def batch_add_documents(self, collection_id: str, documents: List[Dict[str, Any]]) -> BatchAddResponse:
        """Add multiple documents to a collection in batch."""
        response = requests.post(
            f'{self.base_url}/documents/{collection_id}/batch',
            headers=self.headers,
            json={"documents": documents}
        )
        response.raise_for_status()
        return response.json()

    def get_document(self, collection_id: str, document_id: str) -> Document:
        """Get a document from a collection."""
        response = requests.get(f'{self.base_url}/documents/{collection_id}/{document_id}', headers=self.headers)
        response.raise_for_status()
        return response.json()

    def delete_document(self, collection_id: str, document_id: str) -> None:
        """Delete a document from a collection."""
        response = requests.delete(f'{self.base_url}/documents/{collection_id}/{document_id}', headers=self.headers)
        response.raise_for_status()

    # Search Operations
    def search(self, collection_id: str, query: str, top_k: int = 10,
               metadata_filter: Optional[Dict[str, Any]] = None) -> SearchResponse:
        """Perform a basic search in a collection."""
        params = {'query': query, 'top_k': top_k}
        if metadata_filter:
            params['metadata_filter'] = json.dumps(metadata_filter)
        
        response = requests.get(f'{self.base_url}/search/{collection_id}', headers=self.headers, params=params)
        response.raise_for_status()
        return response.json()

    def search_all(self, query: str, top_k: int = 10,
                   metadata_filter: Optional[Dict[str, Any]] = None) -> MultiCollectionSearchResponse:
        """Search across all collections."""
        params = {'query': query, 'top_k': top_k}
        if metadata_filter:
            params['metadata_filter'] = json.dumps(metadata_filter)
        
        response = requests.get(f'{self.base_url}/search', headers=self.headers, params=params)
        response.raise_for_status()
        return response.json()

    def advanced_search(
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

        response = requests.get(f'{self.base_url}/advanced-search/{collection_id}', headers=self.headers, params=params)
        response.raise_for_status()
        return response.json()

    def advanced_search_all(
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

        response = requests.get(f'{self.base_url}/advanced-search', headers=self.headers, params=params)
        response.raise_for_status()
        return response.json()
