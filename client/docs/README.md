# MemSplora API Client Documentation

This documentation covers the MemSplora API client libraries for interacting with the SPLADE Content Server.

## Table of Contents

- [Installation](#installation)
- [Clients Overview](#clients-overview)
    - [Synchronous Client](#synchronous-client)
    - [Asynchronous Client](#asynchronous-client)
- [API Reference](#api-reference)
    - [Collection Management](#collection-management)
    - [Document Management](#document-management)
    - [Search Operations](#search-operations)
  - [Geo-Spatial Search](#geo-spatial-search)
- [Data Types](#data-types)
- [Examples](#examples)
    - [Synchronous Client Examples](#synchronous-client-examples)
    - [Asynchronous Client Examples](#asynchronous-client-examples)
- [Error Handling](#error-handling)
- [Best Practices](#best-practices)

## Installation

Install the MemSplora client libraries using pip:

```bash
# Navigate to the client directory
cd client

# Install the required dependencies
pip install -r requirements.txt
```

## Clients Overview

The MemSplora API provides two client implementations:

### Synchronous Client

The `MemSploraClient` is a synchronous client for applications where simplicity is preferred over performance. It's
suitable for scripts, command-line tools, and applications with moderate throughput requirements.

```python
from memsplora_client import MemSploraClient

# Initialize the client
client = MemSploraClient(base_url="http://localhost:3000", api_key="optional-api-key")

# Examples
collections = client.list_collections()
search_results = client.search("technical-docs", "neural networks")
```

### Asynchronous Client

The `MemSploraAsyncClient` is designed for high-throughput applications where concurrency is important. It uses`aiohttp`
for asynchronous HTTP requests and is suitable for web servers, background workers, and applications that need to make
many concurrent API calls.

```python
import asyncio
from memsplora_async_client import MemSploraAsyncClient


async def main():
    # Initialize the client using a context manager
    async with MemSploraAsyncClient(base_url="http://localhost:3000") as client:
        # Examples
        collections = await client.list_collections()
        search_results = await client.search("technical-docs", "neural networks")


# Run the async function
asyncio.run(main())
```

## API Reference

### Collection Management

#### List Collections

Retrieves all collections from the server.

```python
# Synchronous
collections = client.list_collections()

# Asynchronous
collections = await client.list_collections()
```

**Returns**: `CollectionList` with the structure:

```json
{
  "collections": [
    {
      "id": "technical-docs",
      "name": "Technical Documentation",
      "description": "Technical documentation for our products"
    },
    ...
  ]
}
```

#### Get Collection

Retrieves details for a specific collection.

```python
# Synchronous
collection = client.get_collection("technical-docs")

# Asynchronous
collection = await client.get_collection("technical-docs")
```

**Parameters**:

- `collection_id` (string): The ID of the collection to retrieve

**Returns**: `CollectionDetails` with the structure:

```json
{
  "id": "technical-docs",
  "name": "Technical Documentation",
  "description": "Technical documentation for our products",
  "model_name": "custom-model-name",
  "stats": {
    "id": "technical-docs",
    "name": "Technical Documentation",
    "document_count": 42,
    "index_size": 42,
    "vocab_size": 30522
  }
}
```

#### Get Collection Stats

Retrieves statistics for a specific collection.

```python
# Synchronous
stats = client.get_collection_stats("technical-docs")

# Asynchronous
stats = await client.get_collection_stats("technical-docs")
```

**Parameters**:

- `collection_id` (string): The ID of the collection to retrieve stats for

**Returns**: `CollectionStats` with the structure:

```json
{
  "id": "technical-docs",
  "name": "Technical Documentation",
  "document_count": 42,
  "index_size": 42,
  "vocab_size": 30522
}
```

#### Create Collection

Creates a new collection.

```python
# Synchronous
collection = client.create_collection("technical-docs", "Technical Documentation",
                                      "Technical documentation for our products",
                                      model_name="custom-model-name")

# Asynchronous
collection = await client.create_collection("technical-docs", "Technical Documentation",
                                           "Technical documentation for our products",
                                           model_name="custom-model-name")
```

**Parameters**:

- `collection_id` (string): The ID for the new collection
- `name` (string): The display name of the collection
- `description` (string, optional): A description of the collection
- `model_name` (string, optional): Domain-specific model to use for this collection

**Returns**: `Collection` with the structure:

```json
{
  "id": "technical-docs",
  "name": "Technical Documentation",
  "description": "Technical documentation for our products",
  "model_name": "custom-model-name"
}
```

#### Delete Collection

Deletes a collection.

```python
# Synchronous
client.delete_collection("technical-docs")

# Asynchronous
await client.delete_collection("technical-docs")
```

**Parameters**:

- `collection_id` (string): The ID of the collection to delete

**Returns**: `None`

### Document Management

#### Add Document

Adds a document to a collection.

```python
# Synchronous
document = {
    "id": "doc-001",
    "content": "SPLADE is a sparse lexical model for information retrieval",
    "metadata": {"category": "AI", "author": "John Doe"},
    "location": {"latitude": 37.7749, "longitude": -122.4194}  # Optional location
}
result = client.add_document("technical-docs", document)

# Asynchronous
result = await client.add_document("technical-docs", document)
```

**Parameters**:

- `collection_id` (string): The ID of the collection to add the document to
- `document` (dict): The document to add, with keys:
    - `id` (string): Unique document identifier
    - `content` (string): Document content to index
    - `metadata` (dict, optional): Additional document metadata
  - `location` (dict, optional): Geographic coordinates as {latitude: float, longitude: float}

**Returns**: `DocumentAddResponse` with the structure:

```json
{
  "id": "doc-001",
  "success": true
}
```

#### Batch Add Documents

Adds multiple documents to a collection in a single request.

```python
# Synchronous
documents = [
    {
        "id": "doc-001",
        "content": "SPLADE is a sparse lexical model for information retrieval",
        "metadata": {"category": "AI", "author": "John Doe"},
        "location": {"latitude": 37.7749, "longitude": -122.4194}
    },
    {
        "id": "doc-002",
        "content": "Information retrieval is the science of searching for information",
        "metadata": {"category": "IR", "author": "Jane Smith"}
    }
]
result = client.batch_add_documents("technical-docs", documents)

# Asynchronous
result = await client.batch_add_documents("technical-docs", documents)
```

**Parameters**:

- `collection_id` (string): The ID of the collection to add the documents to
- `documents` (list): List of document objects to add, each can include location data

**Returns**: `BatchAddResponse` with the structure:

```json
{
  "added_count": 2,
  "success": true
}
```

#### Get Document

Retrieves a document from a collection.

```python
# Synchronous
document = client.get_document("technical-docs", "doc-001")

# Asynchronous
document = await client.get_document("technical-docs", "doc-001")
```

**Parameters**:

- `collection_id` (string): The ID of the collection containing the document
- `document_id` (string): The ID of the document to retrieve

**Returns**: `Document` with the structure:

```json
{
  "id": "doc-001",
  "content": "SPLADE is a sparse lexical model for information retrieval",
  "metadata": {
    "category": "AI",
    "author": "John Doe"
  },
  "location": {
    "latitude": 37.7749,
    "longitude": -122.4194
  }
}
```

#### Delete Document

Deletes a document from a collection.

```python
# Synchronous
client.delete_document("technical-docs", "doc-001")

# Asynchronous
await client.delete_document("technical-docs", "doc-001")
```

**Parameters**:

- `collection_id` (string): The ID of the collection containing the document
- `document_id` (string): The ID of the document to delete

**Returns**: `None`

### Search Operations

#### Basic Search

Performs a basic search within a specific collection.

```python
# Synchronous
results = client.search("technical-docs", "neural networks", top_k=5,
                        metadata_filter={"category": "AI"},
                        min_score=0.3)

# Asynchronous
results = await client.search("technical-docs", "neural networks", top_k=5,
                             metadata_filter={"category": "AI"},
                             min_score=0.3)
```

**Parameters**:

- `collection_id` (string): The ID of the collection to search in
- `query` (string): The search query
- `top_k` (int, optional): Number of results to return (default: 10)
- `metadata_filter` (dict, optional): Filter results by metadata fields
- `min_score` (float, optional): Minimum similarity score threshold (default: 0.3)

**Returns**: `SearchResponse` with the structure:

```json
{
  "results": [
    {
      "id": "doc-001",
      "content": "Neural networks are computational models...",
      "metadata": {
        "category": "AI",
        "author": "John Doe"
      },
      "score": 0.89
    },
    ...
  ],
  "query_time_ms": 12.5
}
```

#### Search All Collections

Performs a search across all collections.

```python
# Synchronous
results = client.search_all("neural networks", top_k=5,
                            metadata_filter={"category": "AI"},
                            min_score=0.3)

# Asynchronous
results = await client.search_all("neural networks", top_k=5,
                                 metadata_filter={"category": "AI"},
                                 min_score=0.3)
```

**Parameters**:

- `query` (string): The search query
- `top_k` (int, optional): Number of results to return per collection (default: 10)
- `metadata_filter` (dict, optional): Filter results by metadata fields
- `min_score` (float, optional): Minimum similarity score threshold (default: 0.3)

**Returns**: `MultiCollectionSearchResponse` with the structure:

```json
{
  "results": {
    "technical-docs": [
      {
        "id": "doc-001",
        "content": "Neural networks are computational models...",
        "metadata": {
          "category": "AI",
          "author": "John Doe"
        },
        "score": 0.89
      },
      ...
    ],
    "research-papers": [
      ...
    ]
  },
  "query_time_ms": 25.7
}
```

#### Advanced Search

Performs an advanced search within a specific collection with additional options for chunking and deduplication.

```python
# Synchronous
results = client.advanced_search(
    "technical-docs",
    "neural networks",
    top_k=5,
    min_score=0.5,
    metadata_filter={"category": "AI"},
    deduplicate=True,
    merge_chunks=True
)

# Asynchronous
results = await client.advanced_search(
    "technical-docs",
    "neural networks",
    top_k=5,
    min_score=0.5,
    metadata_filter={"category": "AI"},
    deduplicate=True,
    merge_chunks=True
)
```

**Parameters**:

- `collection_id` (string): The ID of the collection to search in
- `query` (string): The search query
- `top_k` (int, optional): Number of results to return (default: 10)
- `min_score` (float, optional): Minimum similarity score threshold (default: 0.3)
- `metadata_filter` (dict, optional): Filter results by metadata fields
- `deduplicate` (bool, optional): Remove duplicate chunks from same document (default: True)
- `merge_chunks` (bool, optional): Merge content from chunks of same document (default: True)

**Returns**: `SearchResponse` with the structure similar to basic search.

#### Advanced Search All Collections

Performs an advanced search across all collections with additional options.

```python
# Synchronous
results = client.advanced_search_all(
    "neural networks",
    top_k=5,
    min_score=0.5,
    metadata_filter={"category": "AI"},
    deduplicate=True,
    merge_chunks=True
)

# Asynchronous
results = await client.advanced_search_all(
    "neural networks",
    top_k=5,
    min_score=0.5,
    metadata_filter={"category": "AI"},
    deduplicate=True,
    merge_chunks=True
)
```

**Parameters**:

- Same as `advanced_search` but without `collection_id`

**Returns**: `MultiCollectionSearchResponse` with the structure similar to `search_all`.

### Geo-Spatial Search

Perform searches based on geographic location by adding location parameters to search methods:

#### Adding Documents with Location Data

```python
# Add document with location
document = {
    "id": "restaurant-123",
    "content": "Italian restaurant in San Francisco",
    "metadata": {"type": "restaurant", "cuisine": "italian"},
    "location": {
        "latitude": 37.7749,
        "longitude": -122.4194
    }
}
client.add_document("places", document)
```

#### Searching by Geographic Location

```python
# Synchronous geo-search
geo_results = client.search(
    "places",
    "restaurant",
    geo_search={
        "latitude": 37.7745,  # Search center point
        "longitude": -122.4190,
        "radius_km": 2.0  # Search radius in kilometers
    }
)

# Asynchronous geo-search
geo_results = await client.search(
    "places",
    "restaurant",
    geo_search={
        "latitude": 37.7745,
        "longitude": -122.4190,
        "radius_km": 2.0
    }
)
```

Results include distance information:

```json
{
  "results": [
    {
      "id": "restaurant-123",
      "content": "Italian restaurant in San Francisco",
      "metadata": {"type": "restaurant", "cuisine": "italian"},
      "location": {"latitude": 37.7749, "longitude": -122.4194},
      "distance_km": 0.53,
      "score": 0.92
    },
    ...
  ],
  "query_time_ms": 8.7
}
```

#### Advanced Geo-Spatial Search

Combine location-based search with other advanced features:

```python
results = client.advanced_search(
    "places",
    "italian restaurant",
    metadata_filter={"cuisine": "italian"},
    deduplicate=True,
    merge_chunks=True,
    geo_search={
        "latitude": 37.7745,
        "longitude": -122.4190,
        "radius_km": 5.0
    }
)
```

## Data Types

The MemSplora client libraries define several types to provide type safety and better IDE support:

### GeoCoordinates

Represents geographic coordinates.

```python
GeoCoordinates = TypedDict('GeoCoordinates', {
    'latitude': float,
    'longitude': float
})
```

### Document

Represents a document in a collection.

```python
Document = TypedDict('Document', {
    'id': str,
    'content': str,
    'metadata': Optional[Dict[str, Any]],
    'location': Optional[GeoCoordinates]  # Geographic location
})
```

### SearchResult

Represents a single search result.

```python
SearchResult = TypedDict('SearchResult', {
    'id': str,
    'content': str,
    'metadata': Optional[Dict[str, Any]],
    'score': float,
    'location': Optional[GeoCoordinates],  # Geographic location
    'distance_km': Optional[float]  # Distance from search point (geo-search only)
})
```

### SearchResponse

Represents the response from a search request on a single collection.

```python
SearchResponse = TypedDict('SearchResponse', {
    'results': List[SearchResult],
    'query_time_ms': float
})
```

### MultiCollectionSearchResponse

Represents the response from a search request across multiple collections.

```python
MultiCollectionSearchResponse = TypedDict('MultiCollectionSearchResponse', {
    'results': Dict[str, List[SearchResult]],
    'query_time_ms': float
})
```

### Collection

Represents a collection.

```python
Collection = TypedDict('Collection', {
    'id': str,
    'name': str,
    'description': Optional[str],
    'model_name': Optional[str]  # Domain-specific model name
})
```

See `memsplora_types.py` for the complete list of type definitions.

## Examples

### Synchronous Client Examples

#### Basic Usage

```python
from memsplora_client import MemSploraClient

# Initialize the client
client = MemSploraClient("http://localhost:3000")

# Create a collection
client.create_collection(
    "technical-docs",
    "Technical Documentation",
    "Technical documentation for our products"
)

# Add a document
document = {
    "id": "neural-networks-intro",
    "content": "Neural networks are computational models inspired by the human brain...",
    "metadata": {"category": "AI", "complexity": "beginner"}
}
client.add_document("technical-docs", document)

# Search for documents
results = client.search("technical-docs", "neural networks", top_k=5)
for result in results["results"]:
    print(f"{result['id']} (score: {result['score']}): {result['content'][:50]}...")
```

#### Geo-Spatial Search Example

```python
from memsplora_client import MemSploraClient

# Initialize the client
client = MemSploraClient("http://localhost:3000")

# Add documents with location data
places = [
    {
        "id": "place-1",
        "content": "Golden Gate Bridge in San Francisco",
        "metadata": {"type": "landmark"},
        "location": {"latitude": 37.8199, "longitude": -122.4783}
    },
    {
        "id": "place-2",
        "content": "Fisherman's Wharf - Popular tourist area",
        "metadata": {"type": "tourist-area"},
        "location": {"latitude": 37.8080, "longitude": -122.4177}
    }
]

for place in places:
    client.add_document("sf-places", place)

# Search for places near a specific location
results = client.search(
    "sf-places",
    "tourist",
    geo_search={
        "latitude": 37.7749,  # San Francisco downtown
        "longitude": -122.4194,
        "radius_km": 5.0  # 5 km radius
    }
)

# Display results with distances
for result in results["results"]:
    name = result["content"].split(" - ")[0]
    distance = result.get("distance_km", "unknown")
    print(f"{name} - {distance} km away")
```

#### Batch Processing

```python
from memsplora_client import MemSploraClient
import json

# Initialize the client
client = MemSploraClient("http://localhost:3000")

# Load documents from a JSON file
with open("documents.json", "r") as f:
    documents = json.load(f)

# Add documents in batch
result = client.batch_add_documents("technical-docs", documents)
print(f"Added {result['added_count']} documents")

# Search with metadata filtering
results = client.search(
    "technical-docs",
    "neural networks",
    metadata_filter={"category": "AI", "complexity": "beginner"}
)
print(f"Found {len(results['results'])} results")
```

### Asynchronous Client Examples

#### Basic Usage

```python
import asyncio
from memsplora_async_client import MemSploraAsyncClient


async def main():
    # Initialize the client using a context manager
    async with MemSploraAsyncClient("http://localhost:3000") as client:
        # Create a collection
        await client.create_collection(
            "technical-docs",
            "Technical Documentation",
            "Technical documentation for our products"
        )

        # Add a document
        document = {
            "id": "neural-networks-intro",
            "content": "Neural networks are computational models inspired by the human brain...",
            "metadata": {"category": "AI", "complexity": "beginner"}
        }
        await client.add_document("technical-docs", document)

        # Search for documents
        results = await client.search("technical-docs", "neural networks", top_k=5)
        for result in results["results"]:
            print(f"{result['id']} (score: {result['score']}): {result['content'][:50]}...")


# Run the async function
asyncio.run(main())
```

#### Concurrent Operations

```python
import asyncio
from memsplora_async_client import MemSploraAsyncClient


async def main():
    async with MemSploraAsyncClient("http://localhost:3000") as client:
        # Define multiple search queries
        queries = [
            "neural networks",
            "natural language processing",
            "computer vision",
            "reinforcement learning"
        ]

        # Create concurrent search tasks
        search_tasks = [
            client.search("technical-docs", query, top_k=3)
            for query in queries
        ]

        # Execute all searches concurrently
        results = await asyncio.gather(*search_tasks)

        # Process results
        for i, (query, response) in enumerate(zip(queries, results)):
            print(f"\nQuery: {query}")
            print(f"Found {len(response['results'])} results in {response['query_time_ms']:.2f}ms")

            for j, result in enumerate(response['results']):
                print(f"  Result {j + 1}: {result['id']} (score: {result['score']:.4f})")


# Run the async function
asyncio.run(main())
```

## Error Handling

Both client libraries use `requests.raise_for_status()` or `response.raise_for_status()` to raise exceptions for HTTP
errors. You should handle these exceptions in your code:

```python
# Synchronous error handling
from memsplora_client import MemSploraClient
import requests

client = MemSploraClient("http://localhost:3000")

try:
    collection = client.get_collection("non-existent-collection")
except requests.HTTPError as e:
    if e.response.status_code == 404:
        print("Collection not found")
    else:
        print(f"HTTP error: {e}")
except requests.RequestException as e:
    print(f"Request error: {e}")

# Asynchronous error handling
import asyncio
from memsplora_async_client import MemSploraAsyncClient
import aiohttp


async def main():
    async with MemSploraAsyncClient("http://localhost:3000") as client:
        try:
            collection = await client.get_collection("non-existent-collection")
        except aiohttp.ClientResponseError as e:
            if e.status == 404:
                print("Collection not found")
            else:
                print(f"HTTP error: {e}")
        except aiohttp.ClientError as e:
            print(f"Request error: {e}")


asyncio.run(main())
```

## Best Practices

### Performance Optimization

1. **Use Batch Operations**: When adding multiple documents, use `batch_add_documents` instead of individual calls to
   `add_document`.

2. **Use the Async Client for Concurrent Operations**: When making multiple API calls, use the async client with
   `asyncio.gather()` to execute operations concurrently.

3. **Filter Early**: Use metadata filters and geo-spatial filters to reduce result sets early in the pipeline.

4. **Set Appropriate Score Thresholds**: Use `min_score` in advanced search to filter out low-quality matches.

### Resource Management

1. **Always Use Context Managers with AsyncClient**: Use `async with` to ensure proper cleanup of resources.

```python
async with MemSploraAsyncClient(base_url) as client:
    # Your code here
    pass  # Resources automatically cleaned up
```

2. **Close Long-Running Clients**: If creating long-lived clients outside context managers, manually close them:

```python
client = MemSploraAsyncClient(base_url)
try:
# Your code here
finally:
    # Ensure resources are cleaned up
    await client.session.close()
```

### Error Handling

1. **Handle HTTP Errors**: Always wrap API calls in try/except blocks to handle HTTP errors gracefully.

2. **Retry Transient Failures**: Consider implementing retry logic for transient errors (5xx status codes).

3. **Log Detailed Error Information**: Log the full response content for debugging.

### Geo-Spatial Search Optimization

1. **Set Appropriate Radius**: Too large a radius can slow down searches and return irrelevant results.

2. **Combine with Text Queries**: For best results, combine geo-spatial search with relevant text queries.

3. **Use Metadata Filters with Geo-Search**: Further refine results by combining location filters with metadata filters.
