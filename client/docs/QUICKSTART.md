# MemSplora Client Quickstart Guide

This quickstart guide will help you get started with the MemSplora client libraries for the SPLADE Content Server.

## Installation

```bash
# Navigate to the client directory
cd client

# Install dependencies
pip install -r requirements.txt
```

## Basic Usage - Synchronous Client

The synchronous client is simple to use and suitable for scripts and applications where performance is not critical:

```python
from memsplora_client import MemSploraClient

# Initialize the client
client = MemSploraClient("http://localhost:3000")

# Create a collection
client.create_collection(
    "my-collection",
    "My First Collection",
    "A collection for testing the client"
)

# Add a document
document = {
    "id": "doc-001",
    "content": "This is a test document.",
    "metadata": {
        "category": "test",
        "author": "User"
    }
}
client.add_document("my-collection", document)

# Search for documents
results = client.search("my-collection", "test document")
print(f"Found {len(results['results'])} results")
```

## Location-Based Search

The client supports geo-spatial search capabilities for finding documents based on geographic location:

```python
from memsplora_client import MemSploraClient

# Initialize the client
client = MemSploraClient("http://localhost:3000")

# Add a document with location data
document = {
    "id": "restaurant-001",
    "content": "Italian Restaurant in San Francisco",
    "metadata": {
        "category": "restaurant",
        "cuisine": "italian"
    },
    "location": {
        "latitude": 37.7749,
        "longitude": -122.4194
    }
}
client.add_document("sf-restaurants", document)

# Search for restaurants near a location
results = client.search(
    "sf-restaurants", 
    "italian", 
    geo_search={
        "latitude": 37.7899,
        "longitude": -122.4000,
        "radius_km": 5.0  # Find results within 5km
    }
)

# Display results with distances
for result in results["results"]:
    print(f"{result['id']} - Distance: {result.get('distance_km', 'N/A')} km")
```

## Advanced Usage - Asynchronous Client

The asynchronous client allows for concurrent operations and is suitable for high-throughput applications:

```python
import asyncio
from memsplora_async_client import MemSploraAsyncClient

async def main():
    # Use the client as a context manager
    async with MemSploraAsyncClient("http://localhost:3000") as client:
        # Create a collection
        await client.create_collection(
            "my-collection",
            "My First Collection",
            "A collection for testing the client"
        )
        
        # Add documents concurrently
        documents = [
            {
                "id": f"doc-{i}",
                "content": f"This is test document {i}.",
                "metadata": {"index": i}
            }
            for i in range(1, 6)
        ]
        
        # Execute tasks concurrently
        tasks = [
            client.add_document("my-collection", doc)
            for doc in documents
        ]
        await asyncio.gather(*tasks)
        
        # Search with advanced features
        results = await client.advanced_search(
            "my-collection",
            "test document",
            top_k=5,
            metadata_filter={"index": 3},
            deduplicate=True,
            merge_chunks=True
        )
        print(f"Found {len(results['results'])} results")

# Run the async function
asyncio.run(main())
```

## Domain-Specific Models

You can create collections that use different embedding models:

```python
# Create a collection with a domain-specific model
client.create_collection(
    "medical-docs",
    "Medical Documentation",
    "Medical records and literature",
    model_name="medical-splade-model"  # Custom model name
)
```

## Command Line Usage

The MemSplora CLI provides a command-line interface to the API:

```bash
# List collections
python memsplora_cli.py --url http://localhost:3000 collections list

# Create a collection
python memsplora_cli.py collections create my-collection "My Collection" --description "A test collection" --model-name "custom-model"

# Add a document (from a JSON file)
python memsplora_cli.py documents add my-collection document.json

# Basic search
python memsplora_cli.py search "test query" --collection-id my-collection --top-k 5

# Advanced search
python memsplora_cli.py search "test query" --collection-id my-collection --mode advanced --min-score 0.3

# Geo-spatial search
python memsplora_cli.py search "restaurant" --collection-id places --latitude 37.7749 --longitude -122.4194 --radius-km 2.0
```

## Example Scripts

Check out the example scripts in the `examples` directory for more detailed usage:

- `basic_usage.py`: Basic usage of the synchronous client
- `batch_operations.py`: Batch operations with documents
- `advanced_search.py`: Advanced search features
- `async_client_example.py`: Using the asynchronous client
- `geo_spatial_search.py`: Location-based search capabilities
- `domain_specific_models.py`: Using custom embedding models

Run an example:

```bash
cd client
python examples/geo_spatial_search.py
```

## Next Steps

For more detailed information:

- Read the full [API documentation](./README.md)
- Explore the example scripts in the `examples` directory
- Review the client source code for more implementation details

## Common Operations

### Collection Management

```python
# List collections
collections = client.list_collections()

# Get collection details
collection = client.get_collection("my-collection")

# Get collection statistics
stats = client.get_collection_stats("my-collection")

# Delete a collection
client.delete_collection("my-collection")
```

### Document Management

```python
# Add a document
client.add_document("my-collection", {
    "id": "doc-001",
    "content": "Document content",
    "metadata": {"key": "value"}
})

# Add a document with location
client.add_document("places", {
    "id": "place-001",
    "content": "Eiffel Tower",
    "location": {"latitude": 48.8584, "longitude": 2.2945}
})

# Add documents in batch
client.batch_add_documents("my-collection", [
    {"id": "doc-001", "content": "First document"},
    {"id": "doc-002", "content": "Second document"}
])

# Get a document
document = client.get_document("my-collection", "doc-001")

# Delete a document
client.delete_document("my-collection", "doc-001")
```

### Search Operations

```python
# Basic search
results = client.search("my-collection", "search query", top_k=5)

# Search with metadata filter
results = client.search(
    "my-collection",
    "search query",
    metadata_filter={"category": "article"}
)

# Geo-spatial search
results = client.search(
    "places",
    "landmark",
    geo_search={
        "latitude": 48.8584,  # Paris
        "longitude": 2.2945,
        "radius_km": 1.0
    }
)

# Advanced search with all options
results = client.advanced_search(
    "my-collection",
    "search query",
    top_k=10,
    min_score=0.5,
    metadata_filter={"category": "article"},
    deduplicate=True,
    merge_chunks=True,
    geo_search={
        "latitude": 48.8584,
        "longitude": 2.2945,
        "radius_km": 1.0
    }
)

# Search across all collections
results = client.search_all("search query")
```
