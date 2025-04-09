# MemSplora Client

A comprehensive client library for interacting with the SPLADE Content Server API.

## Features

- **Synchronous and Asynchronous Clients**: Choose between a simple synchronous client or a high-performance
  asynchronous client
- **Comprehensive API Coverage**: Full access to all API endpoints including collections, documents, and search
  operations
- **Geo-Spatial Search**: Support for location-based document retrieval and distance-aware results
- **Advanced Search Options**: Deduplication, chunk merging, and metadata filtering
- **Type Safety**: Strong typing for all API requests and responses
- **CLI Tool**: Command-line interface for common operations
- **Example Scripts**: Ready-to-use example scripts demonstrating common workflows

## Quick Links

- [Quickstart Guide](docs/QUICKSTART.md)
- [Full Documentation](docs/README.md)
- [Example Scripts](examples/)

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Basic Usage

```python
from memsplora_client import MemSploraClient

# Initialize client
client = MemSploraClient("http://localhost:3000")

# Create a collection
client.create_collection("my-collection", "My Collection")

# Add a document
client.add_document("my-collection", {
    "id": "doc-001",
    "content": "This is a test document",
    "metadata": {"category": "test"}
})

# Add a document with location
client.add_document("my-collection", {
    "id": "doc-002",
    "content": "This is a document with location data",
    "metadata": {"category": "geo"},
    "location": {
        "latitude": 37.7749,
        "longitude": -122.4194
    }
})

# Basic search
results = client.search("my-collection", "test document")

# Geo-spatial search
geo_results = client.search(
    "my-collection", 
    "location", 
    geo_search={
        "latitude": 37.7749,
        "longitude": -122.4194,
        "radius_km": 5.0
    }
)

# Advanced search with deduplication and chunk merging
advanced_results = client.advanced_search(
    "my-collection",
    "test query",
    deduplicate=True,
    merge_chunks=True
)
```

For more examples, see the [Quickstart Guide](docs/QUICKSTART.md) and [Full Documentation](docs/README.md).

## CLI Usage

```bash
# List collections
python memsplora_cli.py collections list

# Basic search
python memsplora_cli.py search "test query" --collection-id my-collection

# Geo-spatial search
python memsplora_cli.py search "cafe" --collection-id my-collection --latitude 37.7749 --longitude -122.4194 --radius-km 2.0

# Advanced search with options
python memsplora_cli.py search "test query" --collection-id my-collection --mode advanced --min-score 0.5 --deduplicate --merge-chunks
```

See [CLI Documentation](docs/README.md#command-line-interface) for more details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
