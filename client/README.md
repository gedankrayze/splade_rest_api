# MemSplora Client

A comprehensive client library for interacting with the SPLADE Content Server API.

## Features

- **Synchronous and Asynchronous Clients**: Choose between a simple synchronous client or a high-performance
  asynchronous client
- **Comprehensive API Coverage**: Full access to all API endpoints including collections, documents, and search
  operations
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

# Search
results = client.search("my-collection", "test document")
```

For more examples, see the [Quickstart Guide](docs/QUICKSTART.md) and [Full Documentation](docs/README.md).

## CLI Usage

```bash
# List collections
python memsplora_cli.py collections list

# Search
python memsplora_cli.py search "test query" --collection-id my-collection
```

See [CLI Documentation](docs/README.md#command-line-interface) for more details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
