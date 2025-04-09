# SPLADE Content Server

An in-memory SPLADE (SParse Lexical AnD Expansion) content server with FAISS integration for efficient semantic search.
This project provides a REST API for managing document collections and performing semantic search across them.

## Features

- **Collection-based Document Management**: Organize documents into separate collections
- **In-Memory Operation**: Keep everything in memory for maximum performance
- **FAISS Integration**: Efficient similarity search using Facebook AI Similarity Search
- **Disk Persistence**: Automatic persistence of changes to disk
- **REST API**: Clean API for document management and search operations
- **Metadata Filtering**: Filter search results by document metadata
- **Automatic Document Chunking**: Split large documents into smaller chunks for better processing
- **Deduplication**: Remove duplicate chunks from search results
- **Score Thresholding**: Filter out low-relevance results based on similarity score
- **Advanced FAISS Indexes**: Support for multiple FAISS index types for optimized search performance
- **Soft Deletion**: Efficient document removal with delayed index rebuilding
- **Large Document Handling**: Special processing for extremely large documents with tables
- **Geo-Spatial Search**: Find documents based on geographical proximity
- **Domain-Specific Models**: Support for using different SPLADE models for different collections

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/splade_rest_api.git
cd splade_rest_api
```

2. Install the requirements:

```bash
pip install -r requirements.txt
```

3. Download a pre-trained SPLADE model or use your own fine-tuned model. The server can automatically download the
   `prithivida/Splade_PP_en_v2` model from Hugging Face on first run.

## Configuration

You can configure the application using environment variables or a `.env` file:

### Core Settings

- `SPLADE_MODEL_NAME`: Name of the model to use (default: `Splade_PP_en_v2`)
- `SPLADE_MODEL_DIR`: Directory template for models (default: `./models/{model_name}`)
- `SPLADE_MODEL_HF_ID`: Hugging Face model ID to download if model directory is empty (default:
  `prithivida/Splade_PP_en_v2`)
- `SPLADE_AUTO_DOWNLOAD_MODEL`: Whether to automatically download the model if not found (default: `true`)
- `SPLADE_MAX_LENGTH`: Maximum sequence length for encoding (default: `512`)
- `SPLADE_DATA_DIR`: Directory for storing data (default: `app/data`)
- `SPLADE_DEFAULT_TOP_K`: Default number of search results (default: `10`)

### Performance Settings

- `SPLADE_FAISS_INDEX_TYPE`: FAISS index type - "flat", "ivf", or "hnsw" (default: `flat`)
- `SPLADE_FAISS_NLIST`: Number of clusters for IVF indexes (default: `100`)
- `SPLADE_FAISS_HNSW_M`: Number of connections for HNSW graph (default: `32`)
- `SPLADE_FAISS_SEARCH_NPROBE`: Number of clusters to search for IVF (default: `10`)
- `SPLADE_SOFT_DELETE_ENABLED`: Enable soft deletion for documents (default: `true`)
- `SPLADE_INDEX_REBUILD_THRESHOLD`: Rebuild index after this many deletions (default: `100`)

### Chunking Settings

- `SPLADE_MAX_CHUNK_SIZE`: Maximum tokens per chunk (default: `500`)
- `SPLADE_TABLE_CHUNK_SIZE`: Maximum tokens for table chunks (default: `1000`)
- `SPLADE_CHUNK_OVERLAP`: Overlap tokens between chunks (default: `50`)

## Running the Server

To start the server:

```bash
python main.py
```

The API will be available at `http://localhost:8000`.

You can also use Uvicorn directly:

```bash
uvicorn app.api.server:app --host 0.0.0.0 --port 8000 --reload
```

## API Documentation

API documentation is automatically generated and available at `http://localhost:8000/docs`.

### Main Endpoints

#### Collections

- `GET /collections`: List all collections
- `GET /collections/{collection_id}`: Get collection details
- `GET /collections/{collection_id}/stats`: Get collection statistics
- `POST /collections`: Create a new collection
- `DELETE /collections/{collection_id}`: Delete a collection

#### Documents

- `POST /documents/{collection_id}`: Add a document to a collection
- `POST /documents/{collection_id}/batch`: Add multiple documents to a collection
- `GET /documents/{collection_id}/{document_id}`: Get a document
- `DELETE /documents/{collection_id}/{document_id}`: Delete a document

#### Search

- `GET /search/{collection_id}`: Search in a specific collection
- `GET /search`: Search across all collections

#### Advanced Search

- `GET /advanced-search/{collection_id}`: Search in a specific collection with chunking and deduplication
- `GET /advanced-search`: Search across all collections with chunking and deduplication

## Example Usage

### Create a Collection

```bash
# Basic collection
curl -X POST "http://localhost:8000/collections" \
     -H "Content-Type: application/json" \
     -d '{"id": "technical-docs", "name": "Technical Documentation", "description": "Technical documentation for our products"}'

# Collection with domain-specific model
curl -X POST "http://localhost:8000/collections" \
     -H "Content-Type: application/json" \
     -d '{"id": "legal-docs", "name": "Legal Documentation", "description": "Legal contracts and documents", "model_name": "legal-splade"}'
```

### Add a Document

```bash
curl -X POST "http://localhost:8000/documents/technical-docs" \
     -H "Content-Type: application/json" \
     -d '{"id": "doc-001", "content": "SPLADE is a sparse lexical model for information retrieval", "metadata": {"category": "AI", "author": "John Doe"}}'
```

### Search for Documents

```bash
curl -X GET "http://localhost:8000/search/technical-docs?query=sparse%20lexical%20retrieval&top_k=5"
```

### Search with Metadata Filtering

```bash
curl -X GET "http://localhost:8000/search/technical-docs?query=sparse%20lexical%20retrieval&top_k=5&metadata_filter=%7B%22category%22%3A%22AI%22%7D"
```

## Performance Optimizations

### FAISS Index Types

The system supports different FAISS index types to optimize search performance:

- **Flat**: Exact search with inner product similarity. Best for smaller collections (<100K documents) or when perfect
  accuracy is required.
- **IVF**: Inverted file structure with approximate search. 10-100x faster than Flat for large collections (100K-10M
  documents).
- **HNSW**: Hierarchical Navigable Small World graphs. Fastest search times for very large collections (1M+ documents).

### Soft Deletion

For improved performance when removing documents:

- Documents are marked as "deleted" but physically remain in the index
- Deleted documents are filtered out during search
- The index is rebuilt after a configurable number of deletions
- Greatly reduces the cost of document deletions

### Large Document Handling

Special handling for extremely large documents:

- Hierarchical document segmentation for very large documents
- Special table handling to preserve their structure
- Adaptive chunking strategies based on content type
- Optimized for documents of any size, including 140+ page documents

### Domain-Specific Models

Support for domain-specific SPLADE models:

- Assign different models to different collections based on domain needs
- Models are loaded dynamically and cached for performance
- Each collection can use either the default model or a domain-specific model
- Queries are automatically encoded with the appropriate model per collection

For example, to create a collection with a domain-specific model:

```bash
curl -X POST "http://localhost:8000/collections" \
     -H "Content-Type: application/json" \
     -d '{"id": "medical-docs", "name": "Medical Documentation", "description": "Medical records and reports", "model_name": "medical-splade-model"}'
```

This allows for more accurate search within specific domains while maintaining flexibility across your entire content
library.

## Additional Documentation

For more detailed information, see the documentation in the `docs/` directory:

- [Document Chunking and Deduplication](docs/chunking_and_deduplication.md)
- [Large Document Handling](docs/large_document_handling.md)
- [Performance Optimizations](docs/performance_optimizations.md)
- [Geo-Spatial Search](docs/geo_spatial_search.md)
- [Domain-Specific Models](docs/domain_specific_models.md)

## License

[MIT License](LICENSE)
