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

## Installation

1. Clone the repository:

```bash
git clone https://github.com/gedankrayze/splade_rest_api.git
cd splade_rest_api
```

2. Install the requirements:

```bash
pip install -r requirements.txt
```

3. Download a pre-trained SPLADE model or use your own fine-tuned model.

## Configuration

You can configure the application using environment variables or a `.env` file:

- `SPLADE_MODEL_DIR`: Directory containing the SPLADE model (default: `./fine_tuned_splade`)
- `SPLADE_MAX_LENGTH`: Maximum sequence length for encoding (default: `512`)
- `SPLADE_DATA_DIR`: Directory for storing data (default: `app/data`)
- `SPLADE_DEFAULT_TOP_K`: Default number of search results (default: `10`)

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

## Example Usage

### Create a Collection

```bash
curl -X POST "http://localhost:8000/collections" \
     -H "Content-Type: application/json" \
     -d '{"id": "technical-docs", "name": "Technical Documentation", "description": "Technical documentation for our products"}'
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

## Performance Considerations

- The SPLADE model runs most efficiently on a GPU.
- For very large collections, consider using more sophisticated FAISS indexes like `IndexIVFFlat` or `IndexHNSWFlat`.
- Document removal currently requires rebuilding the entire collection index, which can be expensive for large
  collections.

## License

[MIT License](LICENSE)
