# Document Chunking and Deduplication

This guide explains the chunking and deduplication features of the SPLADE Content Server.

## Document Chunking

When working with large documents, the server automatically chunks them into smaller pieces for better processing. This
is essential because:

1. **Token Limitations**: SPLADE models have a maximum token limit (512 tokens by default)
2. **Relevance Precision**: Smaller chunks capture specific topics better than large documents
3. **Performance**: Processing smaller chunks is more efficient

### How Chunking Works

The server includes an automatic document chunking system that:

1. Estimates if a document exceeds the token limit
2. Splits documents at semantic boundaries (paragraphs)
3. Maintains context with overlapping text between chunks
4. Preserves document relationships through metadata

### Chunk Metadata

Each chunk contains metadata that tracks its relationship to the original document:

```json
{
  "id": "original-doc-id_chunk_0",
  "content": "Chunk content...",
  "metadata": {
    "original_document_id": "original-doc-id",
    "chunk_index": 0,
    "is_chunk": true,
    "other_metadata": "Preserved from original document"
  }
}
```

## Deduplication and Score Thresholding

When searching, you may receive multiple chunks from the same document. The advanced search endpoint provides mechanisms
to handle this:

### Score Thresholding

You can filter out low-relevance results using the `min_score` parameter:

```
GET /advanced-search/my-collection?query=search&min_score=0.3
```

This ensures that only results with a similarity score ≥ 0.3 are returned.

### Deduplication

The deduplication feature removes redundant chunks from the same document:

```
GET /advanced-search/my-collection?query=search&deduplicate=true
```

When enabled:

- Only the highest-scoring chunk from each original document is kept
- Redundant information is eliminated from search results

### Chunk Merging

For a more comprehensive view, you can merge chunks from the same document:

```
GET /advanced-search/my-collection?query=search&merge_chunks=true
```

This combines content from different chunks of the same document, providing a more complete context.

## API Parameters

The advanced search endpoint (`/advanced-search/{collection_id}`) supports:

| Parameter       | Type        | Default    | Description                                    |
|-----------------|-------------|------------|------------------------------------------------|
| query           | string      | (required) | The search query                               |
| top_k           | integer     | 10         | Number of results to return                    |
| min_score       | float       | 0.3        | Minimum similarity score (0.0-1.0)             |
| deduplicate     | boolean     | true       | Remove duplicate chunks from the same document |
| merge_chunks    | boolean     | true       | Merge content from chunks of the same document |
| metadata_filter | JSON string | null       | Filter results by metadata fields              |

## Examples

### Basic Search with Score Threshold

```http
GET /advanced-search/my-collection?query=neural%20networks&min_score=0.4
```

### Search with Deduplication and Merging

```http
GET /advanced-search/my-collection?query=machine%20learning&deduplicate=true&merge_chunks=true
```

### Search with Metadata Filtering

```http
GET /advanced-search/my-collection?query=python&metadata_filter=%7B%22category%22%3A%22programming%22%7D
```

## Best Practices

1. **Start with a higher min_score (0.3-0.4)**: This filters out noise while keeping relevant results
2. **Use deduplication for most searches**: This improves result diversity
3. **Enable merge_chunks for comprehensive answers**: This provides more context from large documents
4. **Adjust parameters based on content type**: Technical content may need lower thresholds than general content
