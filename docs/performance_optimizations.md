# SPLADE Performance Optimizations

This document outlines performance optimizations for the SPLADE Content Server, with a focus on handling large
collections efficiently.

## FAISS Index Types

The system now supports different FAISS index types to optimize search performance for collections of different sizes:

### Available Index Types

| Index Type         | Description                                     | Best For                                                                            | Trade-offs                                                                                                                 |
|--------------------|-------------------------------------------------|-------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------|
| **Flat** (default) | Exact search with inner product similarity      | Small to medium collections (<100K docs)<br>Applications requiring perfect accuracy | - Most accurate<br>- Slowest for large collections<br>- Linear scaling with collection size                                |
| **IVF**            | Inverted file structure with approximate search | Medium to large collections (100K-10M docs)<br>Balanced accuracy/speed needs        | - Requires initial training<br>- 10-100x faster than Flat<br>- Small accuracy trade-off<br>- Configurable search precision |
| **HNSW**           | Hierarchical Navigable Small World graphs       | Very large collections (1M+ docs)<br>Speed-critical applications                    | - Fastest search times<br>- Better accuracy than IVF<br>- Most memory-intensive<br>- Index construction is slower          |

### Configuration

```python
# In app/core/config.py or .env file
SPLADE_FAISS_INDEX_TYPE="flat"  # Options: "flat", "ivf", "hnsw"
SPLADE_FAISS_NLIST=100          # Number of clusters for IVF
SPLADE_FAISS_HNSW_M=32          # Number of connections for HNSW
SPLADE_FAISS_SEARCH_NPROBE=10   # Number of clusters to search for IVF
```

### Performance Comparison

| Index Type | 10K Documents | 100K Documents | 1M Documents |
|------------|---------------|----------------|--------------|
| **Flat**   | ~10ms         | ~100ms         | ~1000ms      |
| **IVF**    | ~5ms          | ~20ms          | ~50ms        |
| **HNSW**   | ~2ms          | ~5ms           | ~10ms        |

*Note: Actual performance will vary based on hardware, vector dimensions, and configuration.*

## Soft Deletion

To improve performance when removing documents, the system now supports soft deletion:

### How It Works

1. When a document is deleted, it's marked as "deleted" in metadata
2. The document remains in the index but is filtered out during search
3. After a configurable number of deletions, the index is rebuilt automatically
4. During rebuild, soft-deleted documents are physically removed

### Benefits

- **Much Faster Deletion**: Immediate operation instead of rebuilding the index
- **Amortized Cost**: Index rebuilding happens less frequently
- **Improved User Experience**: Deletion appears instant to users

### Configuration

```python
# In app/core/config.py or .env file
SPLADE_SOFT_DELETE_ENABLED=true       # Enable/disable soft deletion
SPLADE_INDEX_REBUILD_THRESHOLD=100    # Rebuild after this many deletions
```

## Implementation Details

### FAISS Factory Pattern

The system uses a factory pattern to create the appropriate FAISS index:

```python
# Create index based on configuration
index = FAISSIndexFactory.create_index(vocab_size, training_vectors)

# Convert between index types if needed
new_index = FAISSIndexFactory.convert_to_index_type(old_index, target_type, vectors)
```

### Soft Deletion Flow

1. Document marked as deleted (added to `deleted_ids` set)
2. Deletion counter incremented
3. When threshold reached, collection signals for rebuild
4. During rebuild, soft-deleted documents are excluded
5. Deletion tracking is reset after rebuild

## Best Practices

### Choosing the Right Index Type

- **Flat**: For small collections or when perfect accuracy is required
- **IVF**: For medium-sized collections with balanced accuracy/speed needs
- **HNSW**: For large collections and speed-critical applications

### Soft Deletion Optimization

- Set `INDEX_REBUILD_THRESHOLD` based on collection update patterns:
    - Lower values (e.g., 50-100): More frequent rebuilds, less memory overhead
    - Higher values (e.g., 500-1000): Less frequent rebuilds, potentially more memory used
- Consider disabling for tiny collections where rebuilding is fast

### Memory Considerations

- **Flat** indexes have the smallest memory footprint
- **IVF** adds a small overhead for cluster centroids
- **HNSW** has the largest memory footprint due to graph structures

## Efficient Pagination

The system now supports efficient pagination for search results, optimizing memory usage and performance for large
result sets:

### How It Works

1. Search results can be paginated with `page` and `page_size` parameters
2. The system calculates total result count and total pages
3. Only the requested page of results is returned, reducing response size
4. For optimum performance, the system retrieves more results than needed for accurate counts, but only returns the
   requested page

### Benefits

- **Reduced Memory Usage**: Only the current page of results is processed and returned
- **Faster API Responses**: Smaller response payloads improve transmission times
- **Better User Experience**: Enables efficient navigation of large result sets
- **Accurate Result Counts**: Provides total counts for implementing pagination controls

### Usage Example

```bash
# Get the second page of results with 10 results per page
curl -X GET "http://localhost:8000/search/collection-id?query=search%20term&page=2&page_size=10"
```

### Response Format

```json
{
  "results": [...],  // Only contains the current page of results
  "query_time_ms": 45.67,
  "pagination": {
    "page": 2,
    "page_size": 10,
    "total_results": 45,
    "total_pages": 5
  }
}
```

## Future Improvements

1. **Incremental Updates**: Allow adding/removing vectors without full rebuild for IVF indexes
2. **Dynamic Index Selection**: Automatically switch index types based on collection size
3. **Background Processing**: Move index rebuilding to background tasks
4. **Index Compression**: Implement vector compression for reduced memory usage
5. **Hybrid Indexes**: Support combined approaches (e.g., IVF-HNSW) for optimal performance
6. **Cursor-based Pagination**: Implement cursor-based pagination for more efficient large result set navigation

These optimizations provide significant performance improvements, especially for large collections with frequent
updates and searches.
