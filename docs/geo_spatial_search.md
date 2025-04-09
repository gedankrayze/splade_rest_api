# Geo-Spatial Search

The SPLADE Content Server now supports geo-spatial search capabilities, allowing you to find documents based on their
geographic proximity to a specified location.

## Overview

Geo-spatial search enables you to:

1. **Store location data** with your documents
2. **Search by proximity** to a geographic coordinate
3. **Filter results** based on distance
4. **Combine** location-based filters with text queries and metadata filters

This functionality is particularly useful for applications involving:

- Points of interest
- Location-based services
- Geographic data analysis
- Local search

## Adding Location Data to Documents

When adding documents to a collection, you can include geographic coordinates:

```python
from app.models.schema import Document, GeoCoordinates

# Create a document with location
document = Document(
    id="restaurant-123",
    content="Delicious Italian restaurant in San Francisco",
    metadata={"category": "restaurant", "cuisine": "Italian"},
    location=GeoCoordinates(
        latitude=37.7749,
        longitude=-122.4194
    )
)

# Add to collection
splade_service.add_document("restaurants", document)
```

### API Example

Using the REST API:

```json
POST /documents/restaurants
{
  "id": "restaurant-123",
  "content": "Delicious Italian restaurant in San Francisco",
  "metadata": {"category": "restaurant", "cuisine": "Italian"},
  "location": {
    "latitude": 37.7749,
    "longitude": -122.4194
  }
}
```

## Performing Geo-Spatial Searches

### Basic Geo Search

To search for documents near a specific location:

```python
# Search for restaurants within 5km of San Francisco
results, query_time = splade_service.search(
    "restaurants",
    "Italian food",
    geo_filter={
        "latitude": 37.7749,
        "longitude": -122.4194,
        "radius_km": 5.0
    }
)
```

### REST API Example

```
GET /search/restaurants?query=Italian+food&latitude=37.7749&longitude=-122.4194&radius_km=5.0
```

### Combining with Metadata Filters

You can combine geo-spatial search with metadata filtering:

```python
# Search for Italian restaurants within 5km of San Francisco that are open now
results, query_time = splade_service.search(
    "restaurants",
    "Italian food",
    filter_metadata={"cuisine": "Italian", "open_now": True},
    geo_filter={
        "latitude": 37.7749,
        "longitude": -122.4194,
        "radius_km": 5.0
    }
)
```

### REST API Example

```
GET /search/restaurants?query=Italian+food&latitude=37.7749&longitude=-122.4194&radius_km=5.0&metadata_filter=%7B%22cuisine%22%3A%22Italian%22%2C%22open_now%22%3Atrue%7D
```

## Understanding Search Results

Search results for geo-spatial queries include distance information:

```json
{
  "results": [
    {
      "id": "restaurant-123",
      "content": "Delicious Italian restaurant in San Francisco",
      "metadata": {"category": "restaurant", "cuisine": "Italian"},
      "score": 0.85,
      "location": {
        "latitude": 37.7749,
        "longitude": -122.4194
      },
      "distance_km": 1.2
    },
    ...
  ],
  "query_time_ms": 12.5
}
```

The `distance_km` field indicates the distance from the query point to the document's location.

## Advanced Geo-Spatial Search

For more sophisticated search needs, you can use the advanced search endpoints with geo-spatial parameters:

```
GET /advanced-search/restaurants?query=Italian+pasta&latitude=37.7749&longitude=-122.4194&radius_km=5.0&min_score=0.3&deduplicate=true
```

This combines geo-spatial filtering with other advanced features like score thresholding and deduplication.

## Implementation Details

### Spatial Indexing

The SPLADE Content Server uses a grid-based spatial index to efficiently locate documents within a geographic radius:

1. Earth's surface is partitioned into a grid based on latitude and longitude
2. Documents are assigned to grid cells based on their coordinates
3. Searches find documents in cells that intersect with the search radius
4. Final results are filtered using the Haversine formula for precise distance calculation

### Performance Considerations

- **Grid Precision**: Controlled by the `GEO_INDEX_PRECISION` setting (default: 6)
- **Search Radius**: Larger radii require checking more grid cells, which can affect performance
- **Index Memory**: The spatial index is kept in memory alongside the FAISS vector index
- **Combined Filtering**: When combining geo-spatial search with metadata filters, the geo filter is applied first

## Configuration

Geo-spatial search can be configured through the following settings:

```python
# In app/core/config.py or .env file
SPLADE_GEO_INDEX_PRECISION = 6       # Grid precision (higher means smaller cells)
SPLADE_GEO_DEFAULT_RADIUS_KM = 10.0  # Default search radius when not specified
```

## Code Examples

### Python Client Example

```python
from app.core.splade_service import splade_service
from app.models.schema import Document, GeoCoordinates

# Add documents with location
coffee_shops = [
    Document(
        id="coffee-1",
        content="Artisan coffee shop with great pastries",
        metadata={"category": "cafe", "rating": 4.5},
        location=GeoCoordinates(latitude=37.7749, longitude=-122.4194)
    ),
    Document(
        id="coffee-2",
        content="Cozy cafe with free wifi and amazing espresso",
        metadata={"category": "cafe", "rating": 4.8},
        location=GeoCoordinates(latitude=37.7746, longitude=-122.4174)
    )
]

# Add to collection
splade_service.batch_add_documents("cafes", coffee_shops)

# Search for coffee shops within 500 meters
results, query_time = splade_service.search(
    "cafes",
    "coffee wifi",
    top_k=5,
    geo_filter={
        "latitude": 37.7750,
        "longitude": -122.4183,
        "radius_km": 0.5
    }
)

# Print results with distances
for result in results:
    print(f"{result['id']} - {result['distance_km']:.2f}km - Score: {result['score']:.2f}")
    print(f"  {result['content'][:50]}...")
```

### HTTP API Examples

#### Search for documents near a location:

```bash
curl -X GET "http://localhost:8000/search/cafes?query=coffee&latitude=37.7750&longitude=-122.4183&radius_km=0.5"
```

#### Search with combined filters:

```bash
curl -X GET "http://localhost:8000/search/cafes?query=coffee&latitude=37.7750&longitude=-122.4183&radius_km=0.5&metadata_filter=%7B%22rating%22%3A4.8%7D"
```

#### Advanced search with geo-spatial parameters:

```bash
curl -X GET "http://localhost:8000/advanced-search/cafes?query=coffee&latitude=37.7750&longitude=-122.4183&radius_km=0.5&min_score=0.3&deduplicate=true"
```

## Best Practices

1. **Index Precision**: Adjust `GEO_INDEX_PRECISION` based on your typical search radius and data density.
2. **Radius Optimization**: Use the smallest radius that meets your needs to improve performance.
3. **Combined Filtering**: When combining geo-spatial and metadata filters, make the metadata filter as selective as
   possible.
4. **Location Validation**: Always validate coordinate data before storing (should be within valid ranges).
5. **Distance Relevance**: Consider combining the text relevance score with distance for ranking results.
