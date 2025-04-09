# Domain-Specific SPLADE Models

The SPLADE Content Server supports using different models for different collections, allowing for domain-optimized
search experiences.

## Why Use Domain-Specific Models?

Different content domains have their own unique vocabularies, semantic relationships, and search requirements. By using
domain-specific SPLADE models, you can:

1. **Improve Relevance**: Get better search results for specialized content
2. **Handle Specialized Vocabulary**: Properly encode and understand domain-specific terminology
3. **Optimize for Different Use Cases**: Use different model trade-offs for different collections

## How Domain-Specific Models Work

Each collection can be associated with a specific SPLADE model:

1. **Model Management**: Models are stored in the `models/{model_name}` directory
2. **Dynamic Loading**: Models are loaded on-demand when needed
3. **Caching**: Models are cached in memory to avoid reloading
4. **Per-Collection Configuration**: Each collection can use a different model

## Creating a Collection with a Domain-Specific Model

When creating a collection, you can specify a model name:

```python
# Using the Python client
splade_service.create_collection(
    collection_id="medical-docs",
    name="Medical Documentation",
    description="Medical records and reports",
    model_name="medical-splade"
)
```

Using the REST API:

```bash
curl -X POST "http://localhost:8000/collections" \
     -H "Content-Type: application/json" \
     -d '{
       "id": "medical-docs", 
       "name": "Medical Documentation", 
       "description": "Medical records and reports", 
       "model_name": "medical-splade"
     }'
```

If you don't specify a model name, the collection will use the default model (specified by `SPLADE_MODEL_NAME` in the
configuration).

## Using Domain-Specific Models

Once a collection is created with a specific model, all operations on that collection will automatically use that model:

1. **Document Indexing**: Documents added to the collection are encoded with the collection's model
2. **Search Queries**: Search queries for the collection are encoded with the same model
3. **Index Rebuilding**: When the collection's index is rebuilt, the correct model is used

This ensures consistency within each collection while allowing for domain-specific optimization.

## Available Models

The system comes with one default model:

- **Splade_PP_en_v2**: General-purpose English SPLADE model (default)

To add your own domain-specific models:

1. Place your model in the `models/{model_name}` directory
2. Ensure it has the standard Hugging Face model structure (config.json, model files, tokenizer files)
3. Assign it to collections as needed

You can also have the system download models from Hugging Face by setting the model directory to a non-existent path and
enabling `AUTO_DOWNLOAD_MODEL`.

## Implementation Details

### Model Loading and Caching

Models are loaded lazily when they're first needed and then cached for future use:

- The first time a collection with a specific model is accessed, the model is loaded into memory
- Subsequent operations on collections with the same model reuse the loaded model
- If a model can't be loaded, the system falls back to the default model

### Collection Persistence

The association between collections and models is persistent:

- Model information is saved with collection data
- When the server restarts, collections maintain their model associations
- You can change a collection's model by recreating it with a new model name

## Performance Considerations

When using multiple models, be aware of these performance implications:

1. **Memory Usage**: Each loaded model consumes memory (typically 300MB-1GB per model)
2. **Initial Loading**: The first operation on a collection may be slower as the model is loaded
3. **GPU Memory**: When using GPU acceleration, multiple models will share the available GPU memory

## Best Practices

1. **Group Similar Content**: Create collections with similar content types and assign them the same domain-specific
   model
2. **Test Model Performance**: Compare search results with different models to find the best fit for each domain
3. **Balance Specificity**: Very domain-specific models may perform poorly on general content
4. **Monitor Memory Usage**: If using many models, monitor system memory usage
5. **Document Your Models**: Keep track of which models are used for which collections and why

## Example Use Cases

### Medical Content

```bash
curl -X POST "http://localhost:8000/collections" \
     -H "Content-Type: application/json" \
     -d '{
       "id": "medical-records", 
       "name": "Medical Records", 
       "description": "Patient records and medical reports", 
       "model_name": "medical-splade"
     }'
```

### Legal Documents

```bash
curl -X POST "http://localhost:8000/collections" \
     -H "Content-Type: application/json" \
     -d '{
       "id": "legal-contracts", 
       "name": "Legal Contracts", 
       "description": "Legal agreements and contracts", 
       "model_name": "legal-splade"
     }'
```

### Technical Documentation

```bash
curl -X POST "http://localhost:8000/collections" \
     -H "Content-Type: application/json" \
     -d '{
       "id": "api-docs", 
       "name": "API Documentation", 
       "description": "Technical API reference documentation", 
       "model_name": "technical-splade"
     }'
```
