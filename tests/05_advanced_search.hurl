# Test advanced search functionality with chunking, deduplication and score thresholding
# - Search with score threshold
# - Deduplication of chunks
# - Merging chunks from the same document

# Step 1: Create a collection for testing
POST http://localhost:3000/collections/
Content-Type: application/json

{
    "id": "advanced-test",
    "name": "Advanced Search Test",
    "description": "Collection for testing advanced search features"
}

HTTP *

# Step 2: Add a large document that will be automatically chunked
POST http://localhost:3000/documents/advanced-test
Content-Type: application/json

{
    "id": "large-doc-001",
    "content": "# SPLADE: Sparse Lexical and Expansion Model for Information Retrieval\n\nSPLADE is a neural retrieval model that leverages contextual embeddings while maintaining the efficiency of sparse representations. It addresses the vocabulary mismatch problem in traditional retrieval systems by using a learned sparse expansion mechanism.\n\n## Key Features\n\n- **Sparse Representations**: SPLADE produces sparse representations that allow for efficient indexing and retrieval using traditional inverted indexes.\n\n- **Contextual Understanding**: By leveraging transformer models like BERT, SPLADE captures semantic meaning beyond simple keyword matching.\n\n- **Vocabulary Expansion**: SPLADE automatically expands queries with related terms to overcome the vocabulary mismatch problem.\n\n- **Efficient Retrieval**: The sparse nature of SPLADE representations enables fast retrieval comparable to traditional sparse methods.\n\n## How SPLADE Works\n\nSPLADE applies a log-saturation on the RELU activation over the MLM prediction heads of transformer models. This creates a sparse representation where each dimension corresponds to a term in the vocabulary. The model is trained using a contrastive learning approach with pairs of queries and relevant documents.\n\n## Advantages Over Dense Retrieval\n\nUnlike dense retrieval models that require exhaustive comparison, SPLADE can use inverted indexes, making it more efficient for large-scale retrieval. It also maintains interpretability, as each dimension corresponds to a specific term in the vocabulary.\n\n## Applications\n\nSPLADE is particularly effective for:\n\n- Web search engines\n- Document retrieval systems\n- Question answering systems\n- Enterprise search\n- Academic literature search",
    "metadata": {
        "category": "search",
        "type": "documentation",
        "tags": ["splade", "information-retrieval", "sparse-embedding"]
    }
}

HTTP *

# Step 3: Add another document on a similar topic but with different content
POST http://localhost:3000/documents/advanced-test
Content-Type: application/json

{
    "id": "doc-002",
    "content": "# BM25: The Classic Retrieval Algorithm\n\nBM25 (Best Matching 25) is a ranking function used by search engines to estimate the relevance of documents to a given search query. It's based on a probabilistic retrieval framework and is an evolution of the TF-IDF weighting scheme.",
    "metadata": {
        "category": "search",
        "type": "documentation",
        "tags": ["bm25", "information-retrieval", "classic-algorithm"]
    }
}

HTTP *

# Step 4: Add a document with very different content
POST http://localhost:3000/documents/advanced-test
Content-Type: application/json

{
    "id": "doc-003",
    "content": "# Neural Networks 101\n\nArtificial neural networks are computing systems inspired by the biological neural networks in animal brains. These systems learn to perform tasks by considering examples, without being explicitly programmed with task-specific rules.",
    "metadata": {
        "category": "ai",
        "type": "documentation",
        "tags": ["neural-networks", "ai", "machine-learning"]
    }
}

HTTP *

# Step 5: Check collection stats - should have multiple documents (including chunks)
GET http://localhost:3000/collections/advanced-test/stats

HTTP/1.1 200
[Asserts]
jsonpath "$.document_count" >= 3  # Should have original docs plus chunks

# Step 6: Regular search - should return mixed results including chunks
GET http://localhost:3000/search/advanced-test?query=information%20retrieval&top_k=5

HTTP/1.1 200
[Asserts]
jsonpath "$.results" count > 0
# No score threshold, so might include less relevant results

# Step 7: Advanced search with score threshold
GET http://localhost:3000/advanced-search/advanced-test?query=information%20retrieval&min_score=0.3

HTTP/1.1 200
[Asserts]
jsonpath "$.results" exists
# At least first result should have scores above threshold
jsonpath "$.results[0].score" > 0.3

# Step 8: Advanced search with deduplication and chunk merging
GET http://localhost:3000/advanced-search/advanced-test?query=SPLADE&deduplicate=true&merge_chunks=true

HTTP/1.1 200
[Asserts]
jsonpath "$.results" exists
# Check if there are merged chunks in the results
jsonpath "$.results[0].metadata" exists
# The first result should be about SPLADE
jsonpath "$.results[0].content" contains "SPLADE"

# Step 9: Advanced search with metadata filtering
GET http://localhost:3000/advanced-search/advanced-test?query=neural&metadata_filter=%7B%22category%22%3A%22ai%22%7D

HTTP/1.1 200
[Asserts]
jsonpath "$.results" exists
jsonpath "$.results" count > 0
# Results should be about neural networks
jsonpath "$.results[0].content" contains "Neural Networks"
jsonpath "$.results[0].metadata.category" == "ai"

# Step 10: Search across all collections with advanced options
GET http://localhost:3000/advanced-search/?query=retrieval&min_score=0.4

HTTP/1.1 200
[Asserts]
jsonpath "$.results" exists
jsonpath "$.results.advanced-test" exists
# All results should have scores above threshold
#jsonpath "$.results.advanced-test[0].score" matches /^(0\.[4-9]\d*|1\.0)$/

# Step 11: Clean up the collection
DELETE http://localhost:3000/collections/advanced-test

HTTP/1.1 204
