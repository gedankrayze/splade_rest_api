# Test document management
# - Add document to collection
# - Get document
# - Batch add documents
# - Delete document

# Step 1: Add a document to the test collection
POST http://localhost:3000/documents/test-collection
Content-Type: application/json

{
    "id": "doc-001",
    "content": "SPLADE is a sparse lexical model for information retrieval that leverages contextual embeddings.",
    "metadata": {
        "category": "AI",
        "author": "John Doe",
        "tags": ["splade", "ir", "search"]
    }
}

HTTP/1.1 201
[Asserts]
jsonpath "$.status" == "success"
jsonpath "$.message" exists

# Step 2: Get the document
GET http://localhost:3000/documents/test-collection/doc-001

HTTP/1.1 200
[Asserts]
jsonpath "$.id" == "doc-001"
jsonpath "$.content" exists
jsonpath "$.metadata.category" == "AI"
jsonpath "$.metadata.author" == "John Doe"
jsonpath "$.metadata.tags" count == 3

# Step 3: Verify collection stats - should have 1 document now
GET http://localhost:3000/collections/test-collection/stats

HTTP/1.1 200
[Asserts]
jsonpath "$.document_count" == 1
jsonpath "$.index_size" == 1

# Step 4: Add multiple documents in batch
POST http://localhost:3000/documents/test-collection/batch
Content-Type: application/json

{
    "documents": [
        {
            "id": "doc-002",
            "content": "Hybrid search combines sparse and dense retrieval methods for better results.",
            "metadata": {
                "category": "Search",
                "author": "Jane Smith",
                "tags": ["hybrid", "search", "retrieval"]
            }
        },
        {
            "id": "doc-003",
            "content": "FAISS is a library for efficient similarity search and clustering of dense vectors.",
            "metadata": {
                "category": "Tools",
                "author": "John Doe",
                "tags": ["faiss", "similarity", "vectors"]
            }
        }
    ]
}

HTTP/1.1 201
[Asserts]
jsonpath "$.status" == "success"
jsonpath "$.message" contains "2 documents"

# Step 5: Verify collection stats - should have 3 documents now
GET http://localhost:3000/collections/test-collection/stats

HTTP/1.1 200
[Asserts]
jsonpath "$.document_count" == 3
jsonpath "$.index_size" == 3

# Step 6: Delete document
DELETE http://localhost:3000/documents/test-collection/doc-002

HTTP/1.1 204

# Step 7: Verify document deletion
GET http://localhost:3000/documents/test-collection/doc-002

HTTP/1.1 404

# Step 8: Verify collection stats - should have 2 documents now
GET http://localhost:3000/collections/test-collection/stats

HTTP/1.1 200
[Asserts]
jsonpath "$.document_count" == 2
jsonpath "$.index_size" == 2

# Step 9: Try to add document with the same ID - should return conflict
POST http://localhost:3000/documents/test-collection
Content-Type: application/json

{
    "id": "doc-001",
    "content": "This is a duplicate document.",
    "metadata": {
        "category": "Duplicate"
    }
}

HTTP/1.1 409
[Asserts]
jsonpath "$.detail" contains "already exists"
