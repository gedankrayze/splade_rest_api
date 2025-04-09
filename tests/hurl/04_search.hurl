# Test search functionality
# - Search in specific collection
# - Search with metadata filtering
# - Search across all collections

# Step 1: Search for documents in the test collection
GET http://localhost:3000/search/test-collection?query=splade%20retrieval&top_k=10

HTTP/1.1 200
[Asserts]
jsonpath "$.results" exists
jsonpath "$.query_time_ms" exists
# At least one document should match
jsonpath "$.results" count >= 1
# The first result should be most relevant to SPLADE
jsonpath "$.results[0].content" contains "SPLADE"
jsonpath "$.results[0].score" exists

# Step 2: Search with metadata filter by category
GET http://localhost:3000/search/test-collection?query=vectors&metadata_filter=%7B%22category%22%3A%22Tools%22%7D

HTTP/1.1 200
[Asserts]
jsonpath "$.results" exists
jsonpath "$.results" count >= 1
# The result should be from the "Tools" category
jsonpath "$.results[0].metadata.category" == "Tools"
jsonpath "$.results[0].content" contains "FAISS"

# Step 3: Create a new collection for cross-collection search
POST http://localhost:3000/collections/
Content-Type: application/json

{
    "id": "research-papers",
    "name": "Research Papers",
    "description": "Academic research papers"
}

HTTP/1.1 201

# Step 4: Add documents to the new collection
POST http://localhost:3000/documents/research-papers
Content-Type: application/json

{
    "id": "paper-001",
    "content": "SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking. This paper introduces a new sparse retrieval method.",
    "metadata": {
        "category": "Research",
        "year": 2021,
        "authors": ["Thibault Formal", "Benjamin Piwowarski", "Stéphane Clinchant"]
    }
}

HTTP/1.1 201

# Step 5: Search across all collections
GET http://localhost:3000/search/?query=splade

HTTP/1.1 200
[Asserts]
jsonpath "$.results" exists
jsonpath "$.query_time_ms" exists
# Results should include matches from both collections
jsonpath "$.results.test-collection" exists
jsonpath "$.results.research-papers" exists

# Step 6: Clean up the test by deleting the research papers collection
DELETE http://localhost:3000/collections/research-papers

HTTP/1.1 204
