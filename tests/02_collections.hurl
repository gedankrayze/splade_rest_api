# Test collection management
# - Create collection
# - List collections 
# - Get collection details
# - Get collection stats
# - Delete collection

# Step 1: Get initial collections list - should be empty at first
GET http://localhost:3000/collections/

HTTP/1.1 200
[Asserts]
jsonpath "$.collections" exists
jsonpath "$.collections" count == 0

# Step 2: Create a new test collection
POST http://localhost:3000/collections/
Content-Type: application/json

{
    "id": "test-collection",
    "name": "Test Collection",
    "description": "A collection for testing purposes"
}

HTTP/1.1 201
[Asserts]
jsonpath "$.id" == "test-collection"
jsonpath "$.name" == "Test Collection"
jsonpath "$.description" == "A collection for testing purposes"

# Step 3: Create another collection
POST http://localhost:3000/collections/
Content-Type: application/json

{
    "id": "technical-docs",
    "name": "Technical Documentation",
    "description": "Technical documentation collection"
}

HTTP/1.1 201

# Step 4: List collections - should have 2 collections now
GET http://localhost:3000/collections/

HTTP/1.1 200
[Asserts]
jsonpath "$.collections" count == 2
jsonpath "$.collections[*].id" includes "test-collection"
jsonpath "$.collections[*].id" includes "technical-docs"

# Step 5: Get collection details
GET http://localhost:3000/collections/test-collection

HTTP/1.1 200
[Asserts]
jsonpath "$.id" == "test-collection"
jsonpath "$.name" == "Test Collection"
jsonpath "$.description" == "A collection for testing purposes"

# Step 6: Get collection stats - should be empty initially
GET http://localhost:3000/collections/test-collection/stats

HTTP/1.1 200
[Asserts]
jsonpath "$.document_count" == 0
jsonpath "$.index_size" == 0

# Step 7: Try to get non-existent collection - should return 404
GET http://localhost:3000/collections/non-existent-collection

HTTP/1.1 404
[Asserts]
jsonpath "$.detail" exists

# Step 8: Delete collection
DELETE http://localhost:3000/collections/technical-docs

HTTP/1.1 204

# Step 9: Verify deletion - list should only have 1 collection now
GET http://localhost:3000/collections/

HTTP/1.1 200
[Asserts]
jsonpath "$.collections" count == 1
jsonpath "$.collections[0].id" == "test-collection"
