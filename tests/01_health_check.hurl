# Health check test
# Verifies that the server is up and running

# Test the root endpoint - should return server status
GET http://localhost:3000/

HTTP/1.1 200
[Asserts]
jsonpath "$.service" == "SPLADE Content Server"
jsonpath "$.status" exists
jsonpath "$.model" exists
jsonpath "$.model.path" exists
jsonpath "$.model.exists" exists
jsonpath "$.model.loaded" exists
