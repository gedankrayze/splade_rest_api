# SPLADE REST API Tests

This directory contains Hurl tests for verifying the functionality of the SPLADE REST API.

## Prerequisites

- [Hurl](https://hurl.dev/) - A command line tool to run HTTP requests and test the response
- The SPLADE REST API server running on port 3000

## Installing Hurl

### macOS

```bash
brew install hurl
```

### Linux

```bash
curl -LO https://github.com/Orange-OpenSource/hurl/releases/download/2.0.1/hurl_2.0.1_amd64.deb
sudo dpkg -i hurl_2.0.1_amd64.deb
```

### Windows

```bash
choco install hurl
```

## Running Tests

1. First, ensure the SPLADE server is running on port 3000:
   ```bash
   cd ~/Desktop/splade_rest_api
   python main.py
   ```

2. Run the tests using the provided script:
   ```bash
   cd ~/Desktop/splade_rest_api
   chmod +x run_tests.sh
   ./run_tests.sh
   ```

3. Or run individual tests manually:
   ```bash
   hurl --test --color tests/01_health_check.hurl
   hurl --test --color tests/02_collections.hurl
   hurl --test --color tests/03_documents.hurl
   hurl --test --color tests/04_search.hurl
   ```

## Test Overview

1. `01_health_check.hurl` - Verifies that the server is up and running
2. `02_collections.hurl` - Tests collection management (create, list, get, delete)
3. `03_documents.hurl` - Tests document management (add, get, batch add, delete)
4. `04_search.hurl` - Tests search functionality (search in collection, with filters, cross-collection)

## Cleaning Up Test Data

The test script automatically cleans up test data before and after running the tests. If you need to manually clean up,
you can use:

```bash
curl -X DELETE http://localhost:3000/collections/test-collection
curl -X DELETE http://localhost:3000/collections/technical-docs
curl -X DELETE http://localhost:3000/collections/research-papers
```
