#!/usr/bin/env python3
"""
Example script demonstrating how to use the sync client to 
perform batch operations with documents.
"""

import sys

from memsplora_client import MemSploraClient


def main():
    # Initialize the client
    client = MemSploraClient("http://localhost:3000")

    try:
        # Prepare collection
        collection_id = "batch-example"

        # Check if collection exists, create if not
        try:
            client.get_collection(collection_id)
        except Exception:
            print(f"Creating collection '{collection_id}'...")
            client.create_collection(
                collection_id,
                "Batch Example Collection",
                "A collection for batch operation examples"
            )
            print("Collection created successfully")

        # Prepare batch of documents
        documents = [
            {
                "id": f"batch-doc-{i}",
                "content": f"This is batch document #{i} with some sample content for demonstration.",
                "metadata": {
                    "batch_id": "example-batch",
                    "document_number": i,
                    "category": "sample" if i % 2 == 0 else "example"
                }
            }
            for i in range(1, 6)  # Create 5 documents
        ]

        # Add documents in batch
        print(f"Adding {len(documents)} documents in batch...")
        result = client.batch_add_documents(collection_id, documents)
        print(f"Added {result['added_count']} documents successfully")

        # Search with different filters
        queries = [
            {"query": "sample content", "filter": None},
            {"query": "batch document", "filter": {"category": "sample"}},
            {"query": "demonstration", "filter": {"category": "example"}}
        ]

        # Run searches
        for i, query_info in enumerate(queries):
            q = query_info["query"]
            filter_info = query_info["filter"]

            print(f"\nSearch #{i + 1}: '{q}'" +
                  (f" with filter {filter_info}" if filter_info else ""))

            # Perform search
            results = client.search(
                collection_id,
                q,
                top_k=10,
                metadata_filter=filter_info
            )

            # Display results
            print(f"Found {len(results['results'])} results in {results['query_time_ms']:.2f}ms")
            for j, result in enumerate(results['results']):
                print(f"Result {j + 1}: {result['id']} (score: {result['score']:.4f})")
                print(f"  Document #{result['metadata']['document_number']}, " +
                      f"Category: {result['metadata']['category']}")

        # Cleanup example (uncomment if needed)
        # print("\nCleaning up documents...")
        # for doc in documents:
        #     try:
        #         client.delete_document(collection_id, doc["id"])
        #         print(f"Deleted document {doc['id']}")
        #     except Exception as e:
        #         print(f"Error deleting document {doc['id']}: {e}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
