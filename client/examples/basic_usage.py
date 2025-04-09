#!/usr/bin/env python3
"""
Example script demonstrating how to use the sync client to
create collections and add documents.
"""

import json
import sys

from memsplora_client import MemSploraClient


def main():
    # Initialize the client
    client = MemSploraClient("http://localhost:3000")

    try:
        # Check if the collection exists
        collection_id = "example-collection"
        try:
            collection = client.get_collection(collection_id)
            print(f"Using existing collection: {collection['name']}")
        except Exception:
            # Create a new collection
            print(f"Creating new collection '{collection_id}'...")
            collection = client.create_collection(
                collection_id,
                "Example Collection",
                "A collection created by the example script"
            )
            print(f"Collection created: {collection['name']}")

        # Add a document
        doc_id = "example-doc-001"
        document = {
            "id": doc_id,
            "content": "This is an example document that demonstrates how to use the MemSplora client.",
            "metadata": {
                "type": "example",
                "created_by": "example_script"
            }
        }

        try:
            result = client.add_document(collection_id, document)
            print(f"Document added: {result}")
        except Exception as e:
            print(f"Could not add document: {e}")

        # Search for the document
        query = "example document"
        print(f"Searching for '{query}'...")
        results = client.search(collection_id, query)

        # Display results
        print(f"Found {len(results['results'])} results in {results['query_time_ms']:.2f}ms")
        for i, result in enumerate(results['results']):
            print(f"Result {i + 1}: {result['id']} (score: {result['score']:.4f})")
            print(f"  Content: {result['content'][:50]}...")
            print(f"  Metadata: {json.dumps(result['metadata'])}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
