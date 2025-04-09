#!/usr/bin/env python3
"""
Example usage of the MemSplora Async Client.
This example shows how to perform concurrent searches across collections.
"""

import asyncio
import json
from typing import List

from memsplora_async_client import MemSploraAsyncClient
from memsplora_types import SearchResponse


async def search_multiple_queries(
        client: MemSploraAsyncClient,
        collection_id: str,
        queries: List[str]
) -> List[SearchResponse]:
    """
    Perform multiple search queries concurrently.
    
    Args:
        client: MemSplora async client
        collection_id: Collection to search in
        queries: List of search queries
    
    Returns:
        List of search responses
    """
    # Create a list of search tasks
    search_tasks = [
        client.search(collection_id, query, top_k=5)
        for query in queries
    ]

    # Execute all tasks concurrently
    results = await asyncio.gather(*search_tasks)
    return results


async def main():
    """Main execution function"""
    # API client configuration
    base_url = "http://localhost:3000"

    # List of search queries
    queries = [
        "neural networks",
        "natural language processing",
        "computer vision",
        "reinforcement learning"
    ]

    # Collection to search in
    collection_id = "technical-docs"

    # Create client as async context manager
    async with MemSploraAsyncClient(base_url) as client:
        try:
            # Get available collections
            collections = await client.list_collections()
            print(f"Available collections: {json.dumps(collections, indent=2)}")

            # Check if our collection exists, if not create it
            try:
                collection = await client.get_collection(collection_id)
                print(f"Using existing collection: {collection['name']}")
            except Exception:
                print(f"Collection '{collection_id}' not found, creating...")
                collection = await client.create_collection(
                    collection_id,
                    "Technical Documentation",
                    "Collection for ML/AI technical documentation"
                )
                print(f"Created collection: {collection['name']}")

            # Add a sample document if needed
            try:
                sample_doc = {
                    "id": "neural-networks-intro",
                    "content": "Neural networks are computational models inspired by the human brain...",
                    "metadata": {"category": "AI", "complexity": "beginner"}
                }
                result = await client.add_document(collection_id, sample_doc)
                print(f"Added document: {result}")
            except Exception as e:
                print(f"Note: {str(e)}")

            # Perform concurrent searches
            print(f"\nPerforming {len(queries)} searches concurrently...")
            results = await search_multiple_queries(client, collection_id, queries)

            # Display results
            for i, (query, response) in enumerate(zip(queries, results)):
                print(f"\nQuery {i + 1}: {query}")
                print(f"Found {len(response['results'])} results in {response['query_time_ms']:.2f}ms")

                for j, result in enumerate(response['results']):
                    print(f"  Result {j + 1}: {result['id']} (score: {result['score']:.4f})")

        except Exception as e:
            print(f"Error: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())
