import argparse
import json
from typing import Optional, Dict, Any

import requests


class MemSploraClient:
    """Client for connecting to the MemSplora advanced search API."""

    def __init__(self, base_url: str, api_key: Optional[str] = None):
        """
        Initialize the search API client.

        Args:
            base_url: Base URL of the API server (e.g., 'https://api.example.com')
            api_key: Optional API key for authentication
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.headers = {}
        if api_key:
            self.headers['Authorization'] = f'Bearer {api_key}'

    def advanced_search(
            self,
            collection_name: str,
            query: str,
            top_k: int = 10,
            min_score: float = 0.3,
            metadata_filter: Optional[Dict[str, Any]] = None,
            deduplicate: bool = True,
            merge_chunks: bool = True
    ) -> Dict[str, Any]:
        """
        Perform an advanced search on the specified collection.

        Args:
            collection_name: Name of the collection to search
            query: Search query text
            top_k: Number of results to return
            min_score: Minimum similarity score threshold (0-1)
            metadata_filter: Dictionary of metadata filters
            deduplicate: Deduplicate results from same document
            merge_chunks: Merge chunks from the same document

        Returns:
            Search results as a dictionary
        """

        # Construct the endpoint URL
        endpoint = f'/advanced-search/{collection_name}'

        # Build query parameters
        params = {
            'query': query,  # The API will handle encoding
            'top_k': top_k,
            'min_score': min_score,
            'deduplicate': deduplicate,
            'merge_chunks': merge_chunks
        }

        # Add metadata filter if provided
        if metadata_filter:
            params['metadata_filter'] = json.dumps(metadata_filter)

        # Make the API request
        url = f'{self.base_url}{endpoint}'
        response = requests.get(url, params=params, headers=self.headers)

        # Check for successful response
        response.raise_for_status()

        # Return the JSON response
        return response.json()


# Example usage
if __name__ == '__main__':
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Search the MemSplora API')
    parser.add_argument('query', help='Search query text')
    parser.add_argument('--collection', '-c', default='your_collection',
                        help='Collection name to search')
    parser.add_argument('--top-k', '-k', type=int, default=20,
                        help='Number of results to return')
    parser.add_argument('--min-score', '-s', type=float, default=0.8,
                        help='Minimum similarity score threshold (0-1)')
    parser.add_argument('--url', '-u', default='http://localhost:3000',
                        help='API server URL')
    parser.add_argument('--api-key', '-a', default='your_api_key_here',
                        help='API key for authentication')

    args = parser.parse_args()

    # Initialize the client
    client = MemSploraClient(args.url, args.api_key)

    # Perform the search with command line arguments
    results = client.advanced_search(
        collection_name=args.collection,
        query=args.query,
        top_k=args.top_k,
        min_score=args.min_score
    )

    # Print the results
    print(f"Found {len(results.get('results', []))} results:")
    print(f"Query time: {results.get('query_time_ms', 'N/A')} ms")

    for i, result in enumerate(results.get('results', []), 1):
        print(f"\n--- Result {i} ---")
        print(f"ID: {result.get('id', 'N/A')}")
        print(f"Score: {result.get('score', 'N/A')}")

        # Print content with a character limit for display
        content = result.get('content', 'N/A')
        if len(content) > 100:
            print(f"Content: {content[:100]}...")
        else:
            print(f"Content: {content}")

        # Handle metadata if present
        if 'metadata' in result:
            print("\nMetadata:")
            metadata = result['metadata']

            # Print source information
            if 'source' in metadata:
                print(f"  Source: {metadata['source']}")
            if 'source_filename' in metadata:
                print(f"  Source filename: {metadata['source_filename']}")

            # Print products as a formatted list if present
            if 'products' in metadata and isinstance(metadata['products'], list):
                print(f"  Products ({len(metadata['products'])}):")
                for product in metadata['products'][:5]:  # Show first 5 products
                    print(f"    - {product}")
                if len(metadata['products']) > 5:
                    print(f"    - ... and {len(metadata['products']) - 5} more")
