#!/usr/bin/env python3
"""
Example script demonstrating how to use advanced search features
like chunking, deduplication, and metadata filtering.
"""

import sys

from memsplora_client import MemSploraClient


def main():
    # Initialize the client
    client = MemSploraClient("http://localhost:3000")

    try:
        # Prepare collection
        collection_id = "advanced-search-example"

        # Check if collection exists, create if not
        try:
            client.get_collection(collection_id)
        except Exception:
            print(f"Creating collection '{collection_id}'...")
            client.create_collection(
                collection_id,
                "Advanced Search Examples",
                "A collection for demonstrating advanced search capabilities"
            )
            print("Collection created successfully")

        # Add some documents with long content that will be chunked
        documents = [
            {
                "id": "long-article-1",
                "content": """
                # Introduction to Neural Networks
                
                Neural networks are computational models inspired by the human brain's structure and function. 
                They are used in machine learning to recognize patterns and solve complex problems.
                
                ## Basic Architecture
                
                A neural network consists of layers of interconnected nodes or "neurons." Each connection 
                has a weight that adjusts as the network learns. The network typically has an input layer, 
                one or more hidden layers, and an output layer.
                
                ## How Neural Networks Learn
                
                Neural networks learn through a process called training. During training, the network 
                processes examples and adjusts its weights based on the error in its predictions. This is 
                typically done using an algorithm called backpropagation and optimization methods like 
                gradient descent.
                
                ## Types of Neural Networks
                
                There are many types of neural networks, including:
                
                1. Feedforward Neural Networks
                2. Convolutional Neural Networks (CNNs)
                3. Recurrent Neural Networks (RNNs)
                4. Long Short-Term Memory Networks (LSTMs)
                5. Generative Adversarial Networks (GANs)
                
                Each type has its own architecture and is suited for different kinds of problems.
                """,
                "metadata": {
                    "type": "article",
                    "subject": "neural networks",
                    "difficulty": "beginner"
                }
            },
            {
                "id": "long-article-2",
                "content": """
                # Advanced Neural Network Architectures
                
                As deep learning has evolved, researchers have developed increasingly sophisticated neural 
                network architectures to tackle complex problems.
                
                ## Transformers
                
                Transformer models like BERT, GPT, and T5 have revolutionized natural language processing. 
                They use self-attention mechanisms to process sequences in parallel, rather than sequentially 
                like RNNs.
                
                ## Graph Neural Networks
                
                Graph Neural Networks (GNNs) operate on graph-structured data, making them suitable for tasks 
                like social network analysis, molecule property prediction, and recommendation systems.
                
                ## Neuroevolution
                
                Neuroevolution involves using evolutionary algorithms to optimize neural network architectures 
                and weights. This approach has shown promise in reinforcement learning and complex control tasks.
                
                ## Neural Architecture Search
                
                Neural Architecture Search (NAS) automates the process of designing neural network architectures, 
                potentially discovering designs that outperform human-created ones.
                
                ## Capsule Networks
                
                Introduced by Geoffrey Hinton, Capsule Networks aim to address limitations of CNNs by better 
                modeling hierarchical relationships and preserving spatial information.
                """,
                "metadata": {
                    "type": "article",
                    "subject": "neural networks",
                    "difficulty": "advanced"
                }
            }
        ]

        # Add documents one by one to ensure chunking
        for doc in documents:
            print(f"Adding document {doc['id']}...")
            result = client.add_document(collection_id, doc)
            print(f"Document added: {result}")

        # Demonstrate different search approaches

        # 1. Basic search
        print("\n1. Basic Search:")
        results = client.search(
            collection_id,
            "transformer models neural networks",
            top_k=5
        )
        print(f"Found {len(results['results'])} results in {results['query_time_ms']:.2f}ms")
        for i, result in enumerate(results['results']):
            print(f"Result {i + 1}: {result['id']} (score: {result['score']:.4f})")
            print(f"  Content: {result['content'][:100]}...")
            if "original_document_id" in result.get("metadata", {}):
                print(f"  Part of document: {result['metadata']['original_document_id']}")

        # 2. Advanced search with deduplication
        print("\n2. Advanced Search with Deduplication:")
        results = client.advanced_search(
            collection_id,
            "transformer models neural networks",
            top_k=5,
            deduplicate=True
        )
        print(f"Found {len(results['results'])} results in {results['query_time_ms']:.2f}ms")
        for i, result in enumerate(results['results']):
            print(f"Result {i + 1}: {result['id']} (score: {result['score']:.4f})")
            print(f"  Content: {result['content'][:100]}...")

        # 3. Advanced search with merging chunks
        print("\n3. Advanced Search with Chunk Merging:")
        results = client.advanced_search(
            collection_id,
            "transformer models neural networks",
            top_k=5,
            deduplicate=True,
            merge_chunks=True
        )
        print(f"Found {len(results['results'])} results in {results['query_time_ms']:.2f}ms")
        for i, result in enumerate(results['results']):
            print(f"Result {i + 1}: {result['id']} (score: {result['score']:.4f})")
            if "merged_from_chunks" in result.get("metadata", {}):
                print(f"  Merged from {result['metadata']['merged_from_chunks']} chunks")
            print(f"  Content: {result['content'][:100]}...")

        # 4. Advanced search with metadata filtering
        print("\n4. Advanced Search with Metadata Filtering:")
        results = client.advanced_search(
            collection_id,
            "neural networks",
            top_k=5,
            metadata_filter={"difficulty": "advanced"},
            deduplicate=True,
            merge_chunks=True
        )
        print(f"Found {len(results['results'])} results in {results['query_time_ms']:.2f}ms")
        for i, result in enumerate(results['results']):
            print(f"Result {i + 1}: {result['id']} (score: {result['score']:.4f})")
            if "metadata" in result and "difficulty" in result["metadata"]:
                print(f"  Difficulty: {result['metadata']['difficulty']}")
            print(f"  Content: {result['content'][:100]}...")

        # 5. Advanced search with score thresholding
        print("\n5. Advanced Search with Score Thresholding:")
        results = client.advanced_search(
            collection_id,
            "capsule networks hinton",
            top_k=5,
            min_score=0.5,  # Higher threshold
            deduplicate=True,
            merge_chunks=True
        )
        print(f"Found {len(results['results'])} results in {results['query_time_ms']:.2f}ms")
        for i, result in enumerate(results['results']):
            print(f"Result {i + 1}: {result['id']} (score: {result['score']:.4f})")
            print(f"  Content: {result['content'][:100]}...")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
