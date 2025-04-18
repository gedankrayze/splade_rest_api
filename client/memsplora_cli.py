#!/usr/bin/env python3
import argparse
import json
import sys

from memsplora_client import MemSploraClient


def handle_collections(args: argparse.Namespace, client: MemSploraClient) -> None:
    if args.action == "list":
        print(json.dumps(client.list_collections(), indent=2))
    elif args.action == "get":
        print(json.dumps(client.get_collection(args.id), indent=2))
    elif args.action == "create":
        result = client.create_collection(args.id, args.name, args.description, args.model_name)
        print(json.dumps(result, indent=2))
    elif args.action == "delete":
        client.delete_collection(args.id)
        print(f"Collection {args.id} deleted successfully")
    elif args.action == "stats":
        print(json.dumps(client.get_collection_stats(args.id), indent=2))


def handle_documents(args: argparse.Namespace, client: MemSploraClient) -> None:
    if args.action == "add":
        with open(args.file) as f:
            doc = json.load(f)
        result = client.add_document(args.collection_id, doc)
        print(json.dumps(result, indent=2))
    elif args.action == "batch":
        with open(args.file) as f:
            docs = json.load(f)
        result = client.batch_add_documents(args.collection_id, docs)
        print(json.dumps(result, indent=2))
    elif args.action == "get":
        print(json.dumps(client.get_document(args.collection_id, args.id), indent=2))
    elif args.action == "delete":
        client.delete_document(args.collection_id, args.id)
        print(f"Document {args.id} deleted successfully")


def handle_search(args: argparse.Namespace, client: MemSploraClient) -> None:
    metadata_filter = json.loads(args.metadata_filter) if args.metadata_filter else None

    # Create geo search parameters if latitude and longitude are provided
    geo_search = None
    if args.latitude is not None and args.longitude is not None:
        geo_search = {
            'latitude': args.latitude,
            'longitude': args.longitude
        }
        if args.radius_km is not None:
            geo_search['radius_km'] = args.radius_km
    
    if args.mode == "basic":
        if args.collection_id:
            result = client.search(
                args.collection_id,
                args.query,
                args.top_k,
                metadata_filter,
                args.min_score,
                geo_search
            )
        else:
            result = client.search_all(
                args.query,
                args.top_k,
                metadata_filter,
                args.min_score,
                geo_search
            )
    else:  # advanced
        if args.collection_id:
            result = client.advanced_search(
                args.collection_id,
                args.query,
                args.top_k,
                args.min_score,
                metadata_filter,
                args.deduplicate,
                args.merge_chunks,
                geo_search
            )
        else:
            result = client.advanced_search_all(
                args.query,
                args.top_k,
                args.min_score,
                metadata_filter,
                args.deduplicate,
                args.merge_chunks,
                geo_search
            )
    
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="MemSplora CLI")
    parser.add_argument("--url", default="http://localhost:3000", help="API base URL")
    parser.add_argument("--api-key", help="API key for authentication")
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Collections commands
    collections_parser = subparsers.add_parser("collections")
    collections_subparsers = collections_parser.add_subparsers(dest="action", required=True)
    
    collections_subparsers.add_parser("list")
    
    get_collection = collections_subparsers.add_parser("get")
    get_collection.add_argument("id", help="Collection ID")
    
    create_collection = collections_subparsers.add_parser("create")
    create_collection.add_argument("id", help="Collection ID")
    create_collection.add_argument("name", help="Collection name")
    create_collection.add_argument("--description", help="Collection description")
    create_collection.add_argument("--model-name", help="Domain-specific model name for this collection")
    
    delete_collection = collections_subparsers.add_parser("delete")
    delete_collection.add_argument("id", help="Collection ID")
    
    stats_collection = collections_subparsers.add_parser("stats")
    stats_collection.add_argument("id", help="Collection ID")
    
    # Documents commands
    documents_parser = subparsers.add_parser("documents")
    documents_subparsers = documents_parser.add_subparsers(dest="action", required=True)
    
    add_document = documents_subparsers.add_parser("add")
    add_document.add_argument("collection_id", help="Collection ID")
    add_document.add_argument("file", help="JSON file containing the document")
    
    batch_document = documents_subparsers.add_parser("batch")
    batch_document.add_argument("collection_id", help="Collection ID")
    batch_document.add_argument("file", help="JSON file containing array of documents")
    
    get_document = documents_subparsers.add_parser("get")
    get_document.add_argument("collection_id", help="Collection ID")
    get_document.add_argument("id", help="Document ID")
    
    delete_document = documents_subparsers.add_parser("delete")
    delete_document.add_argument("collection_id", help="Collection ID")
    delete_document.add_argument("id", help="Document ID")
    
    # Search commands
    search_parser = subparsers.add_parser("search")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--collection-id", help="Collection ID (optional)")
    search_parser.add_argument("--mode", choices=["basic", "advanced"], default="basic", help="Search mode")
    search_parser.add_argument("--top-k", type=int, default=10, help="Number of results")
    search_parser.add_argument("--metadata-filter", help="JSON metadata filter")
    search_parser.add_argument("--min-score", type=float, default=0.3, help="Minimum similarity score threshold (0-1)")
    search_parser.add_argument("--deduplicate", action="store_true", default=True,
                               help="Deduplicate results from same document")
    search_parser.add_argument("--merge-chunks", action="store_true", default=True,
                               help="Merge chunks from the same document")

    # Geo-spatial search options
    search_parser.add_argument("--latitude", type=float, help="Latitude for geo search (-90 to 90)")
    search_parser.add_argument("--longitude", type=float, help="Longitude for geo search (-180 to 180)")
    search_parser.add_argument("--radius-km", type=float, default=10.0, help="Search radius in kilometers")
    
    args = parser.parse_args()
    client = MemSploraClient(args.url, args.api_key)
    
    try:
        if args.command == "collections":
            handle_collections(args, client)
        elif args.command == "documents":
            handle_documents(args, client)
        elif args.command == "search":
            handle_search(args, client)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
