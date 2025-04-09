#!/usr/bin/env python3
"""
Example demonstrating the geo-spatial search capabilities of the MemSplora client.
This script shows how to:
1. Add documents with location data
2. Perform geo-spatial search within a specific radius
3. Access location and distance information in results
"""

import time
from uuid import uuid4

from memsplora_client import MemSploraClient

# Initialize client
client = MemSploraClient("http://localhost:3000")

# Create a test collection
collection_id = f"geo-test-{int(time.time())}"
print(f"Creating collection: {collection_id}")
client.create_collection(collection_id, "Geo-Spatial Test Collection")

# Example POI data
pois = [
    {
        "name": "Golden Gate Bridge",
        "description": "Iconic suspension bridge in San Francisco",
        "latitude": 37.8199,
        "longitude": -122.4783
    },
    {
        "name": "Fisherman's Wharf",
        "description": "Popular tourist attraction with seafood restaurants",
        "latitude": 37.8080,
        "longitude": -122.4177
    },
    {
        "name": "Alcatraz Island",
        "description": "Historic federal prison on an island",
        "latitude": 37.8270,
        "longitude": -122.4230
    },
    {
        "name": "Union Square",
        "description": "Shopping district in downtown San Francisco",
        "latitude": 37.7881,
        "longitude": -122.4075
    },
    {
        "name": "Chinatown",
        "description": "Oldest Chinatown in North America",
        "latitude": 37.7941,
        "longitude": -122.4078
    },
    {
        "name": "Pier 39",
        "description": "Popular shopping center and tourist attraction",
        "latitude": 37.8087,
        "longitude": -122.4098
    },
    {
        "name": "Twin Peaks",
        "description": "Famous hills offering panoramic views of the city",
        "latitude": 37.7544,
        "longitude": -122.4477
    }
]

# Add documents with location data
print(f"Adding {len(pois)} documents with location data")
for poi in pois:
    document = {
        "id": f"poi-{uuid4()}",
        "content": f"{poi['name']}: {poi['description']}",
        "metadata": {
            "name": poi['name'],
            "type": "poi"
        },
        "location": {
            "latitude": poi['latitude'],
            "longitude": poi['longitude']
        }
    }

    result = client.add_document(collection_id, document)
    print(f"Added {result['id']}")

# Wait a moment for indexing
print("Waiting for documents to be indexed...")
time.sleep(1)

# Define a search location (Union Square)
search_point = {
    "latitude": 37.7881,  # Union Square
    "longitude": -122.4075
}

# Perform geo-spatial search
print("\nPerforming geo-spatial search from Union Square (1 km radius):")
geo_results = client.search(
    collection_id,
    query="",  # Empty query to match all documents
    geo_search={
        "latitude": search_point["latitude"],
        "longitude": search_point["longitude"],
        "radius_km": 1.0  # 1 km radius
    }
)

# Display results with distances
print(f"Found {len(geo_results['results'])} results within 1 km:")
for result in geo_results['results']:
    name = result['metadata']['name']
    distance = result.get('distance_km', 'unknown')
    print(f"- {name} (Distance: {distance} km)")

# Now perform a search with a larger radius
print("\nPerforming geo-spatial search from Union Square (3 km radius):")
geo_results = client.search(
    collection_id,
    query="",  # Empty query to match all documents
    geo_search={
        "latitude": search_point["latitude"],
        "longitude": search_point["longitude"],
        "radius_km": 3.0  # 3 km radius
    }
)

# Display results with distances
print(f"Found {len(geo_results['results'])} results within 3 km:")
for result in geo_results['results']:
    name = result['metadata']['name']
    distance = result.get('distance_km', 'unknown')
    print(f"- {name} (Distance: {distance} km)")

# Combine text query with geo search
print("\nCombining text query with geo-spatial search:")
geo_results = client.search(
    collection_id,
    query="restaurant OR shopping",
    geo_search={
        "latitude": search_point["latitude"],
        "longitude": search_point["longitude"],
        "radius_km": 5.0  # 5 km radius
    }
)

# Display results with distances
print(f"Found {len(geo_results['results'])} results matching query within 5 km:")
for result in geo_results['results']:
    name = result['metadata']['name']
    distance = result.get('distance_km', 'unknown')
    print(f"- {name} (Distance: {distance} km)")
    print(f"  Score: {result['score']}")
    print(f"  Content: {result['content'][:50]}...")

# Clean up - delete the test collection
print(f"\nCleaning up - deleting collection {collection_id}")
client.delete_collection(collection_id)
print("Done!")
