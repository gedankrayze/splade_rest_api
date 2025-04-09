"""
Geo-spatial indexing and search functionality
"""

import logging
import math
from typing import List, Tuple, Set, Optional

logger = logging.getLogger("geo_spatial_index")


class GeoSpatialIndex:
    """
    Spatial index for geo queries using grid-based spatial partitioning
    
    This implementation uses a simple but efficient grid-based approach
    that works well for most use cases without external dependencies.
    """

    def __init__(self, precision: int = 6):
        """Initialize spatial index
        
        Args:
            precision: Grid precision (1-12, higher is more precise)
        """
        self.precision = precision
        self.cell_to_docs = {}  # cell_id -> set of doc_ids
        self.doc_to_coords = {}  # doc_id -> (lat, lon)

    def add_document(self, doc_id: str, lat: float, lon: float) -> None:
        """
        Add document to spatial index
        
        Args:
            doc_id: Document identifier
            lat: Latitude (-90 to 90)
            lon: Longitude (-180 to 180)
        """
        # Validate coordinates
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            logger.warning(f"Invalid coordinates for document {doc_id}: {lat}, {lon}")
            return

        # Get cell ID for these coordinates
        cell = self._get_cell_id(lat, lon)

        # Add document to cell
        if cell not in self.cell_to_docs:
            self.cell_to_docs[cell] = set()

        self.cell_to_docs[cell].add(doc_id)
        self.doc_to_coords[doc_id] = (lat, lon)

    def remove_document(self, doc_id: str) -> None:
        """Remove document from spatial index"""
        if doc_id not in self.doc_to_coords:
            return

        # Get coordinates and cell
        lat, lon = self.doc_to_coords[doc_id]
        cell = self._get_cell_id(lat, lon)

        # Remove from cell
        if cell in self.cell_to_docs and doc_id in self.cell_to_docs[cell]:
            self.cell_to_docs[cell].remove(doc_id)

            # Clean up empty cells
            if not self.cell_to_docs[cell]:
                del self.cell_to_docs[cell]

        # Remove coordinate mapping
        del self.doc_to_coords[doc_id]

    def search_radius(self, lat: float, lon: float, radius_km: float) -> Set[str]:
        """
        Find documents within radius of point
        
        Args:
            lat: Latitude of query point
            lon: Longitude of query point
            radius_km: Search radius in kilometers
            
        Returns:
            Set of document IDs within the radius
        """
        # Calculate cell coverage for the radius
        cells = self._get_cells_for_radius(lat, lon, radius_km)

        # Collect candidate documents from all cells
        candidates = set()
        for cell in cells:
            if cell in self.cell_to_docs:
                candidates.update(self.cell_to_docs[cell])

        # Filter by actual distance
        results = set()
        for doc_id in candidates:
            doc_lat, doc_lon = self.doc_to_coords[doc_id]
            distance = self._haversine_distance(lat, lon, doc_lat, doc_lon)
            if distance <= radius_km:
                results.add(doc_id)

        return results

    def get_location(self, doc_id: str) -> Optional[Tuple[float, float]]:
        """Get the coordinates for a document"""
        return self.doc_to_coords.get(doc_id)

    def _get_cell_id(self, lat: float, lon: float) -> str:
        """
        Convert coordinates to cell ID based on precision
        
        This is a simplified version of geohashing that uses a grid system
        """
        # Scale to avoid negative numbers
        lat_scaled = (lat + 90) / 180.0
        lon_scaled = (lon + 180) / 360.0

        # Determine cell size based on precision
        grid_size = 2 ** self.precision

        # Calculate grid coordinates
        lat_grid = min(int(lat_scaled * grid_size), grid_size - 1)
        lon_grid = min(int(lon_scaled * grid_size), grid_size - 1)

        # Create cell ID
        return f"{lat_grid}:{lon_grid}"

    def _get_cells_for_radius(self, lat: float, lon: float, radius_km: float) -> List[str]:
        """Get all cells that might contain points within the given radius"""
        # Calculate the rough degree change for the given radius
        # 1 degree of latitude is approximately 111 km
        lat_radius = radius_km / 111.0

        # 1 degree of longitude varies with latitude
        lon_radius = radius_km / (111.0 * math.cos(math.radians(abs(lat))))

        # Calculate bounding box
        min_lat = max(lat - lat_radius, -90)
        max_lat = min(lat + lat_radius, 90)
        min_lon = max(lon - lon_radius, -180)
        max_lon = min(lon + lon_radius, 180)

        # Get cell ranges
        min_lat_scaled = (min_lat + 90) / 180.0
        max_lat_scaled = (max_lat + 90) / 180.0
        min_lon_scaled = (min_lon + 180) / 360.0
        max_lon_scaled = (max_lon + 180) / 360.0

        grid_size = 2 ** self.precision

        min_lat_grid = max(0, min(int(min_lat_scaled * grid_size), grid_size - 1))
        max_lat_grid = max(0, min(int(max_lat_scaled * grid_size), grid_size - 1))
        min_lon_grid = max(0, min(int(min_lon_scaled * grid_size), grid_size - 1))
        max_lon_grid = max(0, min(int(max_lon_scaled * grid_size), grid_size - 1))

        # Generate all cells in the bounding box
        cells = []
        for lat_grid in range(min_lat_grid, max_lat_grid + 1):
            for lon_grid in range(min_lon_grid, max_lon_grid + 1):
                cells.append(f"{lat_grid}:{lon_grid}")

        return cells

    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate the great circle distance between two points in kilometers
        
        Args:
            lat1, lon1: First point coordinates
            lat2, lon2: Second point coordinates
            
        Returns:
            Distance in kilometers
        """
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = (math.sin(dlat / 2) ** 2 +
             math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
        c = 2 * math.asin(math.sqrt(a))
        # Radius of earth in kilometers is 6371
        km = 6371.0 * c
        return km
