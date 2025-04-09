"""
Core SPLADE service with FAISS integration and collection support
"""

import logging
import os
import pickle
import threading
import time
from typing import List, Dict, Any, Optional, Tuple

import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

from app.core.chunking import document_chunker
from app.core.config import settings
from app.models.schema import Document

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("splade_service")


class SpladeCollection:
    """
    SPLADE Collection with FAISS index for a group of documents
    """

    def __init__(
            self,
            collection_id: str,
            name: str,
            description: Optional[str] = None,
            vocab_size: int = 30522,  # Default BERT vocab size
            data_dir: str = settings.DATA_DIR
    ):
        """Initialize a SPLADE collection with its own FAISS index"""
        self.id = collection_id
        self.name = name
        self.description = description
        self.vocab_size = vocab_size
        self.data_dir = data_dir

        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)

        # Collection data paths
        self.index_path = os.path.join(data_dir, f"{collection_id}_index.faiss")
        self.data_path = os.path.join(data_dir, f"{collection_id}_data.pkl")

        # Storage for documents and mappings
        self.documents = {}  # id -> Document
        self.id_to_index = {}  # id -> index in FAISS
        self.index_to_id = {}  # index in FAISS -> id
        self.next_index = 0  # Next index to use in FAISS

        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(vocab_size)

        # Lock for thread safety
        self.lock = threading.RLock()

        # Load existing data if available
        self._load_data()

    def add_document(self, doc: Document, vector: np.ndarray) -> bool:
        """Add a document to the collection with its pre-encoded vector"""
        with self.lock:
            if doc.id in self.documents:
                logger.warning(f"Document with ID {doc.id} already exists in collection {self.id}")
                return False

            # Normalize vector for cosine similarity
            faiss.normalize_L2(vector.reshape(1, -1))

            # Add to FAISS index
            self.index.add(vector.reshape(1, -1))

            # Store document and mapping
            self.documents[doc.id] = doc
            self.id_to_index[doc.id] = self.next_index
            self.index_to_id[self.next_index] = doc.id
            self.next_index += 1

            # Persist changes
            self._save_data()

            logger.info(f"Added document {doc.id} to collection {self.id}")
            return True

    def batch_add_documents(self, docs: List[Document], vectors: List[np.ndarray]) -> int:
        """Add multiple documents to the collection in batch"""
        with self.lock:
            if len(docs) != len(vectors):
                raise ValueError("Number of documents and vectors must match")

            # Filter out documents that already exist
            new_docs = []
            new_vectors = []
            for doc, vec in zip(docs, vectors):
                if doc.id not in self.documents:
                    new_docs.append(doc)
                    new_vectors.append(vec)

            if not new_docs:
                logger.warning(f"No new documents to add to collection {self.id}")
                return 0

            # Normalize vectors
            vectors_array = np.vstack(new_vectors)
            faiss.normalize_L2(vectors_array)

            # Add to FAISS index
            self.index.add(vectors_array)

            # Store documents and mappings
            for i, doc in enumerate(new_docs):
                self.documents[doc.id] = doc
                self.id_to_index[doc.id] = self.next_index
                self.index_to_id[self.next_index] = doc.id
                self.next_index += 1

            # Persist changes
            self._save_data()

            logger.info(f"Added {len(new_docs)} documents to collection {self.id}")
            return len(new_docs)

    def remove_document(self, doc_id: str) -> bool:
        """Remove a document from the collection"""
        with self.lock:
            if doc_id not in self.documents:
                logger.warning(f"Document with ID {doc_id} not found in collection {self.id}")
                return False

            # FAISS does not support direct removal, need to rebuild index
            logger.info(f"Removing document {doc_id} from collection {self.id}")

            # Get index in FAISS
            removed_index = self.id_to_index[doc_id]

            # Remove from storage
            del self.documents[doc_id]
            del self.id_to_index[doc_id]
            del self.index_to_id[removed_index]

            # If collection is now empty, just reset the index
            if not self.documents:
                self.index = faiss.IndexFlatIP(self.vocab_size)
                self.next_index = 0
                self._save_data()
                return True

            # Otherwise, we need to rebuild the index
            # This is done by the parent service which has access to the model
            return True

    def search(self, query_vector: np.ndarray, top_k: int, filter_metadata: Optional[Dict[str, Any]] = None,
               min_score: float = settings.MIN_SCORE_THRESHOLD) -> List[Dict[str, Any]]:
        """Search the collection with a query vector"""
        with self.lock:
            # If collection is empty, return empty results
            if self.index.ntotal == 0:
                return []

            # Normalize query vector
            query_vector_normalized = query_vector.copy()
            faiss.normalize_L2(query_vector_normalized.reshape(1, -1))

            # Search FAISS index - get more results than needed to account for filtering
            scores, indices = self.index.search(
                query_vector_normalized.reshape(1, -1), 
                min(top_k * 3, self.index.ntotal)  # Get extra results for filtering
            )

            results = []
            for i, idx in enumerate(indices[0]):
                # Skip invalid indices or scores below threshold
                if idx == -1 or idx not in self.index_to_id or scores[0][i] < min_score:  
                    continue

                doc_id = self.index_to_id[idx]
                doc = self.documents[doc_id]

                # Apply metadata filtering if specified
                if filter_metadata and doc.metadata:
                    skip = False
                    for key, value in filter_metadata.items():
                        if key not in doc.metadata or doc.metadata[key] != value:
                            skip = True
                            break
                    if skip:
                        continue

                results.append({
                    "id": doc.id,
                    "content": doc.content,
                    "metadata": doc.metadata,
                    "score": float(scores[0][i])
                })

                # If we have enough results after filtering, stop
                if len(results) >= top_k:
                    break

            return results

    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get a document by ID"""
        with self.lock:
            return self.documents.get(doc_id)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        with self.lock:
            return {
                "id": self.id,
                "name": self.name,
                "document_count": len(self.documents),
                "index_size": self.index.ntotal,
                "vocab_size": self.vocab_size
            }

    def _save_data(self):
        """Save collection data to disk"""
        with self.lock:
            # Save document store
            with open(self.data_path, "wb") as f:
                pickle.dump({
                    "id": self.id,
                    "name": self.name,
                    "description": self.description,
                    "documents": self.documents,
                    "id_to_index": self.id_to_index,
                    "index_to_id": self.index_to_id,
                    "next_index": self.next_index
                }, f)

            # Save FAISS index
            faiss.write_index(self.index, self.index_path)

            logger.info(f"Collection {self.id} persisted to disk: {len(self.documents)} documents")

    def _load_data(self):
        """Load collection data from disk"""
        try:
            # Check if data files exist
            if not (os.path.exists(self.data_path) and os.path.exists(self.index_path)):
                logger.info(f"No existing data found for collection {self.id}, starting with empty collection")
                return

            # Load document store
            with open(self.data_path, "rb") as f:
                data = pickle.load(f)
                self.name = data["name"]
                self.description = data["description"]
                self.documents = data["documents"]
                self.id_to_index = data["id_to_index"]
                self.index_to_id = data["index_to_id"]
                self.next_index = data["next_index"]

            # Load FAISS index
            self.index = faiss.read_index(self.index_path)

            logger.info(f"Loaded collection {self.id} with {len(self.documents)} documents")
        except Exception as e:
            logger.error(f"Error loading data for collection {self.id}: {e}")
            logger.info(f"Starting with empty collection for {self.id}")


class SpladeService:
    """
    SPLADE service with collection management
    """

    def __init__(self, model_dir: str = settings.MODEL_DIR, data_dir: str = settings.DATA_DIR):
        """Initialize SPLADE service"""
        self.model_dir = model_dir
        self.data_dir = data_dir

        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)

        # Path for collections metadata
        self.collections_path = os.path.join(data_dir, "collections.pkl")

        # Initialize tokenizer and model
        logger.info(f"Loading model from {model_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForMaskedLM.from_pretrained(model_dir)

        # Set device
        if torch.backends.mps.is_available():
            logger.info("Using MPS (Metal Performance Shaders) for GPU acceleration")
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            logger.info("Using CUDA for GPU acceleration")
            self.device = torch.device("cuda")
        else:
            logger.info("Using CPU for inference")
            self.device = torch.device("cpu")

        self.model.to(self.device)
        self.model.eval()

        # Get vocabulary size from the model
        self.vocab_size = self.model.config.vocab_size
        logger.info(f"Model vocabulary size: {self.vocab_size}")

        # Collection storage
        self.collections = {}  # id -> SpladeCollection

        # Lock for thread safety
        self.lock = threading.RLock()

        # Load existing collections
        self._load_collections()

    def encode_text(self, text: str) -> np.ndarray:
        """Encode text into SPLADE sparse representation"""
        inputs = self.tokenizer(
            text,
            max_length=settings.MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, return_dict=True)

        # Apply SPLADE pooling (log(1 + ReLU(x)))
        logits = outputs.logits
        activated = torch.log(1 + torch.relu(logits))

        # Max pooling over sequence dimension
        attention_expanded = inputs["attention_mask"].unsqueeze(-1).expand_as(activated)
        masked_activated = activated * attention_expanded
        sparse_rep = torch.max(masked_activated, dim=1)[0]

        return sparse_rep.cpu().numpy()[0]

    def batch_encode_texts(self, texts: List[str]) -> List[np.ndarray]:
        """Encode multiple texts in batches"""
        results = []
        for text in texts:
            vector = self.encode_text(text)
            results.append(vector)
        return results

    def create_collection(self, collection_id: str, name: str, description: Optional[str] = None) -> bool:
        """Create a new collection"""
        with self.lock:
            if collection_id in self.collections:
                logger.warning(f"Collection with ID {collection_id} already exists")
                return False

            # Create new collection
            collection = SpladeCollection(
                collection_id=collection_id,
                name=name,
                description=description,
                vocab_size=self.vocab_size,
                data_dir=self.data_dir
            )

            # Add to collections
            self.collections[collection_id] = collection

            # Save collections metadata
            self._save_collections()

            logger.info(f"Created collection {collection_id}")
            return True

    def delete_collection(self, collection_id: str) -> bool:
        """Delete a collection"""
        with self.lock:
            if collection_id not in self.collections:
                logger.warning(f"Collection with ID {collection_id} not found")
                return False

            # Get collection
            collection = self.collections[collection_id]

            # Remove collection files
            try:
                if os.path.exists(collection.index_path):
                    os.remove(collection.index_path)
                if os.path.exists(collection.data_path):
                    os.remove(collection.data_path)
            except Exception as e:
                logger.error(f"Error removing collection files: {e}")

            # Remove from collections
            del self.collections[collection_id]

            # Save collections metadata
            self._save_collections()

            logger.info(f"Deleted collection {collection_id}")
            return True

    def add_document(self, collection_id: str, doc: Document, auto_chunk: bool = True) -> bool:
        """Add a document to a collection"""
        # Check if collection exists
        if collection_id not in self.collections:
            logger.warning(f"Collection with ID {collection_id} not found")
            return False

        # If auto_chunk is enabled and document is large, split it into chunks
        if auto_chunk:
            # Get document chunks
            chunks = document_chunker.chunk_document(doc)

            if len(chunks) > 1:
                logger.info(f"Document {doc.id} split into {len(chunks)} chunks")

                # Add each chunk
                success = True
                for chunk in chunks:
                    # Encode document
                    vector = self.encode_text(chunk.content)

                    # Add to collection (don't auto-chunk to avoid infinite recursion)
                    chunk_success = self.collections[collection_id].add_document(chunk, vector)
                    success = success and chunk_success

                return success

        # If not auto-chunking or document is small enough
        # Encode document
        vector = self.encode_text(doc.content)

        # Add to collection
        return self.collections[collection_id].add_document(doc, vector)

    def batch_add_documents(self, collection_id: str, docs: List[Document], auto_chunk: bool = True) -> int:
        """Add multiple documents to a collection in batch"""
        # Check if collection exists
        if collection_id not in self.collections:
            logger.warning(f"Collection with ID {collection_id} not found")
            return 0

        added_count = 0

        # If auto-chunking is enabled
        if auto_chunk:
            # Process each document, potentially splitting into chunks
            all_chunks = []
            for doc in docs:
                chunks = document_chunker.chunk_document(doc)
                all_chunks.extend(chunks)

            if len(all_chunks) > len(docs):
                logger.info(f"Split {len(docs)} documents into {len(all_chunks)} chunks")

                # Encode all chunks
                vectors = self.batch_encode_texts([chunk.content for chunk in all_chunks])

                # Add chunks to collection
                added_count = self.collections[collection_id].batch_add_documents(all_chunks, vectors)
                return added_count

        # If not auto-chunking or no documents were chunked
        # Encode documents
        vectors = self.batch_encode_texts([doc.content for doc in docs])

        # Add to collection
        return self.collections[collection_id].batch_add_documents(docs, vectors)

    def remove_document(self, collection_id: str, doc_id: str) -> bool:
        """Remove a document from a collection"""
        # Check if collection exists
        if collection_id not in self.collections:
            logger.warning(f"Collection with ID {collection_id} not found")
            return False

        # Remove from collection
        result = self.collections[collection_id].remove_document(doc_id)

        # If collection needs index rebuild, do it here
        if result and self.collections[collection_id].documents:
            self._rebuild_collection_index(collection_id)

        return result

    def _rebuild_collection_index(self, collection_id: str):
        """Rebuild the FAISS index for a collection after document removal"""
        with self.lock:
            collection = self.collections[collection_id]

            # Encode all documents
            docs = list(collection.documents.values())
            vectors = self.batch_encode_texts([doc.content for doc in docs])

            # Create new mappings
            id_to_index = {}
            index_to_id = {}

            # Reset the index
            collection.index = faiss.IndexFlatIP(self.vocab_size)

            # Add vectors
            if vectors:
                vectors_array = np.vstack(vectors)
                faiss.normalize_L2(vectors_array)
                collection.index.add(vectors_array)

            # Update mappings
            for i, doc in enumerate(docs):
                id_to_index[doc.id] = i
                index_to_id[i] = doc.id

            # Update collection
            collection.id_to_index = id_to_index
            collection.index_to_id = index_to_id
            collection.next_index = len(docs)

            # Save collection
            collection._save_data()

            logger.info(f"Rebuilt index for collection {collection_id} with {len(docs)} documents")

    def search(self, collection_id: str, query: str, top_k: int = settings.DEFAULT_TOP_K,
               filter_metadata: Optional[Dict[str, Any]] = None, min_score: float = settings.MIN_SCORE_THRESHOLD) -> \
    Tuple[List[Dict[str, Any]], float]:
        """Search for documents in a collection"""
        # Check if collection exists
        if collection_id not in self.collections:
            logger.warning(f"Collection with ID {collection_id} not found")
            return [], 0.0

        # Start timing
        start_time = time.time()

        # Encode query
        query_vector = self.encode_text(query)

        # Search collection
        results = self.collections[collection_id].search(query_vector, top_k, filter_metadata, min_score)

        # End timing
        query_time = (time.time() - start_time) * 1000  # in milliseconds

        return results, query_time

    def search_all_collections(self, query: str, top_k: int = settings.DEFAULT_TOP_K,
                               filter_metadata: Optional[Dict[str, Any]] = None,
                               min_score: float = settings.MIN_SCORE_THRESHOLD) -> Dict[str, Any]:
        """Search across all collections"""
        # Start timing
        start_time = time.time()

        # Encode query
        query_vector = self.encode_text(query)

        # Search all collections
        all_results = {}
        for collection_id, collection in self.collections.items():
            collection_results = collection.search(query_vector, top_k, filter_metadata, min_score)
            if collection_results:
                all_results[collection_id] = collection_results

        # End timing
        query_time = (time.time() - start_time) * 1000  # in milliseconds

        return {
            "results": all_results,
            "query_time_ms": query_time
        }

    def get_document(self, collection_id: str, doc_id: str) -> Optional[Document]:
        """Get a document from a collection"""
        # Check if collection exists
        if collection_id not in self.collections:
            logger.warning(f"Collection with ID {collection_id} not found")
            return None

        # Get document from collection
        return self.collections[collection_id].get_document(doc_id)

    def get_collection(self, collection_id: str) -> Optional[Dict[str, Any]]:
        """Get collection details"""
        if collection_id not in self.collections:
            return None

        collection = self.collections[collection_id]
        return {
            "id": collection.id,
            "name": collection.name,
            "description": collection.description,
            "stats": collection.get_stats()
        }

    def list_collections(self) -> List[Dict[str, Any]]:
        """List all collections"""
        return [
            {
                "id": collection.id,
                "name": collection.name,
                "description": collection.description
            }
            for collection in self.collections.values()
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all collections"""
        return {
            "collection_count": len(self.collections),
            "collections": [collection.get_stats() for collection in self.collections.values()]
        }

    def _save_collections(self):
        """Save collections metadata to disk"""
        with self.lock:
            collections_data = [
                {
                    "id": collection.id,
                    "name": collection.name,
                    "description": collection.description
                }
                for collection in self.collections.values()
            ]

            with open(self.collections_path, "wb") as f:
                pickle.dump(collections_data, f)

            logger.info(f"Saved metadata for {len(collections_data)} collections")

    def _load_collections(self):
        """Load collections from disk"""
        try:
            if not os.path.exists(self.collections_path):
                logger.info("No collections metadata found, starting with empty collections")
                return

            with open(self.collections_path, "rb") as f:
                collections_data = pickle.load(f)

            for collection_data in collections_data:
                collection_id = collection_data["id"]
                name = collection_data["name"]
                description = collection_data["description"]

                # Create collection object
                collection = SpladeCollection(
                    collection_id=collection_id,
                    name=name,
                    description=description,
                    vocab_size=self.vocab_size,
                    data_dir=self.data_dir
                )

                self.collections[collection_id] = collection

            logger.info(f"Loaded {len(self.collections)} collections")
        except Exception as e:
            logger.error(f"Error loading collections: {e}")
            logger.info("Starting with empty collections")


# Initialize the SPLADE service as a singleton
try:
    splade_service = SpladeService()
    logger.info(f"SPLADE service initialized with model directory: {settings.MODEL_DIR}")
except Exception as e:
    logger.error(f"Error initializing SPLADE service: {e}")
    # Create a placeholder service that will be properly initialized when the model is available
    splade_service = None
