"""
Enhanced Vector Store Module for 2025 Stack
Supports Pinecone 7.x with gRPC client, enhanced error handling,
and performance optimizations.
"""

import os
import time
import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    from pinecone import Pinecone, ServerlessSpec, PodSpec
except (ImportError, FileNotFoundError):
    try:
        # Fallback import for Pinecone 7.x with error handling
        import pinecone as pc
        Pinecone = pc.Pinecone
        ServerlessSpec = pc.ServerlessSpec
        PodSpec = pc.PodSpec
    except (ImportError, AttributeError, FileNotFoundError):
        # Create mock classes for development/testing
        logger.warning("Pinecone import failed, using mock classes for development")

        class MockPinecone:
            def __init__(self, api_key=None): pass
            def Index(self, name): return MockIndex()
            def list_indexes(self): return []
            def create_index(self, **kwargs): pass
            def describe_index(self, name): return MockIndexDescription()
            def describe_index_stats(self, name): return MockIndexStats()

        class MockIndex:
            def upsert(self, **kwargs): pass
            def query(self, **kwargs): return {"matches": []}
            def delete(self, **kwargs): pass

        class MockIndexDescription:
            status = type('obj', (object,), {'ready': True})()

        class MockIndexStats:
            total_vector_count = 0
            namespaces = {}

        class MockServerlessSpec:
            def __init__(self, cloud="aws", region="us-east-1"): pass

        class MockPodSpec:
            def __init__(self, environment="us-east-1", pod_type="p1.x1", pods=1): pass

        Pinecone = MockPinecone
        ServerlessSpec = MockServerlessSpec
        PodSpec = MockPodSpec

import numpy as np

@dataclass
class VectorStoreConfig:
    """Configuration for vector store operations"""
    api_key: str
    index_name: str
    cloud: str = "aws"
    region: str = "us-east-1"
    dimension: Optional[int] = None
    metric: str = "cosine"
    use_grpc: bool = True
    batch_size: int = 100
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0

class VectorStoreError(Exception):
    """Custom exception for vector store operations"""
    pass

class EnhancedPineconeClient:
    """Enhanced Pinecone client with error handling and performance optimizations"""

    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self._client: Optional[Pinecone] = None
        self._index = None

    @property
    def client(self) -> Pinecone:
        """Lazy initialization of Pinecone client"""
        if self._client is None:
            try:
                logger.info(f"Initializing Pinecone client (gRPC: {self.config.use_grpc})")
                self._client = Pinecone(
                    api_key=self.config.api_key,
                    # gRPC is enabled by default in pinecone[grpc]>=7.0.0
                )
            except Exception as e:
                logger.error(f"Failed to initialize Pinecone client: {e}")
                raise VectorStoreError(f"Pinecone client initialization failed: {e}")
        return self._client

    def get_index(self, index_name: Optional[str] = None):
        """Get index with lazy loading and caching"""
        index_name = index_name or self.config.index_name

        if self._index is None or self._index.name != index_name:
            try:
                logger.info(f"Connecting to index: {index_name}")
                self._index = self.client.Index(index_name)
            except Exception as e:
                logger.error(f"Failed to connect to index {index_name}: {e}")
                raise VectorStoreError(f"Index connection failed: {e}")

        return self._index

def _get_config_from_env() -> VectorStoreConfig:
    """Get configuration from environment variables"""
    return VectorStoreConfig(
        api_key=os.getenv("PINECONE_API_KEY", ""),
        index_name=os.getenv("PINECONE_INDEX", "personal-rag"),
        cloud=os.getenv("PINECONE_CLOUD", "aws"),
        region=os.getenv("PINECONE_REGION", "us-east-1"),
        use_grpc=os.getenv("PINECONE_GRPC_ENABLED", "true").lower() == "true",
        batch_size=int(os.getenv("PINECONE_BATCH_SIZE", "100")),
        timeout=int(os.getenv("PINECONE_TIMEOUT", "30")),
    )

def _retry_operation(operation, max_attempts: int = 3, delay: float = 1.0):
    """Retry operation with exponential backoff"""
    last_exception = Exception("Unknown error")

    for attempt in range(max_attempts):
        try:
            return operation()
        except Exception as e:
            last_exception = e
            if attempt < max_attempts - 1:
                wait_time = delay * (2 ** attempt)
                logger.warning(f"Operation failed (attempt {attempt + 1}/{max_attempts}), retrying in {wait_time}s: {e}")
                time.sleep(wait_time)
            else:
                logger.error(f"Operation failed after {max_attempts} attempts: {e}")

    raise last_exception

def ensure_index(cfg, dim: int, use_pod_spec: bool = False):
    """
    Ensure index exists with enhanced error handling.

    Args:
        cfg: Configuration object
        dim: Vector dimension
        use_pod_spec: Whether to use PodSpec instead of ServerlessSpec
    """
    config = _get_config_from_env()
    client_wrapper = EnhancedPineconeClient(config)

    try:
        pc = client_wrapper.client

        # Check if index exists
        existing_indexes = pc.list_indexes()
        exists = any(ix.name == config.index_name for ix in existing_indexes)

        if not exists:
            logger.info(f"Creating index: {config.index_name} with dimension {dim}")

            # Choose spec type
            if use_pod_spec:
                spec = PodSpec(
                    environment=config.region,
                    pod_type="p1.x1",
                    pods=1
                )
            else:
                spec = ServerlessSpec(
                    cloud=config.cloud,
                    region=config.region
                )

            # Create index
            pc.create_index(
                name=config.index_name,
                dimension=dim,
                metric=config.metric,
                spec=spec,
            )

            # Wait for index to be ready
            logger.info("Waiting for index to be ready...")
            while True:
                index_info = pc.describe_index(config.index_name)
                if index_info.status.ready:
                    break
                time.sleep(5)

            logger.info(f"Index {config.index_name} created successfully")
        else:
            logger.info(f"Index {config.index_name} already exists")

    except Exception as e:
        logger.error(f"Failed to ensure index exists: {e}")
        raise VectorStoreError(f"Index creation/verification failed: {e}")

def upsert_props(cfg, vectors: List[Dict], namespace: str = "default", batch_size: int = 100):
    """
    Upsert vectors with batching and error handling.

    Args:
        cfg: Configuration object
        vectors: List of vector dictionaries
        namespace: Namespace for vectors
        batch_size: Batch size for upsert operations
    """
    if not vectors:
        logger.warning("No vectors to upsert")
        return

    config = _get_config_from_env()
    client_wrapper = EnhancedPineconeClient(config)

    try:
        index = client_wrapper.get_index()

        # Process in batches
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]

            def _upsert_batch():
                return index.upsert(vectors=batch, namespace=namespace)  # type: ignore

            _retry_operation(_upsert_batch, config.retry_attempts, config.retry_delay)

            logger.debug(f"Upserted batch {i//batch_size + 1}/{(len(vectors) + batch_size - 1)//batch_size}")

        logger.info(f"Successfully upserted {len(vectors)} vectors to namespace '{namespace}'")

    except Exception as e:
        logger.error(f"Failed to upsert vectors: {e}")
        raise VectorStoreError(f"Vector upsert failed: {e}")

def query(
    cfg,
    vector: Union[List[float], np.ndarray],
    top_k: int = 10,
    namespace: str = "default",
    include_metadata: bool = True,
    filter: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Query vectors with enhanced error handling and filtering support.

    Args:
        cfg: Configuration object
        vector: Query vector
        top_k: Number of results to return
        namespace: Namespace to search in
        include_metadata: Whether to include metadata
        filter: Optional metadata filter

    Returns:
        Query results dictionary
    """
    config = _get_config_from_env()
    client_wrapper = EnhancedPineconeClient(config)

    try:
        index = client_wrapper.get_index()

        # Convert numpy array to list if needed
        if isinstance(vector, np.ndarray):
            vector = vector.tolist()

        def _query_operation():
            return index.query(  # type: ignore
                vector=vector,
                top_k=top_k,
                namespace=namespace,
                include_metadata=include_metadata,
                filter=filter
            )

        result = _retry_operation(_query_operation, config.retry_attempts, config.retry_delay)

        logger.debug(f"Query returned {len(result.matches)} results")  # type: ignore
        return result  # type: ignore

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise VectorStoreError(f"Vector query failed: {e}")

def delete_vectors(cfg, ids: List[str], namespace: str = "default"):
    """
    Delete vectors by IDs.

    Args:
        cfg: Configuration object
        ids: List of vector IDs to delete
        namespace: Namespace containing the vectors
    """
    config = _get_config_from_env()
    client_wrapper = EnhancedPineconeClient(config)

    try:
        index = client_wrapper.get_index()

        def _delete_operation():
            return index.delete(ids=ids, namespace=namespace)

        _retry_operation(_delete_operation, config.retry_attempts, config.retry_delay)

        logger.info(f"Successfully deleted {len(ids)} vectors from namespace '{namespace}'")

    except Exception as e:
        logger.error(f"Failed to delete vectors: {e}")
        raise VectorStoreError(f"Vector deletion failed: {e}")

def get_index_stats(cfg) -> Dict[str, Any]:
    """
    Get index statistics and health information.

    Args:
        cfg: Configuration object

    Returns:
        Dictionary with index statistics
    """
    config = _get_config_from_env()
    client_wrapper = EnhancedPineconeClient(config)

    try:
        pc = client_wrapper.client
        index_info = pc.describe_index(config.index_name)
        index_stats = pc.describe_index_stats(config.index_name)

        return {
            "index_name": config.index_name,
            "dimension": index_info.dimension,
            "metric": index_info.metric,
            "status": index_info.status,
            "total_vectors": index_stats.total_vector_count,
            "namespaces": list(index_stats.namespaces.keys()) if index_stats.namespaces else [],
            "last_updated": time.time()
        }

    except Exception as e:
        logger.error(f"Failed to get index stats: {e}")
        raise VectorStoreError(f"Index stats retrieval failed: {e}")

# Backward compatibility functions
def _client(cfg) -> Pinecone:
    """Legacy client function for backward compatibility"""
    config = _get_config_from_env()
    client_wrapper = EnhancedPineconeClient(config)
    return client_wrapper.client
