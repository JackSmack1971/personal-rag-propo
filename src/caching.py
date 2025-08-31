"""
Advanced Caching System for Personal RAG
Implements multi-level caching for embeddings, queries, and model warm-up mechanisms.
"""

import time
import logging
import hashlib
import threading
import statistics
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import pickle
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    created_at: float
    accessed_at: float
    access_count: int = 0
    size_bytes: int = 0
    ttl: Optional[float] = None  # Time to live in seconds
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if cache entry has expired"""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl

    def touch(self):
        """Update access time and count"""
        self.accessed_at = time.time()
        self.access_count += 1

@dataclass
class CacheStats:
    """Cache performance statistics"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_entries: int = 0
    total_size_bytes: int = 0
    avg_access_time: float = 0.0
    last_cleanup: float = 0.0

class EmbeddingCache:
    """LRU cache for embeddings with intelligent invalidation"""

    def __init__(self, max_size_mb: int = 512, ttl_hours: int = 24):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.default_ttl = ttl_hours * 3600
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: List[str] = []  # For LRU tracking
        self._stats = CacheStats()
        self._lock = threading.RLock()

    def _get_cache_key(self, text: str, model_name: str, **kwargs) -> str:
        """Generate cache key for embedding"""
        content = f"{text}_{model_name}_{sorted(kwargs.items())}"
        return hashlib.md5(content.encode()).hexdigest()

    def _estimate_size(self, obj: Any) -> int:
        """Estimate memory size of object"""
        if isinstance(obj, np.ndarray):
            return obj.nbytes
        elif isinstance(obj, (list, tuple)):
            return sum(self._estimate_size(item) for item in obj)
        elif isinstance(obj, dict):
            return sum(self._estimate_size(k) + self._estimate_size(v) for k, v in obj.items())
        else:
            return len(pickle.dumps(obj))

    def _evict_lru(self):
        """Evict least recently used entries to free memory"""
        while self._stats.total_size_bytes > self.max_size_bytes and self._access_order:
            oldest_key = self._access_order.pop(0)
            if oldest_key in self._cache:
                entry = self._cache[oldest_key]
                self._stats.total_size_bytes -= entry.size_bytes
                self._stats.evictions += 1
                del self._cache[oldest_key]
                logger.debug(f"Evicted cache entry: {oldest_key}")

    def get(self, text: str, model_name: str, **kwargs) -> Optional[np.ndarray]:
        """Get cached embedding"""
        with self._lock:
            key = self._get_cache_key(text, model_name, **kwargs)

            if key in self._cache:
                entry = self._cache[key]
                if entry.is_expired():
                    self.invalidate(key)
                    self._stats.misses += 1
                    return None

                entry.touch()
                # Move to end of access order (most recently used)
                if key in self._access_order:
                    self._access_order.remove(key)
                self._access_order.append(key)

                self._stats.hits += 1
                logger.debug(f"Cache hit for key: {key}")
                return entry.value
            else:
                self._stats.misses += 1
                return None

    def put(self, text: str, model_name: str, embedding: np.ndarray, **kwargs):
        """Cache embedding"""
        with self._lock:
            key = self._get_cache_key(text, model_name, **kwargs)

            # Remove existing entry if present
            if key in self._cache:
                old_entry = self._cache[key]
                self._stats.total_size_bytes -= old_entry.size_bytes
                if key in self._access_order:
                    self._access_order.remove(key)

            # Create new entry
            size_bytes = self._estimate_size(embedding)
            entry = CacheEntry(
                key=key,
                value=embedding,
                created_at=time.time(),
                accessed_at=time.time(),
                size_bytes=size_bytes,
                ttl=self.default_ttl,
                metadata={"model": model_name, "text_length": len(text)}
            )

            self._cache[key] = entry
            self._access_order.append(key)
            self._stats.total_size_bytes += size_bytes
            self._stats.total_entries = len(self._cache)

            # Evict if necessary
            self._evict_lru()

            logger.debug(f"Cached embedding for key: {key}, size: {size_bytes} bytes")

    def invalidate(self, key: str):
        """Invalidate specific cache entry"""
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                self._stats.total_size_bytes -= entry.size_bytes
                self._stats.evictions += 1
                del self._cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)
                logger.debug(f"Invalidated cache entry: {key}")

    def clear(self):
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._stats = CacheStats()
            logger.info("Cleared embedding cache")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            hit_rate = self._stats.hits / (self._stats.hits + self._stats.misses) if (self._stats.hits + self._stats.misses) > 0 else 0
            return {
                "total_entries": self._stats.total_entries,
                "total_size_mb": self._stats.total_size_bytes / (1024 * 1024),
                "max_size_mb": self.max_size_bytes / (1024 * 1024),
                "hit_rate": hit_rate,
                "hits": self._stats.hits,
                "misses": self._stats.misses,
                "evictions": self._stats.evictions,
                "utilization_percent": (self._stats.total_size_bytes / self.max_size_bytes) * 100 if self.max_size_bytes > 0 else 0
            }

class QueryResultCache:
    """Cache for RAG query results with semantic similarity matching"""

    def __init__(self, max_entries: int = 1000, similarity_threshold: float = 0.85):
        self.max_entries = max_entries
        self.similarity_threshold = similarity_threshold
        self._cache: Dict[str, CacheEntry] = {}
        self._query_embeddings: Dict[str, np.ndarray] = {}  # Store query embeddings for similarity
        self._stats = CacheStats()
        self._lock = threading.RLock()

    def _get_cache_key(self, query: str) -> str:
        """Generate cache key for query"""
        return hashlib.md5(query.encode()).hexdigest()

    def _compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between embeddings"""
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(emb1, emb2) / (norm1 * norm2))

    def get_similar(self, query: str, query_embedding: np.ndarray) -> Optional[Any]:
        """Get cached result for similar query"""
        with self._lock:
            # Find most similar cached query
            best_match = None
            best_similarity = 0.0

            for cached_key, cached_emb in self._query_embeddings.items():
                if cached_key in self._cache:
                    similarity = self._compute_similarity(query_embedding, cached_emb)
                    if similarity > best_similarity and similarity >= self.similarity_threshold:
                        best_similarity = similarity
                        best_match = cached_key

            if best_match and best_match in self._cache:
                entry = self._cache[best_match]
                if not entry.is_expired():
                    entry.touch()
                    self._stats.hits += 1
                    logger.debug(f"Cache hit for similar query (similarity: {best_similarity:.3f})")
                    return entry.value

            self._stats.misses += 1
            return None

    def put(self, query: str, query_embedding: np.ndarray, result: Any, ttl_seconds: int = 3600):
        """Cache query result"""
        with self._lock:
            key = self._get_cache_key(query)

            # Remove existing entry if present
            if key in self._cache:
                del self._cache[key]
                if key in self._query_embeddings:
                    del self._query_embeddings[key]

            # Evict oldest if at capacity
            if len(self._cache) >= self.max_entries:
                oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k].accessed_at)
                del self._cache[oldest_key]
                if oldest_key in self._query_embeddings:
                    del self._query_embeddings[oldest_key]
                self._stats.evictions += 1

            # Create new entry
            size_bytes = len(pickle.dumps(result))
            entry = CacheEntry(
                key=key,
                value=result,
                created_at=time.time(),
                accessed_at=time.time(),
                size_bytes=size_bytes,
                ttl=ttl_seconds,
                metadata={"query_length": len(query)}
            )

            self._cache[key] = entry
            self._query_embeddings[key] = query_embedding.copy()
            self._stats.total_entries = len(self._cache)

            logger.debug(f"Cached query result for key: {key}")

    def invalidate_expired(self):
        """Invalidate expired entries"""
        with self._lock:
            expired_keys = [k for k, v in self._cache.items() if v.is_expired()]
            for key in expired_keys:
                del self._cache[key]
                if key in self._query_embeddings:
                    del self._query_embeddings[key]
                self._stats.evictions += 1

            if expired_keys:
                logger.debug(f"Invalidated {len(expired_keys)} expired cache entries")

    def clear(self):
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()
            self._query_embeddings.clear()
            self._stats = CacheStats()
            logger.info("Cleared query result cache")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            hit_rate = self._stats.hits / (self._stats.hits + self._stats.misses) if (self._stats.hits + self._stats.misses) > 0 else 0
            return {
                "total_entries": self._stats.total_entries,
                "hit_rate": hit_rate,
                "hits": self._stats.hits,
                "misses": self._stats.misses,
                "evictions": self._stats.evictions,
                "similarity_threshold": self.similarity_threshold
            }

class ModelWarmUpManager:
    """Manages model warm-up to improve initial performance"""

    def __init__(self):
        self._warmed_models: Dict[str, float] = {}  # model_key -> warm_up_time
        self._warm_up_samples = [
            "This is a sample sentence for model warm-up.",
            "Another example sentence to prepare the model.",
            "Warm-up text to optimize initial inference performance.",
            "Sample content for neural network initialization.",
            "Test sentence for model preparation and optimization."
        ]

    def warm_up_model(self, embedder, model_name: str, backend: str = "torch") -> float:
        """Warm up model with sample encodings"""
        model_key = f"{model_name}_{backend}"

        if model_key in self._warmed_models:
            logger.debug(f"Model {model_key} already warmed up")
            return 0.0

        start_time = time.time()

        try:
            # Perform warm-up encodings
            for i, sample in enumerate(self._warm_up_samples):
                batch_size = min(i + 1, 4)  # Increasing batch sizes
                embedder.encode([sample], batch_size=batch_size, show_progress_bar=False)

            warm_up_time = time.time() - start_time
            self._warmed_models[model_key] = warm_up_time

            logger.info(f"Warmed up model {model_key} in {warm_up_time:.3f}s")
            return warm_up_time

        except Exception as e:
            logger.error(f"Model warm-up failed for {model_key}: {e}")
            return 0.0

    def is_warmed_up(self, model_name: str, backend: str = "torch") -> bool:
        """Check if model has been warmed up"""
        model_key = f"{model_name}_{backend}"
        return model_key in self._warmed_models

    def get_warm_up_stats(self) -> Dict[str, Any]:
        """Get warm-up statistics"""
        return {
            "warmed_models": list(self._warmed_models.keys()),
            "total_warm_up_time": sum(self._warmed_models.values()),
            "average_warm_up_time": statistics.mean(self._warmed_models.values()) if self._warmed_models else 0
        }

# Global cache instances
embedding_cache = EmbeddingCache()
query_cache = QueryResultCache()
warm_up_manager = ModelWarmUpManager()

def get_cache_stats() -> Dict[str, Any]:
    """Get comprehensive cache statistics"""
    return {
        "embedding_cache": embedding_cache.get_stats(),
        "query_cache": query_cache.get_stats(),
        "warm_up_manager": warm_up_manager.get_warm_up_stats()
    }

def clear_all_caches():
    """Clear all caches"""
    embedding_cache.clear()
    query_cache.clear()
    logger.info("Cleared all caches")

def cleanup_expired_entries():
    """Clean up expired cache entries"""
    query_cache.invalidate_expired()
    logger.debug("Cleaned up expired cache entries")