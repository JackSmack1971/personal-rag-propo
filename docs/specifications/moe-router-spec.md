# Expert Router Detailed Specification

## Document Information
- **Document ID:** MOE-ROUTER-SPEC-001
- **Version:** 1.0.0
- **Created:** 2025-08-30
- **Last Updated:** 2025-08-30
- **Status:** Draft

## Overview

The Expert Router is a core component of the Mixture of Experts (MoE) system responsible for intelligently routing user queries to the most relevant retrieval experts based on semantic similarity calculations.

## 1. Functional Requirements

### 1.1 Core Functionality

**FR-ROUTER-001**: The Expert Router SHALL accept a query embedding and return a ranked list of relevant experts with confidence scores.

**FR-ROUTER-002**: The Expert Router SHALL maintain expert centroids representing the semantic center of each expert's knowledge domain.

**FR-ROUTER-003**: The Expert Router SHALL calculate cosine similarity between query embeddings and expert centroids.

**FR-ROUTER-004**: The Expert Router SHALL support configurable top-K expert selection.

**FR-ROUTER-005**: The Expert Router SHALL provide routing confidence scores for downstream decision making.

### 1.2 Expert Management

**FR-ROUTER-006**: The Expert Router SHALL support dynamic expert configuration through YAML configuration files.

**FR-ROUTER-007**: The Expert Router SHALL maintain expert centroids with automatic refresh capabilities.

**FR-ROUTER-008**: The Expert Router SHALL track expert metadata including document count and last update timestamp.

### 1.3 Performance Requirements

**FR-ROUTER-009**: Expert Router operations SHALL complete in under 10ms for typical query embeddings.

**FR-ROUTER-010**: The Expert Router SHALL support concurrent query processing without race conditions.

**FR-ROUTER-011**: Memory usage SHALL remain under 50MB for expert centroids and metadata.

## 2. Interface Specifications

### 2.1 Class Definition

```python
class ExpertRouter:
    """Production expert router with centroid management"""

    def __init__(self, config: AppConfig):
        """Initialize router with configuration"""
        pass

    def route_query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 2
    ) -> Tuple[List[str], Dict[str, float]]:
        """Route query to top-k experts with similarities"""
        pass

    def update_centroids(
        self,
        expert_embeddings: Dict[str, List[np.ndarray]]
    ) -> None:
        """Update expert centroids from recent embeddings"""
        pass

    def get_expert_info(self, expert_id: str) -> Optional[ExpertCentroid]:
        """Get metadata for specific expert"""
        pass

    def list_experts(self) -> List[str]:
        """List all available experts"""
        pass
```

### 2.2 Data Structures

```python
@dataclass
class ExpertCentroid:
    """Expert centroid with metadata"""
    expert_id: str
    centroid: np.ndarray
    document_count: int
    last_updated: float
    confidence_score: float

@dataclass
class RoutingResult:
    """Result of expert routing operation"""
    query_id: str
    chosen_experts: List[str]
    similarities: Dict[str, float]
    routing_confidence: float
    processing_time_ms: float
    timestamp: float
```

### 2.3 Configuration Schema

```yaml
moe:
  router:
    enabled: true
    experts: ["general", "technical", "personal", "code"]
    centroid_refresh_interval: 3600  # seconds
    top_k_experts: 2
    similarity_threshold: 0.3  # minimum similarity for routing
    confidence_decay_factor: 0.95  # confidence decay over time
```

## 3. Algorithm Specifications

### 3.1 Similarity Calculation

**Algorithm: Cosine Similarity Computation**

```python
def _calculate_similarity(
    self,
    query_embedding: np.ndarray,
    expert_centroid: np.ndarray
) -> float:
    """
    Calculate cosine similarity between query and expert centroid.

    Formula: similarity = (query • centroid) / (||query|| * ||centroid||)

    Args:
        query_embedding: Normalized query embedding vector
        expert_centroid: Normalized expert centroid vector

    Returns:
        float: Cosine similarity score in range [-1, 1]
    """
    # Input validation
    if query_embedding.shape != expert_centroid.shape:
        raise ValueError("Embedding dimensions must match")

    # Cosine similarity calculation
    dot_product = np.dot(query_embedding, expert_centroid)
    query_norm = np.linalg.norm(query_embedding)
    centroid_norm = np.linalg.norm(expert_centroid)

    # Handle zero vectors
    if query_norm == 0 or centroid_norm == 0:
        return 0.0

    similarity = dot_product / (query_norm * centroid_norm)

    # Clamp to valid range (numerical stability)
    return max(-1.0, min(1.0, similarity))
```

**Performance Characteristics**:
- **Time Complexity**: O(d) where d is embedding dimension (typically 384-768)
- **Space Complexity**: O(d) for vector operations
- **Numerical Stability**: Handles edge cases (zero vectors, NaN values)

### 3.2 Expert Selection

**Algorithm: Top-K Expert Selection with Confidence**

```python
def _select_top_k_experts(
    self,
    similarities: Dict[str, float],
    top_k: int
) -> Tuple[List[str], Dict[str, float]]:
    """
    Select top-K experts based on similarity scores.

    Process:
    1. Filter experts above similarity threshold
    2. Sort by similarity (descending)
    3. Select top-K experts
    4. Calculate routing confidence

    Args:
        similarities: Dict mapping expert_id to similarity score
        top_k: Number of experts to select

    Returns:
        Tuple of (selected_experts, filtered_similarities)
    """
    # Filter by threshold
    filtered_similarities = {
        expert_id: sim for expert_id, sim in similarities.items()
        if sim >= self.config.moe.router.similarity_threshold
    }

    if not filtered_similarities:
        # Fallback: select experts with highest similarities
        logger.warning("No experts above threshold, using fallback selection")
        filtered_similarities = similarities

    # Sort by similarity (descending)
    sorted_experts = sorted(
        filtered_similarities.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # Select top-K
    selected_experts = [expert_id for expert_id, _ in sorted_experts[:top_k]]

    # Calculate routing confidence
    if selected_experts:
        max_similarity = max(filtered_similarities[expert_id] for expert_id in selected_experts)
        routing_confidence = min(max_similarity, 1.0)  # Cap at 1.0
    else:
        routing_confidence = 0.0

    return selected_experts, dict(sorted_experts[:top_k])
```

### 3.3 Centroid Update Algorithm

**Algorithm: Incremental Centroid Update**

```python
def _update_centroid_incremental(
    self,
    expert_id: str,
    new_embeddings: List[np.ndarray]
) -> None:
    """
    Update expert centroid using incremental learning.

    Formula: new_centroid = (old_centroid * old_count + sum(new_embeddings)) / new_count

    Process:
    1. Retrieve current centroid and count
    2. Calculate new centroid using weighted average
    3. Update metadata (count, timestamp, confidence)
    4. Normalize centroid vector

    Args:
        expert_id: Identifier of expert to update
        new_embeddings: List of new embedding vectors for this expert
    """
    if not new_embeddings:
        return

    current_centroid = self.centroids.get(expert_id)
    if current_centroid is None:
        # Initialize new centroid
        new_centroid_vector = np.mean(new_embeddings, axis=0)
        confidence = min(len(new_embeddings) / 100.0, 1.0)
        self.centroids[expert_id] = ExpertCentroid(
            expert_id=expert_id,
            centroid=self._normalize_vector(new_centroid_vector),
            document_count=len(new_embeddings),
            last_updated=time.time(),
            confidence_score=confidence
        )
        return

    # Incremental update
    old_centroid = current_centroid.centroid
    old_count = current_centroid.document_count
    new_count = len(new_embeddings)

    # Weighted average: (old * old_count + new_sum) / total_count
    new_embeddings_sum = np.sum(new_embeddings, axis=0)
    total_count = old_count + new_count

    updated_centroid = (
        old_centroid * old_count + new_embeddings_sum
    ) / total_count

    # Update confidence based on total sample size
    confidence = min(total_count / 1000.0, 1.0)  # Confidence increases with samples

    # Update centroid
    self.centroids[expert_id] = ExpertCentroid(
        expert_id=expert_id,
        centroid=self._normalize_vector(updated_centroid),
        document_count=total_count,
        last_updated=time.time(),
        confidence_score=confidence
    )
```

## 4. Error Handling & Resilience

### 4.1 Error Scenarios

| Error Scenario | Handling Strategy | Fallback Behavior |
|----------------|-------------------|-------------------|
| Empty centroids | Log warning | Use default expert routing |
| Invalid embeddings | Input validation | Return empty routing result |
| Similarity calculation failure | Exception handling | Return zero similarity |
| Centroid update failure | Transaction rollback | Skip update, log error |
| Memory allocation error | Garbage collection | Reduce centroid precision |

### 4.2 Validation Checks

```python
def _validate_inputs(self, query_embedding: np.ndarray) -> bool:
    """Validate query embedding input"""
    if query_embedding is None:
        raise ValueError("Query embedding cannot be None")

    if len(query_embedding.shape) != 1:
        raise ValueError("Query embedding must be 1-dimensional")

    if query_embedding.shape[0] == 0:
        raise ValueError("Query embedding cannot be empty")

    # Check for NaN or infinite values
    if not np.isfinite(query_embedding).all():
        raise ValueError("Query embedding contains NaN or infinite values")

    return True
```

### 4.3 Logging & Monitoring

```python
def _log_routing_decision(
    self,
    query_id: str,
    chosen_experts: List[str],
    similarities: Dict[str, float],
    processing_time: float
) -> None:
    """Log routing decision for monitoring"""
    logger.info(
        f"Query {query_id} routed to {chosen_experts} "
        f"with similarities {similarities} "
        f"in {processing_time:.2f}ms"
    )

    # Log performance metrics
    if processing_time > 10.0:  # ms
        logger.warning(f"Slow routing for query {query_id}: {processing_time:.2f}ms")

    # Log low confidence routings
    max_similarity = max(similarities.values()) if similarities else 0.0
    if max_similarity < 0.5:
        logger.debug(f"Low confidence routing for query {query_id}: {max_similarity:.3f}")
```

## 5. Performance Optimization

### 5.1 Memory Optimization

**Technique: Vector Normalization Caching**
```python
def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
    """Normalize vector with caching for repeated operations"""
    # Cache normalized centroids to avoid repeated calculations
    vector_key = hash(vector.tobytes())
    if vector_key in self._normalization_cache:
        return self._normalization_cache[vector_key]

    norm = np.linalg.norm(vector)
    if norm == 0:
        normalized = np.zeros_like(vector)
    else:
        normalized = vector / norm

    # Cache with LRU eviction
    if len(self._normalization_cache) > 1000:
        self._normalization_cache.pop(next(iter(self._normalization_cache)))

    self._normalization_cache[vector_key] = normalized
    return normalized
```

**Technique: Expert Centroid Sharding**
```python
def _get_expert_shard(self, expert_id: str) -> str:
    """Determine which shard contains the expert centroid"""
    # Simple hash-based sharding for memory distribution
    shard_count = 4  # Configurable
    shard = hash(expert_id) % shard_count
    return f"shard_{shard}"
```

### 5.2 Computational Optimization

**Technique: Batch Similarity Calculation**
```python
def _batch_similarity_calculation(
    self,
    query_embedding: np.ndarray,
    expert_centroids: Dict[str, np.ndarray]
) -> Dict[str, float]:
    """Calculate similarities for multiple experts in batch"""
    if not expert_centroids:
        return {}

    # Stack centroids into matrix for vectorized operations
    centroid_matrix = np.stack(list(expert_centroids.values()))
    expert_ids = list(expert_centroids.keys())

    # Vectorized cosine similarity
    query_norm = np.linalg.norm(query_embedding)
    centroid_norms = np.linalg.norm(centroid_matrix, axis=1)

    # Handle zero norms
    valid_mask = (query_norm > 0) & (centroid_norms > 0)
    similarities = np.zeros(len(expert_ids))

    if np.any(valid_mask):
        dot_products = np.dot(centroid_matrix[valid_mask], query_embedding)
        similarities[valid_mask] = dot_products / (centroid_norms[valid_mask] * query_norm)

    # Clamp to valid range
    similarities = np.clip(similarities, -1.0, 1.0)

    return dict(zip(expert_ids, similarities))
```

### 5.3 Caching Strategy

```python
class RoutingCache:
    """LRU cache for routing decisions"""

    def __init__(self, max_size: int = 10000):
        self.cache = {}
        self.max_size = max_size
        self.access_order = []

    def get(self, query_embedding_hash: str) -> Optional[RoutingResult]:
        """Retrieve cached routing result"""
        if query_embedding_hash in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(query_embedding_hash)
            self.access_order.append(query_embedding_hash)
            return self.cache[query_embedding_hash]
        return None

    def put(self, query_embedding_hash: str, result: RoutingResult) -> None:
        """Store routing result in cache"""
        if len(self.cache) >= self.max_size:
            # Evict least recently used
            lru_key = self.access_order.pop(0)
            del self.cache[lru_key]

        self.cache[query_embedding_hash] = result
        self.access_order.append(query_embedding_hash)
```

## 6. Testing Specifications

### 6.1 Unit Tests

```python
class TestExpertRouter:

    def test_similarity_calculation(self):
        """Test cosine similarity computation"""
        router = ExpertRouter(self.config)

        # Test identical vectors
        vec1 = np.array([1.0, 0.0, 0.0])
        assert router._calculate_similarity(vec1, vec1) == pytest.approx(1.0)

        # Test orthogonal vectors
        vec2 = np.array([0.0, 1.0, 0.0])
        assert router._calculate_similarity(vec1, vec2) == pytest.approx(0.0)

        # Test opposite vectors
        vec3 = np.array([-1.0, 0.0, 0.0])
        assert router._calculate_similarity(vec1, vec3) == pytest.approx(-1.0)

    def test_expert_selection(self):
        """Test top-K expert selection"""
        router = ExpertRouter(self.config)

        similarities = {
            "expert_a": 0.8,
            "expert_b": 0.6,
            "expert_c": 0.4,
            "expert_d": 0.2
        }

        chosen, selected_sims = router._select_top_k_experts(similarities, 2)

        assert len(chosen) == 2
        assert chosen == ["expert_a", "expert_b"]
        assert selected_sims["expert_a"] == 0.8
        assert selected_sims["expert_b"] == 0.6

    def test_centroid_update(self):
        """Test incremental centroid updates"""
        router = ExpertRouter(self.config)

        # Initial embeddings
        embeddings1 = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
        router.update_centroids({"expert_1": embeddings1})

        initial_centroid = router.centroids["expert_1"].centroid
        expected_initial = np.array([0.5, 0.5])  # Mean of inputs
        assert np.allclose(initial_centroid, expected_initial)

        # Additional embeddings
        embeddings2 = [np.array([1.0, 1.0])]
        router.update_centroids({"expert_1": embeddings2})

        updated_centroid = router.centroids["expert_1"].centroid
        # Expected: ([0.5, 0.5] * 2 + [1.0, 1.0]) / 3 = [2/3, 2/3, 1/3]
        expected_updated = np.array([2/3, 2/3])
        assert np.allclose(updated_centroid, expected_updated)
```

### 6.2 Integration Tests

```python
class TestExpertRouterIntegration:

    def test_full_routing_pipeline(self):
        """Test complete routing pipeline"""
        router = ExpertRouter(self.config)

        # Setup test centroids
        test_embeddings = {
            "general": [np.array([1.0, 0.0, 0.0])],
            "technical": [np.array([0.0, 1.0, 0.0])],
            "personal": [np.array([0.0, 0.0, 1.0])]
        }
        router.update_centroids(test_embeddings)

        # Test routing
        query_embedding = np.array([0.9, 0.1, 0.0])  # Similar to "general"
        chosen_experts, similarities = router.route_query(query_embedding, top_k=2)

        assert "general" in chosen_experts
        assert similarities["general"] > similarities["technical"]
        assert similarities["general"] > similarities["personal"]

    def test_performance_requirements(self):
        """Test performance meets requirements"""
        router = ExpertRouter(self.config)

        # Setup multiple centroids
        for i in range(10):
            embeddings = [np.random.randn(384) for _ in range(5)]
            router.update_centroids({f"expert_{i}": embeddings})

        query_embedding = np.random.randn(384)

        # Measure performance
        start_time = time.time()
        for _ in range(100):
            router.route_query(query_embedding)
        end_time = time.time()

        avg_time = (end_time - start_time) / 100 * 1000  # ms
        assert avg_time < 10.0  # Less than 10ms requirement
```

### 6.3 Edge Case Tests

```python
class TestExpertRouterEdgeCases:

    def test_empty_centroids(self):
        """Test behavior with no centroids"""
        router = ExpertRouter(self.config)

        query_embedding = np.array([1.0, 0.0, 0.0])
        chosen_experts, similarities = router.route_query(query_embedding)

        # Should return default experts or empty list
        assert isinstance(chosen_experts, list)
        assert isinstance(similarities, dict)

    def test_single_expert(self):
        """Test routing with single expert"""
        router = ExpertRouter(self.config)

        embeddings = [np.array([1.0, 0.0, 0.0])]
        router.update_centroids({"only_expert": embeddings})

        query_embedding = np.array([1.0, 0.0, 0.0])
        chosen_experts, similarities = router.route_query(query_embedding, top_k=5)

        assert len(chosen_experts) == 1
        assert chosen_experts[0] == "only_expert"

    def test_zero_similarity(self):
        """Test routing with orthogonal vectors"""
        router = ExpertRouter(self.config)

        embeddings = [np.array([1.0, 0.0, 0.0])]
        router.update_centroids({"expert_1": embeddings})

        query_embedding = np.array([0.0, 1.0, 0.0])  # Orthogonal
        chosen_experts, similarities = router.route_query(query_embedding)

        assert similarities["expert_1"] == pytest.approx(0.0, abs=1e-6)
```

## 7. Monitoring & Observability

### 7.1 Metrics Collection

```python
class RouterMetrics:
    """Metrics collection for expert router"""

    def __init__(self):
        self.routing_requests = 0
        self.routing_latency_ms = []
        self.similarity_distribution = []
        self.cache_hit_rate = 0.0
        self.error_count = 0

    def record_routing_request(
        self,
        query_id: str,
        latency_ms: float,
        similarities: Dict[str, float],
        cache_hit: bool
    ):
        """Record routing request metrics"""
        self.routing_requests += 1
        self.routing_latency_ms.append(latency_ms)

        for similarity in similarities.values():
            self.similarity_distribution.append(similarity)

        # Keep only recent metrics
        if len(self.routing_latency_ms) > 1000:
            self.routing_latency_ms = self.routing_latency_ms[-1000:]

        if len(self.similarity_distribution) > 10000:
            self.similarity_distribution = self.similarity_distribution[-10000:]

    def get_summary_stats(self) -> Dict[str, float]:
        """Get summary statistics"""
        return {
            "total_requests": self.routing_requests,
            "avg_latency_ms": np.mean(self.routing_latency_ms) if self.routing_latency_ms else 0.0,
            "p95_latency_ms": np.percentile(self.routing_latency_ms, 95) if self.routing_latency_ms else 0.0,
            "avg_similarity": np.mean(self.similarity_distribution) if self.similarity_distribution else 0.0,
            "cache_hit_rate": self.cache_hit_rate,
            "error_rate": self.error_count / max(self.routing_requests, 1)
        }
```

### 7.2 Health Checks

```python
def health_check(self) -> Dict[str, Any]:
    """Router health check"""
    health_status = {
        "component": "expert_router",
        "status": "healthy",
        "timestamp": time.time(),
        "metrics": self.metrics.get_summary_stats(),
        "centroid_count": len(self.centroids),
        "total_documents": sum(c.document_count for c in self.centroids.values()),
        "last_centroid_update": max((c.last_updated for c in self.centroids.values()), default=0),
        "memory_usage_mb": self._get_memory_usage()
    }

    # Check for unhealthy conditions
    if len(self.centroids) == 0:
        health_status["status"] = "degraded"
        health_status["issues"] = ["No expert centroids available"]

    if health_status["metrics"]["avg_latency_ms"] > 50.0:
        health_status["status"] = "degraded"
        health_status["issues"] = health_status.get("issues", []) + ["High latency detected"]

    return health_status
```

## 8. Configuration & Deployment

### 8.1 Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | true | Enable/disable expert routing |
| `experts` | List[str] | ["general", "technical", "personal"] | Available experts |
| `centroid_refresh_interval` | int | 3600 | Centroid update interval (seconds) |
| `top_k_experts` | int | 2 | Default number of experts to route to |
| `similarity_threshold` | float | 0.3 | Minimum similarity for routing |
| `confidence_decay_factor` | float | 0.95 | Confidence decay over time |
| `cache_size` | int | 10000 | LRU cache size for routing results |

### 8.2 Deployment Checklist

- [ ] Expert definitions configured in YAML
- [ ] Initial centroids populated from existing documents
- [ ] Memory limits configured for centroid storage
- [ ] Performance monitoring enabled
- [ ] Health check endpoints configured
- [ ] Error handling and fallback mechanisms tested
- [ ] Cache size tuned for expected query volume

## 9. Future Enhancements

### 9.1 Advanced Routing Algorithms

**Hierarchical Routing**: Multi-level expert routing with coarse-to-fine granularity
**Temporal Routing**: Time-aware routing considering document recency
**Contextual Routing**: Query context integration for improved routing decisions
**Adaptive Thresholds**: Dynamic similarity thresholds based on expert performance

### 9.2 Scalability Improvements

**Distributed Centroids**: Sharded centroid storage across multiple nodes
**Streaming Updates**: Real-time centroid updates with change detection
**Compression**: Vector compression for reduced memory footprint
**GPU Acceleration**: CUDA support for similarity calculations

### 9.3 Learning Enhancements

**Online Learning**: Continuous centroid adaptation based on user feedback
**Expert Discovery**: Automatic expert identification from document clustering
**Performance-Based Routing**: Route optimization based on historical performance
**Multi-Objective Optimization**: Balance multiple routing criteria (accuracy, latency, diversity)

This detailed specification provides the foundation for implementing a robust, performant Expert Router that serves as the intelligent routing backbone of the MoE system.