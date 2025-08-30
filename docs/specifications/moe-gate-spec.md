# Selective Gate Detailed Specification

## Document Information
- **Document ID:** MOE-GATE-SPEC-001
- **Version:** 1.0.0
- **Created:** 2025-08-30
- **Last Updated:** 2025-08-30
- **Status:** Draft

## Overview

The Selective Gate is a critical decision-making component in the Mixture of Experts (MoE) system that determines whether vector retrieval is necessary and selects optimal retrieval parameters based on routing confidence and query characteristics.

## 1. Functional Requirements

### 1.1 Core Decision Making

**FR-GATE-001**: The Selective Gate SHALL evaluate routing confidence to determine if retrieval is necessary.

**FR-GATE-002**: The Selective Gate SHALL select optimal retrieval parameters (k-value) based on confidence levels.

**FR-GATE-003**: The Selective Gate SHALL apply dynamic score filtering to retrieved results.

**FR-GATE-004**: The Selective Gate SHALL provide fallback mechanisms when confidence is insufficient.

**FR-GATE-005**: The Selective Gate SHALL support configurable threshold parameters.

### 1.2 Performance Optimization

**FR-GATE-006**: Selective Gate decisions SHALL complete in under 5ms.

**FR-GATE-007**: The Selective Gate SHALL minimize unnecessary retrieval operations.

**FR-GATE-008**: The Selective Gate SHALL balance retrieval quality with computational efficiency.

### 1.3 Adaptive Behavior

**FR-GATE-009**: The Selective Gate SHALL adapt retrieval breadth based on confidence levels.

**FR-GATE-010**: The Selective Gate SHALL support dynamic threshold adjustment.

**FR-GATE-011**: The Selective Gate SHALL provide decision transparency through logging.

## 2. Interface Specifications

### 2.1 Class Definition

```python
class SelectiveGate:
    """Intelligent retrieval gating with adaptive k-selection"""

    def __init__(self, config: AppConfig):
        """Initialize gate with configuration"""
        pass

    def should_retrieve_and_k(
        self,
        router_similarities: Dict[str, float],
        query_complexity_score: float = 0.5
    ) -> Tuple[bool, int]:
        """Decide whether to retrieve and choose optimal k"""
        pass

    def apply_score_filtering(
        self,
        matches: List[Dict],
        query_embedding: np.ndarray
    ) -> List[Dict]:
        """Apply dynamic score-based filtering"""
        pass

    def update_thresholds(
        self,
        performance_metrics: Dict[str, float]
    ) -> None:
        """Update thresholds based on performance feedback"""
        pass

    def get_gate_statistics(self) -> Dict[str, Any]:
        """Get gate decision statistics"""
        pass
```

### 2.2 Data Structures

```python
@dataclass
class GateDecision:
    """Result of gate decision process"""
    should_retrieve: bool
    optimal_k: int
    confidence_level: float
    decision_reason: str
    processing_time_ms: float
    timestamp: float

@dataclass
class FilteringResult:
    """Result of score-based filtering"""
    original_count: int
    filtered_count: int
    min_score: float
    max_score: float
    avg_score: float
    filtering_threshold: float
    processing_time_ms: float
```

### 2.3 Configuration Schema

```yaml
moe:
  gate:
    enabled: true

    # Retrieval decision thresholds
    retrieve_sim_threshold: 0.62    # Skip retrieval above this confidence
    low_sim_threshold: 0.45         # Use max k below this confidence

    # Retrieval parameters
    k_min: 4                        # Minimum retrieval count
    k_max: 15                       # Maximum retrieval count
    default_top_k: 8                # Standard retrieval count

    # Score filtering thresholds
    high_score_cutoff: 0.8          # Strict filtering threshold
    low_score_cutoff: 0.5           # Lenient filtering threshold

    # Adaptive parameters
    confidence_weight: 0.7          # Weight for confidence in decisions
    complexity_weight: 0.3          # Weight for query complexity
    adaptation_rate: 0.01           # Rate of threshold adaptation
```

## 3. Decision Algorithms

### 3.1 Retrieval Decision Logic

**Algorithm: Confidence-Based Retrieval Decision**

```python
def should_retrieve_and_k(
    self,
    router_similarities: Dict[str, float],
    query_complexity_score: float = 0.5
) -> Tuple[bool, int]:
    """
    Determine retrieval necessity and optimal k based on routing confidence.

    Decision Process:
    1. Calculate overall routing confidence from similarities
    2. Apply complexity adjustment to confidence
    3. Compare against retrieval threshold
    4. Select k based on confidence level

    Args:
        router_similarities: Similarities from expert router
        query_complexity_score: Query complexity (0.0-1.0)

    Returns:
        Tuple of (should_retrieve, optimal_k)
    """

    if not router_similarities:
        logger.debug("No similarities provided, using default retrieval")
        return True, self.config.moe.gate.default_top_k

    # Calculate maximum similarity as primary confidence indicator
    max_similarity = max(router_similarities.values())

    # Apply complexity adjustment
    # Higher complexity reduces effective confidence (need more retrieval)
    adjusted_confidence = max_similarity * (1.0 - query_complexity_score * self.config.moe.gate.complexity_weight)

    # Decision: retrieve if confidence is below threshold
    should_retrieve = adjusted_confidence < self.config.moe.gate.retrieve_sim_threshold

    if not should_retrieve:
        logger.debug(".3f")
        return False, self.config.moe.gate.k_min

    # Adaptive k selection based on confidence level
    optimal_k = self._select_optimal_k(adjusted_confidence)

    logger.debug(
        f"Gate decision: retrieve={should_retrieve}, k={optimal_k}, "
        ".3f"
    )

    return should_retrieve, optimal_k
```

**Algorithm: Adaptive K-Selection**

```python
def _select_optimal_k(self, confidence: float) -> int:
    """
    Select optimal retrieval count based on confidence level.

    Selection Logic:
    - High confidence: Use minimum k (efficient)
    - Medium confidence: Use default k (balanced)
    - Low confidence: Use maximum k (comprehensive)

    Args:
        confidence: Adjusted confidence score (0.0-1.0)

    Returns:
        Optimal k value for retrieval
    """

    if confidence < self.config.moe.gate.low_sim_threshold:
        # Low confidence: cast wide net
        k = self.config.moe.gate.k_max
        reason = "low_confidence"
    elif confidence > self.config.moe.gate.retrieve_sim_threshold:
        # High confidence: minimal retrieval
        k = self.config.moe.gate.k_min
        reason = "high_confidence"
    else:
        # Medium confidence: balanced approach
        k = self.config.moe.gate.default_top_k
        reason = "medium_confidence"

    self._record_decision(k, confidence, reason)
    return k
```

### 3.2 Score-Based Filtering

**Algorithm: Dynamic Score Filtering**

```python
def apply_score_filtering(
    self,
    matches: List[Dict],
    query_embedding: np.ndarray
) -> List[Dict]:
    """
    Apply dynamic filtering based on match scores and distribution.

    Filtering Strategy:
    1. Analyze score distribution
    2. Apply adaptive threshold based on score characteristics
    3. Filter matches while preserving diversity
    4. Ensure minimum result count

    Args:
        matches: Raw retrieval matches with scores
        query_embedding: Query embedding for fallback scoring

    Returns:
        Filtered list of matches
    """

    if not matches:
        return matches

    # Extract scores for analysis
    scores = [match.get('score', 0.0) for match in matches]
    if not scores:
        return matches

    max_score = max(scores)
    min_score = min(scores)
    avg_score = sum(scores) / len(scores)

    # Calculate score statistics
    score_std = np.std(scores) if len(scores) > 1 else 0.0
    score_range = max_score - min_score

    # Adaptive threshold selection
    if max_score >= self.config.moe.gate.high_score_cutoff:
        # High confidence: strict filtering
        threshold = self._calculate_strict_threshold(max_score, score_std)
        reason = "strict_filtering"
    elif max_score <= self.config.moe.gate.low_score_cutoff:
        # Low confidence: lenient filtering
        threshold = self._calculate_lenient_threshold(min_score, avg_score)
        reason = "lenient_filtering"
    else:
        # Medium confidence: moderate filtering
        threshold = self._calculate_moderate_threshold(avg_score, score_std)
        reason = "moderate_filtering"

    # Apply filtering
    filtered_matches = []
    for match in matches:
        score = match.get('score', 0.0)
        if score >= threshold:
            filtered_matches.append(match)

    # Ensure minimum results (fallback to original if needed)
    if len(filtered_matches) < self.config.moe.gate.k_min:
        logger.debug(
            f"Filtering too aggressive ({len(filtered_matches)} < {self.config.moe.gate.k_min}), "
            "using top results"
        )
        filtered_matches = matches[:self.config.moe.gate.k_min]

    # Record filtering statistics
    self._record_filtering_stats(
        original_count=len(matches),
        filtered_count=len(filtered_matches),
        threshold=threshold,
        max_score=max_score,
        reason=reason
    )

    logger.debug(
        f"Applied {reason}: {len(matches)} -> {len(filtered_matches)} matches "
        ".3f"
    )

    return filtered_matches
```

**Algorithm: Adaptive Threshold Calculation**

```python
def _calculate_strict_threshold(self, max_score: float, score_std: float) -> float:
    """Calculate strict filtering threshold for high-confidence scenarios"""
    # Use high percentile with standard deviation adjustment
    base_threshold = self.config.moe.gate.high_score_cutoff
    std_adjustment = score_std * 0.5  # Moderate std adjustment
    return max(base_threshold - std_adjustment, max_score * 0.8)

def _calculate_lenient_threshold(self, min_score: float, avg_score: float) -> float:
    """Calculate lenient filtering threshold for low-confidence scenarios"""
    # Use lower bound with safety margin
    base_threshold = self.config.moe.gate.low_score_cutoff
    return max(min_score, avg_score * 0.7, base_threshold * 0.8)

def _calculate_moderate_threshold(self, avg_score: float, score_std: float) -> float:
    """Calculate moderate filtering threshold for balanced scenarios"""
    # Balance between average and standard deviation
    return avg_score - (score_std * 0.25)
```

## 4. Query Complexity Analysis

### 4.1 Complexity Features

**Algorithm: Query Complexity Scoring**

```python
def _analyze_query_complexity(self, query: str) -> float:
    """
    Analyze query complexity for gate decision adjustment.

    Complexity Factors:
    1. Query length (words, characters)
    2. Lexical diversity (unique words / total words)
    3. Question type indicators (wh-words, how, why)
    4. Technical terminology presence
    5. Multi-part question indicators

    Args:
        query: Raw query string

    Returns:
        Complexity score (0.0-1.0, higher = more complex)
    """

    # Length-based complexity
    words = query.split()
    word_count = len(words)
    char_count = len(query)

    length_score = min((word_count / 20.0) + (char_count / 200.0), 1.0)

    # Lexical diversity
    unique_words = len(set(words))
    diversity_score = unique_words / max(word_count, 1)

    # Question type indicators
    question_words = ['what', 'how', 'why', 'when', 'where', 'which', 'who']
    question_score = 0.0
    for word in words:
        if word.lower() in question_words:
            question_score = 1.0
            break

    # Technical terminology (simple heuristic)
    technical_terms = ['algorithm', 'function', 'method', 'class', 'api', 'database']
    technical_score = 0.0
    for word in words:
        if word.lower() in technical_terms:
            technical_score = 1.0
            break

    # Multi-part question indicators
    multi_part_indicators = [' and ', ' or ', ' vs ', ' versus ', ' compared to ']
    multi_part_score = 0.0
    for indicator in multi_part_indicators:
        if indicator in query.lower():
            multi_part_score = 1.0
            break

    # Combine factors with weights
    complexity_score = (
        length_score * 0.3 +
        diversity_score * 0.2 +
        question_score * 0.2 +
        technical_score * 0.15 +
        multi_part_score * 0.15
    )

    return min(complexity_score, 1.0)
```

## 5. Adaptive Learning

### 5.1 Threshold Adaptation

**Algorithm: Performance-Based Threshold Updates**

```python
def update_thresholds(self, performance_metrics: Dict[str, float]) -> None:
    """
    Adapt gate thresholds based on observed performance.

    Adaptation Strategy:
    1. Monitor retrieval quality metrics
    2. Adjust thresholds to optimize quality-efficiency balance
    3. Apply gradual changes to prevent oscillation
    4. Respect threshold bounds

    Args:
        performance_metrics: Recent performance statistics
    """

    if not performance_metrics:
        return

    # Extract relevant metrics
    avg_relevance = performance_metrics.get('avg_relevance', 0.5)
    retrieval_efficiency = performance_metrics.get('retrieval_efficiency', 0.5)
    false_positive_rate = performance_metrics.get('false_positive_rate', 0.5)

    # Calculate adjustment factors
    quality_factor = (avg_relevance - 0.7) * self.config.moe.gate.adaptation_rate
    efficiency_factor = (retrieval_efficiency - 0.8) * self.config.moe.gate.adaptation_rate
    precision_factor = (0.3 - false_positive_rate) * self.config.moe.gate.adaptation_rate

    # Combined adjustment
    total_adjustment = quality_factor + efficiency_factor + precision_factor

    # Apply adjustments with bounds checking
    old_threshold = self.config.moe.gate.retrieve_sim_threshold

    new_threshold = old_threshold + total_adjustment
    new_threshold = max(0.3, min(0.8, new_threshold))  # Bounds: 0.3-0.8

    if abs(new_threshold - old_threshold) > 0.01:  # Minimum change threshold
        self.config.moe.gate.retrieve_sim_threshold = new_threshold
        logger.info(
            f"Adapted retrieve threshold: {old_threshold:.3f} -> {new_threshold:.3f} "
            f"(quality: {avg_relevance:.3f}, efficiency: {retrieval_efficiency:.3f})"
        )
```

### 5.2 Decision Statistics

```python
class GateStatistics:
    """Statistics for gate decision analysis"""

    def __init__(self):
        self.total_decisions = 0
        self.retrieval_decisions = 0
        self.skip_decisions = 0
        self.k_distribution = defaultdict(int)
        self.confidence_distribution = []
        self.decision_reasons = defaultdict(int)

    def record_decision(
        self,
        should_retrieve: bool,
        k: int,
        confidence: float,
        reason: str
    ):
        """Record gate decision for analysis"""
        self.total_decisions += 1

        if should_retrieve:
            self.retrieval_decisions += 1
        else:
            self.skip_decisions += 1

        self.k_distribution[k] += 1
        self.confidence_distribution.append(confidence)
        self.decision_reasons[reason] += 1

        # Maintain bounded history
        if len(self.confidence_distribution) > 1000:
            self.confidence_distribution = self.confidence_distribution[-1000:]

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive gate statistics"""
        if self.total_decisions == 0:
            return {}

        retrieval_rate = self.retrieval_decisions / self.total_decisions
        avg_confidence = sum(self.confidence_distribution) / len(self.confidence_distribution)

        return {
            "total_decisions": self.total_decisions,
            "retrieval_rate": retrieval_rate,
            "skip_rate": 1.0 - retrieval_rate,
            "avg_confidence": avg_confidence,
            "k_distribution": dict(self.k_distribution),
            "decision_reasons": dict(self.decision_reasons),
            "most_common_k": max(self.k_distribution.keys(), key=self.k_distribution.get) if self.k_distribution else None
        }
```

## 6. Error Handling & Resilience

### 6.1 Error Scenarios

| Error Scenario | Handling Strategy | Fallback Behavior |
|----------------|-------------------|-------------------|
| Empty similarities | Log warning | Default retrieval with standard k |
| Invalid confidence values | Input validation | Use default confidence (0.5) |
| Score filtering failure | Exception handling | Return original matches |
| Complexity analysis failure | Graceful degradation | Use default complexity (0.5) |
| Threshold adaptation failure | Rollback changes | Keep previous thresholds |

### 6.2 Validation Checks

```python
def _validate_inputs(
    self,
    router_similarities: Optional[Dict[str, float]] = None,
    matches: Optional[List[Dict]] = None
) -> bool:
    """Validate inputs for gate operations"""

    if router_similarities is not None:
        for expert_id, similarity in router_similarities.items():
            if not isinstance(similarity, (int, float)):
                raise ValueError(f"Invalid similarity value for {expert_id}: {similarity}")
            if not (-1.0 <= similarity <= 1.0):
                raise ValueError(f"Similarity out of range for {expert_id}: {similarity}")

    if matches is not None:
        for i, match in enumerate(matches):
            if not isinstance(match, dict):
                raise ValueError(f"Invalid match at index {i}: {match}")
            if 'score' in match and not isinstance(match['score'], (int, float)):
                raise ValueError(f"Invalid score in match {i}: {match['score']}")

    return True
```

## 7. Performance Optimization

### 7.1 Computational Efficiency

**Technique: Early Termination**

```python
def _early_termination_check(self, confidence: float) -> bool:
    """Check if decision can be made early to avoid further computation"""
    # If confidence is very high, skip detailed analysis
    if confidence > 0.8:
        return True

    # If confidence is very low, skip complexity analysis
    if confidence < 0.3:
        return True

    return False
```

**Technique: Caching for Repeated Decisions**

```python
class GateCache:
    """Cache for gate decisions to improve performance"""

    def __init__(self, max_size: int = 5000):
        self.cache = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def get_cache_key(
        self,
        similarities_hash: str,
        complexity_score: float
    ) -> str:
        """Generate cache key from inputs"""
        return f"{similarities_hash}:{complexity_score:.2f}"

    def get(self, cache_key: str) -> Optional[GateDecision]:
        """Retrieve cached decision"""
        if cache_key in self.cache:
            self.hits += 1
            return self.cache[cache_key]
        self.hits += 1
        return None

    def put(self, cache_key: str, decision: GateDecision) -> None:
        """Store decision in cache"""
        if len(self.cache) >= self.max_size:
            # Simple eviction: remove random entry
            remove_key = next(iter(self.cache.keys()))
            del self.cache[remove_key]

        self.cache[cache_key] = decision

    def get_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
```

### 7.2 Memory Optimization

**Technique: Bounded Statistics History**

```python
def _maintain_bounded_history(self):
    """Maintain bounded history for statistics to control memory usage"""

    # Confidence distribution
    max_confidence_history = 1000
    if len(self.confidence_distribution) > max_confidence_history:
        # Keep most recent entries
        self.confidence_distribution = self.confidence_distribution[-max_confidence_history:]

    # Decision reasons (prevent unbounded growth)
    max_reasons = 50
    if len(self.decision_reasons) > max_reasons:
        # Keep most frequent reasons
        sorted_reasons = sorted(
            self.decision_reasons.items(),
            key=lambda x: x[1],
            reverse=True
        )
        self.decision_reasons = dict(sorted_reasons[:max_reasons])
```

## 8. Testing Specifications

### 8.1 Unit Tests

```python
class TestSelectiveGate:

    def test_retrieval_decision_logic(self):
        """Test retrieval decision logic with various confidence levels"""
        gate = SelectiveGate(self.config)

        # High confidence - should skip retrieval
        high_similarities = {"expert_a": 0.8}
        should_retrieve, k = gate.should_retrieve_and_k(high_similarities)
        assert should_retrieve == False
        assert k == self.config.moe.gate.k_min

        # Low confidence - should retrieve with max k
        low_similarities = {"expert_a": 0.3}
        should_retrieve, k = gate.should_retrieve_and_k(low_similarities)
        assert should_retrieve == True
        assert k == self.config.moe.gate.k_max

        # Medium confidence - should retrieve with default k
        medium_similarities = {"expert_a": 0.5}
        should_retrieve, k = gate.should_retrieve_and_k(medium_similarities)
        assert should_retrieve == True
        assert k == self.config.moe.gate.default_top_k

    def test_score_filtering(self):
        """Test dynamic score filtering"""
        gate = SelectiveGate(self.config)

        # Create test matches with varying scores
        matches = [
            {"id": "1", "score": 0.9, "metadata": {}},
            {"id": "2", "score": 0.7, "metadata": {}},
            {"id": "3", "score": 0.5, "metadata": {}},
            {"id": "4", "score": 0.3, "metadata": {}}
        ]

        filtered = gate.apply_score_filtering(matches, np.random.randn(384))

        # Should filter out low-scoring matches
        assert len(filtered) <= len(matches)
        assert all(match["score"] >= gate._calculate_moderate_threshold(0.6, 0.2) for match in filtered)

    def test_query_complexity_analysis(self):
        """Test query complexity scoring"""
        gate = SelectiveGate(self.config)

        # Simple query
        simple_query = "What is AI?"
        simple_score = gate._analyze_query_complexity(simple_query)
        assert simple_score < 0.5

        # Complex query
        complex_query = "How do neural networks work and what are the differences between convolutional and recurrent architectures?"
        complex_score = gate._analyze_query_complexity(complex_query)
        assert complex_score > 0.7

    def test_adaptive_thresholds(self):
        """Test threshold adaptation based on performance"""
        gate = SelectiveGate(self.config)
        original_threshold = gate.config.moe.gate.retrieve_sim_threshold

        # Good performance - should increase threshold
        good_metrics = {
            "avg_relevance": 0.9,
            "retrieval_efficiency": 0.9,
            "false_positive_rate": 0.1
        }
        gate.update_thresholds(good_metrics)
        assert gate.config.moe.gate.retrieve_sim_threshold >= original_threshold

        # Poor performance - should decrease threshold
        poor_metrics = {
            "avg_relevance": 0.5,
            "retrieval_efficiency": 0.5,
            "false_positive_rate": 0.5
        }
        gate.update_thresholds(poor_metrics)
        assert gate.config.moe.gate.retrieve_sim_threshold <= original_threshold
```

### 8.2 Integration Tests

```python
class TestSelectiveGateIntegration:

    def test_full_gate_pipeline(self):
        """Test complete gate decision pipeline"""
        gate = SelectiveGate(self.config)

        # Simulate router output
        router_similarities = {
            "general": 0.7,
            "technical": 0.4,
            "personal": 0.3
        }

        # Test decision making
        should_retrieve, k = gate.should_retrieve_and_k(router_similarities)
        assert isinstance(should_retrieve, bool)
        assert isinstance(k, int)
        assert k >= gate.config.moe.gate.k_min
        assert k <= gate.config.moe.gate.k_max

        # Test score filtering if retrieving
        if should_retrieve:
            matches = self._generate_test_matches(k + 5)  # Extra matches
            filtered = gate.apply_score_filtering(matches, np.random.randn(384))
            assert len(filtered) <= len(matches)
            assert len(filtered) >= gate.config.moe.gate.k_min

    def test_performance_requirements(self):
        """Test performance meets requirements"""
        gate = SelectiveGate(self.config)

        # Measure decision time
        start_time = time.time()
        for _ in range(1000):
            similarities = {"expert_a": random.random()}
            gate.should_retrieve_and_k(similarities)
        end_time = time.time()

        avg_time = (end_time - start_time) / 1000 * 1000  # Convert to ms
        assert avg_time < 5.0  # Less than 5ms requirement
```

### 8.3 Edge Case Tests

```python
class TestSelectiveGateEdgeCases:

    def test_empty_similarities(self):
        """Test behavior with empty similarities"""
        gate = SelectiveGate(self.config)

        should_retrieve, k = gate.should_retrieve_and_k({})
        assert should_retrieve == True  # Default to retrieval
        assert k == gate.config.moe.gate.default_top_k

    def test_extreme_scores(self):
        """Test behavior with extreme similarity scores"""
        gate = SelectiveGate(self.config)

        # Perfect similarity
        should_retrieve, k = gate.should_retrieve_and_k({"expert_a": 1.0})
        assert should_retrieve == False

        # Zero similarity
        should_retrieve, k = gate.should_retrieve_and_k({"expert_a": 0.0})
        assert should_retrieve == True
        assert k == gate.config.moe.gate.k_max

    def test_empty_matches(self):
        """Test score filtering with empty matches"""
        gate = SelectiveGate(self.config)

        filtered = gate.apply_score_filtering([], np.random.randn(384))
        assert filtered == []
```

## 9. Monitoring & Observability

### 9.1 Metrics Collection

```python
class GateMetrics:
    """Comprehensive metrics for selective gate"""

    def __init__(self):
        self.decisions = []
        self.filtering_operations = []
        self.performance_history = []
        self.threshold_history = []

    def record_decision(self, decision: GateDecision):
        """Record gate decision"""
        self.decisions.append({
            "timestamp": decision.timestamp,
            "should_retrieve": decision.should_retrieve,
            "k": decision.optimal_k,
            "confidence": decision.confidence_level,
            "reason": decision.decision_reason,
            "processing_time": decision.processing_time_ms
        })

        # Maintain bounded history
        if len(self.decisions) > 5000:
            self.decisions = self.decisions[-5000:]

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        if not self.decisions:
            return {}

        recent_decisions = self.decisions[-100:]  # Last 100 decisions

        retrieval_rate = sum(1 for d in recent_decisions if d["should_retrieve"]) / len(recent_decisions)
        avg_confidence = sum(d["confidence"] for d in recent_decisions) / len(recent_decisions)
        avg_processing_time = sum(d["processing_time"] for d in recent_decisions) / len(recent_decisions)

        return {
            "total_decisions": len(self.decisions),
            "retrieval_rate": retrieval_rate,
            "avg_confidence": avg_confidence,
            "avg_processing_time_ms": avg_processing_time,
            "most_common_reason": max(set(d["reason"] for d in recent_decisions), key=lambda x: sum(1 for d in recent_decisions if d["reason"] == x))
        }
```

### 9.2 Health Monitoring

```python
def health_check(self) -> Dict[str, Any]:
    """Gate health check"""
    summary = self.metrics.get_performance_summary()

    health_status = {
        "component": "selective_gate",
        "status": "healthy",
        "timestamp": time.time(),
        "metrics": summary
    }

    # Check for unhealthy conditions
    if summary.get("avg_processing_time_ms", 0) > 10.0:
        health_status["status"] = "degraded"
        health_status["issues"] = ["High processing latency"]

    if summary.get("retrieval_rate", 0) > 0.95:
        health_status["status"] = "warning"
        health_status["issues"] = health_status.get("issues", []) + ["Very high retrieval rate - check thresholds"]

    if summary.get("retrieval_rate", 1) < 0.05:
        health_status["status"] = "warning"
        health_status["issues"] = health_status.get("issues", []) + ["Very low retrieval rate - check thresholds"]

    return health_status
```

## 10. Configuration & Deployment

### 10.1 Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | true | Enable/disable selective gate |
| `retrieve_sim_threshold` | float | 0.62 | Skip retrieval above this confidence |
| `low_sim_threshold` | float | 0.45 | Use max k below this confidence |
| `k_min` | int | 4 | Minimum retrieval count |
| `k_max` | int | 15 | Maximum retrieval count |
| `default_top_k` | int | 8 | Standard retrieval count |
| `high_score_cutoff` | float | 0.8 | Strict filtering threshold |
| `low_score_cutoff` | float | 0.5 | Lenient filtering threshold |
| `confidence_weight` | float | 0.7 | Weight for confidence in decisions |
| `complexity_weight` | float | 0.3 | Weight for query complexity |
| `adaptation_rate` | float | 0.01 | Rate of threshold adaptation |

### 10.2 Deployment Checklist

- [ ] Gate thresholds tuned for expected query patterns
- [ ] Performance monitoring configured
- [ ] Cache size optimized for query volume
- [ ] Error handling and fallback mechanisms tested
- [ ] Adaptive learning parameters configured
- [ ] Health check endpoints enabled
- [ ] Statistics collection configured

## 11. Future Enhancements

### 11.1 Advanced Decision Making

**Context-Aware Decisions**: Incorporate query context and user history
**Multi-Criteria Optimization**: Balance multiple objectives (quality, latency, cost)
**Predictive Gating**: Use machine learning to predict optimal gate decisions
**Dynamic Thresholds**: Real-time threshold adjustment based on system load

### 11.2 Enhanced Filtering

**Semantic Filtering**: Use embeddings for semantic relevance filtering
**Diversity-Aware Filtering**: Ensure result diversity in filtering
**User Preference Integration**: Personalize filtering based on user behavior
**Multi-Modal Filtering**: Support for different content types

### 11.3 Learning Improvements

**Reinforcement Learning**: Learn optimal gate decisions through feedback
**Online Adaptation**: Continuous model updates from query patterns
**Transfer Learning**: Apply learning from one domain to others
**Ensemble Methods**: Combine multiple gate decision strategies

This detailed specification provides the foundation for implementing an intelligent Selective Gate that optimizes the balance between retrieval quality and computational efficiency in the MoE system.