# Two-Stage Reranker Detailed Specification

## Document Information
- **Document ID:** MOE-RERANKER-SPEC-001
- **Version:** 1.0.0
- **Created:** 2025-08-30
- **Last Updated:** 2025-08-30
- **Status:** Draft

## Overview

The Two-Stage Reranker is the final component in the Mixture of Experts (MoE) pipeline, responsible for enhancing retrieval quality through sophisticated cross-encoder reranking and optional LLM-based refinement. This component provides significant improvements in answer relevance and citation accuracy.

## 1. Functional Requirements

### 1.1 Core Reranking Functionality

**FR-RERANKER-001**: The Two-Stage Reranker SHALL perform cross-encoder reranking on retrieved documents.

**FR-RERANKER-002**: The Two-Stage Reranker SHALL support conditional LLM-based reranking.

**FR-RERANKER-003**: The Two-Stage Reranker SHALL calculate uncertainty metrics from reranking results.

**FR-RERANKER-004**: The Two-Stage Reranker SHALL provide configurable activation conditions.

**FR-RERANKER-005**: The Two-Stage Reranker SHALL maintain performance within latency budgets.

### 1.2 Quality Enhancement

**FR-RERANKER-006**: The Two-Stage Reranker SHALL improve retrieval quality by 10-20% over baseline.

**FR-RERANKER-007**: The Two-Stage Reranker SHALL optimize for both relevance and diversity.

**FR-RERANKER-008**: The Two-Stage Reranker SHALL handle edge cases gracefully.

### 1.3 Performance Optimization

**FR-RERANKER-009**: Stage 1 reranking SHALL complete within 50-200ms for typical result sets.

**FR-RERANKER-010**: The Two-Stage Reranker SHALL minimize computational overhead.

**FR-RERANKER-011**: The Two-Stage Reranker SHALL support batch processing for efficiency.

## 2. Interface Specifications

### 2.1 Class Definition

```python
class TwoStageReranker:
    """Production two-stage reranking pipeline"""

    def __init__(self, config: AppConfig):
        """Initialize reranker with configuration"""
        pass

    def rerank_stage1(
        self,
        query: str,
        matches: List[Dict]
    ) -> Tuple[List[Dict], float]:
        """Stage 1: Cross-encoder reranking (always applied)"""
        pass

    def rerank_stage2_llm(
        self,
        query: str,
        matches: List[Dict],
        uncertainty: float
    ) -> List[Dict]:
        """Stage 2: LLM-based reranking (conditional)"""
        pass

    def rerank(
        self,
        query: str,
        matches: List[Dict]
    ) -> List[Dict]:
        """Complete reranking pipeline"""
        pass

    def get_reranking_statistics(self) -> Dict[str, Any]:
        """Get reranking performance statistics"""
        pass
```

### 2.2 Data Structures

```python
@dataclass
class RerankingResult:
    """Result of reranking operation"""
    original_matches: List[Dict]
    reranked_matches: List[Dict]
    stage1_uncertainty: float
    stage2_applied: bool
    processing_time_ms: float
    improvement_score: float
    timestamp: float

@dataclass
class CrossEncoderResult:
    """Result of cross-encoder reranking"""
    query_passage_pairs: List[Tuple[str, str]]
    scores: List[float]
    uncertainty: float
    processing_time_ms: float
    model_name: str

@dataclass
class LLMRerankingResult:
    """Result of LLM-based reranking"""
    ranking: List[int]
    reasoning: str
    confidence: float
    processing_time_ms: float
    tokens_used: int
```

### 2.3 Configuration Schema

```yaml
moe:
  reranker:
    enabled: true

    # Stage 1: Cross-encoder configuration
    stage1_enabled: true
    cross_encoder_model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
    max_rerank_candidates: 50
    batch_size: 16

    # Stage 2: LLM reranking configuration
    stage2_enabled: false  # Expensive, disabled by default
    uncertainty_threshold: 0.15
    llm_temperature: 0.0
    max_llm_candidates: 10
    llm_prompt_template: "complex_reranking"

    # Performance optimization
    cache_enabled: true
    cache_size: 1000
    early_termination_threshold: 0.8

    # Quality thresholds
    min_improvement_threshold: 0.05
    max_diversity_penalty: 0.1
```

## 3. Stage 1: Cross-Encoder Reranking

### 3.1 Algorithm Overview

**Algorithm: Cross-Encoder Relevance Scoring**

```python
def rerank_stage1(
    self,
    query: str,
    matches: List[Dict]
) -> Tuple[List[Dict], float]:
    """
    Perform cross-encoder reranking with uncertainty calculation.

    Process:
    1. Prepare query-passage pairs from matches
    2. Score pairs using cross-encoder model
    3. Combine with original similarity scores
    4. Sort by combined relevance
    5. Calculate uncertainty from score distribution

    Args:
        query: User query string
        matches: Retrieved matches with metadata

    Returns:
        Tuple of (reranked_matches, uncertainty_score)
    """

    if not matches or not self.config.moe.reranker.stage1_enabled:
        return matches, 0.0

    # Prepare passages from matches
    passages = []
    for match in matches[:self.config.moe.reranker.max_rerank_candidates]:
        text = match.get('metadata', {}).get('text', '')
        if text:
            passages.append(text)
        else:
            passages.append('')  # Fallback for missing text

    if not passages:
        return matches, 0.0

    # Score query-passage pairs
    pairs = [(query, passage) for passage in passages]
    scores = self._batch_score_pairs(pairs)

    # Enhance matches with cross-encoder scores
    enhanced_matches = []
    for match, ce_score in zip(matches, scores):
        enhanced_match = match.copy()
        enhanced_match['cross_encoder_score'] = float(ce_score)
        enhanced_match['original_score'] = match.get('score', 0.0)

        # Combined scoring (weighted combination)
        combined_score = self._combine_scores(
            ce_score,
            match.get('score', 0.0)
        )
        enhanced_match['combined_score'] = combined_score

        enhanced_matches.append(enhanced_match)

    # Sort by combined score (descending)
    reranked_matches = sorted(
        enhanced_matches,
        key=lambda x: x['combined_score'],
        reverse=True
    )

    # Calculate uncertainty
    uncertainty = self._calculate_uncertainty(scores)

    # Record statistics
    self._record_stage1_stats(
        len(matches),
        len(reranked_matches),
        uncertainty,
        scores
    )

    logger.debug(
        f"Stage1 reranking completed: {len(matches)} matches, "
        f"uncertainty: {uncertainty:.3f}"
    )

    return reranked_matches, uncertainty
```

**Algorithm: Batch Scoring Optimization**

```python
def _batch_score_pairs(self, pairs: List[Tuple[str, str]]) -> List[float]:
    """
    Score multiple query-passage pairs in optimized batches.

    Optimization Techniques:
    1. Batch processing for GPU efficiency
    2. Memory-efficient tensor operations
    3. Early termination for high-confidence results
    4. Caching for repeated queries

    Args:
        pairs: List of (query, passage) tuples

    Returns:
        List of relevance scores
    """

    if not pairs:
        return []

    # Check cache first
    cache_key = self._get_cache_key(pairs)
    if self.cache_enabled and cache_key in self.score_cache:
        return self.score_cache[cache_key]

    # Batch processing
    batch_size = self.config.moe.reranker.batch_size
    all_scores = []

    for i in range(0, len(pairs), batch_size):
        batch_pairs = pairs[i:i + batch_size]

        try:
            batch_scores = self.cross_encoder.predict(batch_pairs)
            all_scores.extend(batch_scores)

            # Early termination check
            if self._should_early_terminate(batch_scores):
                logger.debug(f"Early termination at batch {i // batch_size}")
                break

        except Exception as e:
            logger.error(f"Batch scoring failed at batch {i // batch_size}: {e}")
            # Fallback: assign neutral scores
            all_scores.extend([0.5] * len(batch_pairs))

    # Cache results
    if self.cache_enabled:
        self._update_cache(cache_key, all_scores)

    return all_scores
```

### 3.2 Score Combination Strategy

**Algorithm: Adaptive Score Combination**

```python
def _combine_scores(
    self,
    cross_encoder_score: float,
    original_score: float
) -> float:
    """
    Combine cross-encoder and original similarity scores.

    Combination Strategy:
    1. Normalize scores to comparable ranges
    2. Apply adaptive weighting based on score characteristics
    3. Handle edge cases and missing values

    Args:
        cross_encoder_score: Cross-encoder relevance score
        original_score: Original vector similarity score

    Returns:
        Combined relevance score
    """

    # Normalize cross-encoder score (typically -1 to 1) to 0-1 range
    normalized_ce = (cross_encoder_score + 1.0) / 2.0

    # Original score is typically already 0-1, but clamp for safety
    normalized_orig = max(0.0, min(1.0, original_score))

    # Adaptive weighting based on score confidence
    ce_confidence = abs(cross_encoder_score)  # Higher magnitude = more confident
    orig_confidence = original_score  # Higher similarity = more confident

    # Weight calculation: favor cross-encoder for high-confidence predictions
    if ce_confidence > 0.7:
        ce_weight = 0.8
        orig_weight = 0.2
    elif ce_confidence > 0.5:
        ce_weight = 0.6
        orig_weight = 0.4
    else:
        ce_weight = 0.4
        orig_weight = 0.6

    # Weighted combination
    combined_score = (
        normalized_ce * ce_weight +
        normalized_orig * orig_weight
    )

    # Diversity bonus (slight preference for original ranking to maintain diversity)
    diversity_factor = 0.02 * (1.0 - abs(normalized_ce - normalized_orig))
    combined_score += diversity_factor

    return min(1.0, combined_score)
```

### 3.3 Uncertainty Calculation

**Algorithm: Score Distribution Uncertainty**

```python
def _calculate_uncertainty(self, scores: List[float]) -> float:
    """
    Calculate uncertainty from score distribution.

    Uncertainty Metrics:
    1. Score variance (higher variance = more uncertainty)
    2. Score range (wider range = more uncertainty)
    3. Confidence intervals
    4. Distribution entropy

    Args:
        scores: List of cross-encoder scores

    Returns:
        Uncertainty score (0.0-1.0, higher = more uncertain)
    """

    if len(scores) < 2:
        return 0.0  # Cannot calculate uncertainty with single score

    scores_array = np.array(scores)

    # Variance-based uncertainty
    variance = np.var(scores_array)
    variance_uncertainty = min(variance * 4.0, 1.0)  # Scale and clamp

    # Range-based uncertainty
    score_range = np.max(scores_array) - np.min(scores_array)
    range_uncertainty = min(score_range * 2.0, 1.0)  # Scale and clamp

    # Entropy-based uncertainty
    # Discretize scores into bins
    hist, _ = np.histogram(scores_array, bins=10, range=(-1, 1))
    hist = hist / len(scores_array)  # Normalize to probabilities

    # Calculate entropy
    entropy = 0.0
    for p in hist:
        if p > 0:
            entropy -= p * np.log2(p)

    max_entropy = np.log2(10)  # Maximum entropy for 10 bins
    entropy_uncertainty = entropy / max_entropy

    # Combine uncertainty measures
    combined_uncertainty = (
        variance_uncertainty * 0.4 +
        range_uncertainty * 0.3 +
        entropy_uncertainty * 0.3
    )

    return min(combined_uncertainty, 1.0)
```

## 4. Stage 2: LLM-Based Reranking

### 4.1 Activation Conditions

**Algorithm: Conditional LLM Activation**

```python
def _should_activate_stage2(self, uncertainty: float, matches: List[Dict]) -> bool:
    """
    Determine if Stage 2 LLM reranking should be activated.

    Activation Criteria:
    1. Uncertainty above threshold
    2. Sufficient number of candidates
    3. Feature enabled in configuration
    4. Cost-benefit analysis (optional)

    Args:
        uncertainty: Stage 1 uncertainty score
        matches: Current matches after Stage 1

    Returns:
        True if Stage 2 should be activated
    """

    # Check basic conditions
    if not self.config.moe.reranker.stage2_enabled:
        return False

    if uncertainty < self.config.moe.reranker.uncertainty_threshold:
        logger.debug(f"Stage2 skipped: uncertainty {uncertainty:.3f} < threshold {self.config.moe.reranker.uncertainty_threshold}")
        return False

    if len(matches) < 3:
        logger.debug(f"Stage2 skipped: insufficient candidates ({len(matches)} < 3)")
        return False

    # Cost-benefit check (optional)
    if self._should_skip_for_cost(matches):
        logger.debug("Stage2 skipped: cost-benefit analysis")
        return False

    return True
```

### 4.2 LLM Reranking Process

**Algorithm: LLM-Based Relevance Assessment**

```python
def rerank_stage2_llm(
    self,
    query: str,
    matches: List[Dict],
    uncertainty: float
) -> List[Dict]:
    """
    Perform LLM-based reranking for uncertain cases.

    Process:
    1. Prepare passages for LLM evaluation
    2. Generate reranking prompt
    3. Call LLM for ranking assessment
    4. Parse and apply LLM ranking
    5. Fallback on failure

    Args:
        query: User query string
        matches: Matches after Stage 1 reranking
        uncertainty: Stage 1 uncertainty score

    Returns:
        LLM-reranked matches
    """

    if not self._should_activate_stage2(uncertainty, matches):
        return matches

    try:
        # Prepare passages for LLM
        passages = self._prepare_passages_for_llm(matches)

        # Generate reranking prompt
        prompt = self._generate_reranking_prompt(query, passages)

        # Call LLM
        llm_response = self._call_llm_for_reranking(prompt)

        # Parse ranking from response
        ranking = self._parse_llm_ranking(llm_response, len(passages))

        # Apply ranking
        reranked_matches = [matches[i] for i in ranking if i < len(matches)]

        # Ensure all original matches are included
        included_indices = set(ranking)
        for i, match in enumerate(matches):
            if i not in included_indices:
                reranked_matches.append(match)

        logger.debug(f"Stage2 LLM reranking completed: {len(reranked_matches)} matches")

        return reranked_matches

    except Exception as e:
        logger.error(f"Stage2 LLM reranking failed: {e}")
        return matches  # Fallback to Stage 1 results
```

### 4.3 Prompt Engineering

**Algorithm: Dynamic Prompt Generation**

```python
def _generate_reranking_prompt(
    self,
    query: str,
    passages: List[str]
) -> str:
    """
    Generate optimized prompt for LLM reranking.

    Prompt Optimization:
    1. Clear ranking instructions
    2. Relevance criteria specification
    3. Format requirements
    4. Example-based guidance

    Args:
        query: User query
        passages: List of passages to rank

    Returns:
        Formatted prompt for LLM
    """

    # Truncate passages for context limits
    max_passage_length = 200
    truncated_passages = []
    for i, passage in enumerate(passages):
        truncated = passage[:max_passage_length]
        if len(passage) > max_passage_length:
            truncated += "..."
        truncated_passages.append(f"[{i}] {truncated}")

    passages_text = "\n".join(truncated_passages)

    prompt = f"""Rank the following passages by their relevance to the query.

Query: {query}

Passages:
{passages_text}

Instructions:
- Rank passages from most relevant (0) to least relevant
- Consider both direct relevance and contextual usefulness
- Return ranking as comma-separated indices only
- Example format: 0,2,1,4,3

Ranking:"""

    return prompt
```

## 5. Performance Optimization

### 5.1 Caching Strategy

**Technique: Multi-Level Caching**

```python
class RerankerCache:
    """Multi-level caching for reranking operations"""

    def __init__(self, max_size: int = 1000):
        self.score_cache = {}  # Cross-encoder scores
        self.llm_cache = {}    # LLM reranking results
        self.uncertainty_cache = {}  # Uncertainty calculations
        self.max_size = max_size

    def get_score_cache_key(self, pairs: List[Tuple[str, str]]) -> str:
        """Generate cache key for score caching"""
        # Use hash of concatenated pairs
        combined = "".join([f"{q}:{p}" for q, p in pairs])
        return hashlib.md5(combined.encode()).hexdigest()

    def get_llm_cache_key(self, query: str, passages_hash: str) -> str:
        """Generate cache key for LLM caching"""
        return f"{hashlib.md5(query.encode()).hexdigest()}:{passages_hash}"

    def cleanup_cache(self):
        """Maintain cache size limits"""
        for cache_name in ['score_cache', 'llm_cache', 'uncertainty_cache']:
            cache = getattr(self, cache_name)
            if len(cache) > self.max_size:
                # Remove oldest entries (simple FIFO)
                items_to_remove = len(cache) - self.max_size
                for _ in range(items_to_remove):
                    cache.pop(next(iter(cache)))
```

### 5.2 Batch Processing Optimization

**Technique: Adaptive Batch Sizing**

```python
def _optimize_batch_size(self, num_pairs: int, available_memory: float) -> int:
    """
    Dynamically optimize batch size based on constraints.

    Optimization Factors:
    1. Available GPU memory
    2. Number of pairs to process
    3. Model memory requirements
    4. Latency requirements

    Args:
        num_pairs: Number of query-passage pairs
        available_memory: Available GPU memory (GB)

    Returns:
        Optimal batch size
    """

    # Base batch size from configuration
    base_batch_size = self.config.moe.reranker.batch_size

    # Memory-based adjustment
    memory_factor = available_memory / 8.0  # Assume 8GB baseline
    memory_adjusted = int(base_batch_size * memory_factor)

    # Pair count adjustment
    if num_pairs < 10:
        pair_adjusted = min(num_pairs, base_batch_size)
    elif num_pairs > 100:
        pair_adjusted = max(base_batch_size * 2, memory_adjusted)
    else:
        pair_adjusted = base_batch_size

    # Final optimization
    optimal_batch_size = max(1, min(pair_adjusted, memory_adjusted, 64))

    logger.debug(f"Optimized batch size: {optimal_batch_size} "
                f"(memory: {available_memory:.1f}GB, pairs: {num_pairs})")

    return optimal_batch_size
```

### 5.3 Early Termination

**Technique: Confidence-Based Early Termination**

```python
def _should_early_terminate(self, batch_scores: List[float]) -> bool:
    """
    Determine if reranking can terminate early.

    Early Termination Criteria:
    1. High confidence in top results
    2. Clear separation between scores
    3. Sufficient high-quality results found

    Args:
        batch_scores: Scores from current batch

    Returns:
        True if processing can terminate early
    """

    if len(batch_scores) < 3:
        return False

    scores_array = np.array(batch_scores)

    # Check for clear winner (high confidence in top result)
    top_score = np.max(scores_array)
    if top_score > self.config.moe.reranker.early_termination_threshold:
        mean_other_scores = np.mean(scores_array[scores_array < top_score])
        if top_score - mean_other_scores > 0.3:  # Clear separation
            return True

    # Check for score distribution stability
    score_std = np.std(scores_array)
    if score_std < 0.1:  # Very similar scores
        return True

    return False
```

## 6. Error Handling & Resilience

### 6.1 Error Scenarios

| Error Scenario | Handling Strategy | Fallback Behavior |
|----------------|-------------------|-------------------|
| Cross-encoder model failure | Log error, disable Stage 1 | Return original matches |
| LLM API failure/timeout | Retry with backoff | Fall back to Stage 1 results |
| Memory allocation error | Reduce batch size | Process in smaller batches |
| Invalid input format | Input validation | Skip problematic inputs |
| Cache corruption | Clear and rebuild cache | Disable caching temporarily |

### 6.2 Validation Checks

```python
def _validate_reranking_inputs(
    self,
    query: str,
    matches: List[Dict]
) -> bool:
    """Validate inputs for reranking operations"""

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be non-empty string")

    if not isinstance(matches, list):
        raise ValueError("Matches must be a list")

    for i, match in enumerate(matches):
        if not isinstance(match, dict):
            raise ValueError(f"Match at index {i} must be a dictionary")

        if 'metadata' not in match:
            logger.warning(f"Match {i} missing metadata, adding empty dict")
            match['metadata'] = {}

    return True
```

## 7. Testing Specifications

### 7.1 Unit Tests

```python
class TestTwoStageReranker:

    def test_cross_encoder_scoring(self):
        """Test cross-encoder scoring functionality"""
        reranker = TwoStageReranker(self.config)

        query = "What is machine learning?"
        passages = [
            "Machine learning is a subset of AI",
            "Python is a programming language",
            "ML algorithms learn from data"
        ]

        scores = reranker._batch_score_pairs([(query, p) for p in passages])

        assert len(scores) == 3
        assert all(isinstance(score, float) for score in scores)
        assert all(-1 <= score <= 1 for score in scores)

    def test_uncertainty_calculation(self):
        """Test uncertainty calculation from score distributions"""
        reranker = TwoStageReranker(self.config)

        # High certainty case (clear separation)
        certain_scores = [0.9, 0.1, 0.05, -0.2]
        uncertainty = reranker._calculate_uncertainty(certain_scores)
        assert uncertainty < 0.3

        # High uncertainty case (similar scores)
        uncertain_scores = [0.6, 0.55, 0.58, 0.52]
        uncertainty = reranker._calculate_uncertainty(uncertain_scores)
        assert uncertainty > 0.7

    def test_score_combination(self):
        """Test adaptive score combination"""
        reranker = TwoStageReranker(self.config)

        # High confidence cross-encoder
        combined = reranker._combine_scores(0.8, 0.6)
        assert combined > 0.7  # Should favor cross-encoder

        # Low confidence cross-encoder
        combined = reranker._combine_scores(0.2, 0.6)
        assert combined < 0.5  # Should favor original score

    def test_stage2_activation_logic(self):
        """Test Stage 2 activation conditions"""
        reranker = TwoStageReranker(self.config)

        # Should activate: high uncertainty, sufficient candidates
        matches = [{"text": "test"}] * 5
        should_activate = reranker._should_activate_stage2(0.8, matches)
        assert should_activate == True

        # Should not activate: low uncertainty
        should_activate = reranker._should_activate_stage2(0.05, matches)
        assert should_activate == False

        # Should not activate: insufficient candidates
        should_activate = reranker._should_activate_stage2(0.8, [{"text": "test"}])
        assert should_activate == False
```

### 7.2 Integration Tests

```python
class TestTwoStageRerankerIntegration:

    def test_full_reranking_pipeline(self):
        """Test complete two-stage reranking pipeline"""
        reranker = TwoStageReranker(self.config)

        query = "What are the benefits of renewable energy?"
        matches = self._generate_test_matches(10)

        # Test Stage 1 only
        reranked, uncertainty = reranker.rerank_stage1(query, matches)
        assert len(reranked) == len(matches)
        assert isinstance(uncertainty, float)
        assert 0 <= uncertainty <= 1

        # Test full pipeline
        final_matches = reranker.rerank(query, matches)
        assert len(final_matches) == len(matches)

        # Verify scoring
        for match in final_matches:
            assert 'cross_encoder_score' in match
            assert 'combined_score' in match

    def test_performance_requirements(self):
        """Test performance meets requirements"""
        reranker = TwoStageReranker(self.config)

        query = "Test query"
        matches = self._generate_test_matches(20)

        # Measure Stage 1 performance
        start_time = time.time()
        for _ in range(10):
            reranked, _ = reranker.rerank_stage1(query, matches)
        end_time = time.time()

        avg_time = (end_time - start_time) / 10 * 1000  # ms
        assert avg_time < 200  # Less than 200ms requirement
```

### 7.3 Edge Case Tests

```python
class TestTwoStageRerankerEdgeCases:

    def test_empty_matches(self):
        """Test reranking with empty matches"""
        reranker = TwoStageReranker(self.config)

        reranked, uncertainty = reranker.rerank_stage1("test query", [])
        assert reranked == []
        assert uncertainty == 0.0

    def test_single_match(self):
        """Test reranking with single match"""
        reranker = TwoStageReranker(self.config)

        matches = [{"metadata": {"text": "single passage"}}]
        reranked, uncertainty = reranker.rerank_stage1("test query", matches)

        assert len(reranked) == 1
        assert uncertainty == 0.0  # Cannot calculate uncertainty

    def test_missing_text(self):
        """Test handling of matches without text"""
        reranker = TwoStageReranker(self.config)

        matches = [{"metadata": {}}, {"metadata": {"text": "valid text"}}]
        reranked, uncertainty = reranker.rerank_stage1("test query", matches)

        assert len(reranked) == 2
        # Should handle missing text gracefully
```

## 8. Monitoring & Observability

### 8.1 Metrics Collection

```python
class RerankerMetrics:
    """Comprehensive metrics for reranking operations"""

    def __init__(self):
        self.stage1_operations = []
        self.stage2_operations = []
        self.cache_performance = {"hits": 0, "misses": 0}
        self.error_counts = defaultdict(int)

    def record_stage1_operation(
        self,
        input_count: int,
        output_count: int,
        uncertainty: float,
        processing_time: float,
        scores: List[float]
    ):
        """Record Stage 1 operation metrics"""
        self.stage1_operations.append({
            "timestamp": time.time(),
            "input_count": input_count,
            "output_count": output_count,
            "uncertainty": uncertainty,
            "processing_time": processing_time,
            "avg_score": np.mean(scores) if scores else 0.0,
            "score_std": np.std(scores) if scores else 0.0
        })

        # Maintain bounded history
        if len(self.stage1_operations) > 1000:
            self.stage1_operations = self.stage1_operations[-1000:]

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        if not self.stage1_operations:
            return {}

        recent_ops = self.stage1_operations[-100:]

        return {
            "total_operations": len(self.stage1_operations),
            "avg_processing_time_ms": np.mean([op["processing_time"] for op in recent_ops]),
            "avg_uncertainty": np.mean([op["uncertainty"] for op in recent_ops]),
            "cache_hit_rate": self._calculate_cache_hit_rate(),
            "stage2_activation_rate": len(self.stage2_operations) / max(len(self.stage1_operations), 1),
            "error_rate": sum(self.error_counts.values()) / max(len(self.stage1_operations), 1)
        }

    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache performance"""
        total = self.cache_performance["hits"] + self.cache_performance["misses"]
        return self.cache_performance["hits"] / total if total > 0 else 0.0
```

### 8.2 Health Monitoring

```python
def health_check(self) -> Dict[str, Any]:
    """Reranker health check"""
    summary = self.metrics.get_performance_summary()

    health_status = {
        "component": "two_stage_reranker",
        "status": "healthy",
        "timestamp": time.time(),
        "metrics": summary
    }

    # Check for unhealthy conditions
    if summary.get("avg_processing_time_ms", 0) > 300:
        health_status["status"] = "degraded"
        health_status["issues"] = ["High processing latency"]

    if summary.get("error_rate", 0) > 0.1:
        health_status["status"] = "degraded"
        health_status["issues"] = health_status.get("issues", []) + ["High error rate"]

    if summary.get("cache_hit_rate", 0) < 0.5:
        health_status["status"] = "warning"
        health_status["issues"] = health_status.get("issues", []) + ["Low cache hit rate"]

    return health_status
```

## 9. Configuration & Deployment

### 9.1 Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | true | Enable/disable reranking |
| `stage1_enabled` | bool | true | Enable Stage 1 cross-encoder |
| `cross_encoder_model` | str | "cross-encoder/ms-marco-MiniLM-L-6-v2" | Cross-encoder model name |
| `max_rerank_candidates` | int | 50 | Maximum candidates for reranking |
| `batch_size` | int | 16 | Batch size for cross-encoder |
| `stage2_enabled` | bool | false | Enable Stage 2 LLM reranking |
| `uncertainty_threshold` | float | 0.15 | Threshold for Stage 2 activation |
| `cache_enabled` | bool | true | Enable result caching |
| `cache_size` | int | 1000 | Maximum cache size |
| `early_termination_threshold` | float | 0.8 | Threshold for early termination |

### 9.2 Deployment Checklist

- [ ] Cross-encoder model downloaded and cached
- [ ] Sufficient GPU/CPU memory for batch processing
- [ ] LLM API configured for Stage 2 (if enabled)
- [ ] Cache directory permissions configured
- [ ] Performance monitoring enabled
- [ ] Error handling and fallback mechanisms tested
- [ ] Model warm-up procedures implemented

## 10. Future Enhancements

### 10.1 Advanced Reranking Techniques

**Ensemble Reranking**: Combine multiple cross-encoder models
**Query-Specific Models**: Dynamic model selection based on query type
**Interactive Reranking**: User feedback integration for continuous improvement
**Multi-Task Reranking**: Joint optimization for relevance and diversity

### 10.2 Performance Optimizations

**Model Quantization**: INT8 quantization for faster inference
**Distillation**: Smaller models distilled from larger cross-encoders
**GPU Optimization**: Advanced GPU memory management and kernel optimization
**Streaming Reranking**: Process results as they arrive

### 10.3 Quality Improvements

**Context-Aware Reranking**: Incorporate conversation history and user context
**Personalized Reranking**: User preference learning and adaptation
**Temporal Reranking**: Recency and freshness considerations
**Multi-Modal Reranking**: Support for images, tables, and structured data

This detailed specification provides the foundation for implementing a sophisticated Two-Stage Reranker that significantly enhances retrieval quality while maintaining excellent performance characteristics.