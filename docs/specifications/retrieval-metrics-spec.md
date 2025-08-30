# Retrieval Metrics Specification

**Document ID:** RETRIEVAL-METRICS-001
**Version:** 1.0.0
**Date:** 2025-08-30
**Authors:** SPARC Specification Writer

## 1. Overview

This specification defines comprehensive retrieval quality metrics for evaluating the Personal RAG Chatbot's document retrieval system. It provides mathematical definitions, implementation details, and validation procedures for metrics including Hit@k, NDCG@k, MRR, MAP, and MoE-specific metrics.

## 2. Core Concepts

### 2.1 Retrieval Evaluation Framework

Retrieval evaluation in information retrieval systems involves comparing retrieved documents against ground truth relevant documents. The evaluation framework uses:

- **Query (q)**: User input query
- **Retrieved Documents (R)**: List of documents returned by the system, ordered by relevance score
- **Relevant Documents (G)**: Ground truth set of documents that are relevant to the query
- **Ranking Position (k)**: Position in the retrieved list (1-based indexing)

### 2.2 Metric Categories

#### Binary Relevance Metrics
Metrics that treat relevance as a binary decision (relevant/not relevant).

#### Graded Relevance Metrics
Metrics that consider different levels of relevance (highly relevant, somewhat relevant, etc.).

#### Ranking Quality Metrics
Metrics that evaluate the quality of the ranking itself, not just binary relevance.

## 3. Mathematical Definitions

### 3.1 Hit@k (Hit Rate at k)

**Definition**: The fraction of queries where at least one relevant document appears in the top-k retrieved results.

**Formula**:
```
Hit@k = 1 if |R_k ∩ G| > 0 else 0
```

Where:
- `R_k`: Top-k retrieved documents
- `G`: Ground truth relevant documents
- `∩`: Set intersection

**Properties**:
- Range: [0, 1]
- Higher values indicate better retrieval quality
- Binary metric (hit or miss)
- Useful for evaluating recall-oriented systems

**Example**:
```
Query: "machine learning algorithms"
Ground truth: ["doc1", "doc3", "doc5"]
Retrieved@3: ["doc2", "doc1", "doc4"]

Hit@3 = 1 (because "doc1" ∈ retrieved ∩ ground_truth)
```

### 3.2 NDCG@k (Normalized Discounted Cumulative Gain at k)

**Definition**: Normalized version of DCG that measures ranking quality by assigning higher importance to relevant documents at higher positions.

**DCG Formula**:
```
DCG@k = ∑_{i=1 to k} rel_i / log₂(i + 1)
```

**NDCG Formula**:
```
NDCG@k = DCG@k / IDCG@k
```

Where:
- `rel_i`: Relevance score of document at position i (typically 0 or 1 for binary, or graded scores)
- `IDCG@k`: Ideal DCG (DCG computed on optimal ranking of relevant documents)

**Properties**:
- Range: [0, 1]
- 1.0 indicates perfect ranking
- Penalizes placing relevant documents at lower positions
- Commonly used in learning-to-rank evaluation

**Example**:
```
Ground truth relevance: [3, 2, 1, 0] (4 relevant docs)
Retrieved ranking: [3, 0, 2, 1] (positions 1,3,2,4)

DCG@4 = 3/log₂(2) + 0/log₂(4) + 2/log₂(3) + 1/log₂(5)
       = 3/1 + 0/2 + 2/1.585 + 1/2.322
       = 3 + 0 + 1.262 + 0.431 = 4.693

IDCG@4 = 3/log₂(2) + 2/log₂(3) + 1/log₂(4) + 0/log₂(5)
        = 3/1 + 2/1.585 + 1/2 + 0/2.322
        = 3 + 1.262 + 0.5 + 0 = 4.762

NDCG@4 = 4.693 / 4.762 ≈ 0.985
```

### 3.3 MRR (Mean Reciprocal Rank)

**Definition**: Average of reciprocal ranks of the first relevant document for each query.

**Formula**:
```
MRR = (1/|Q|) * ∑_{q∈Q} (1 / rank_q)
```

Where:
- `Q`: Set of queries
- `rank_q`: Position of first relevant document for query q (∞ if no relevant document retrieved)

**Properties**:
- Range: [0, 1]
- Higher values indicate better ranking quality
- Particularly useful for question-answering systems
- Sensitive to the position of the first relevant result

**Example**:
```
Query 1: First relevant at position 1 → RR = 1/1 = 1.0
Query 2: First relevant at position 3 → RR = 1/3 ≈ 0.333
Query 3: No relevant documents → RR = 0

MRR = (1.0 + 0.333 + 0) / 3 ≈ 0.444
```

### 3.4 MAP@k (Mean Average Precision at k)

**Definition**: Mean of average precision scores for each query, considering only the top-k results.

**Average Precision Formula**:
```
AP@k = (1/|G|) * ∑_{i=1 to k} (Precision@i * rel_i)
```

**MAP Formula**:
```
MAP@k = (1/|Q|) * ∑_{q∈Q} AP@k_q
```

Where:
- `Precision@i`: Precision at position i (relevant docs retrieved so far / total retrieved so far)
- `rel_i`: Binary relevance of document at position i

**Properties**:
- Range: [0, 1]
- Considers both precision and recall aspects
- Penalizes false positives at higher positions
- Useful for evaluating ranked retrieval systems

**Example**:
```
Query: "neural networks"
Ground truth: 3 relevant docs total
Retrieved@5: [rel=1, rel=0, rel=1, rel=0, rel=1]

Precision@1: 1/1 = 1.0
Precision@2: 1/2 = 0.5
Precision@3: 2/3 ≈ 0.667
Precision@4: 2/4 = 0.5
Precision@5: 3/5 = 0.6

AP@5 = (1/3) * (1.0 + 0.667 + 0.6) = (1/3) * 2.267 ≈ 0.756
```

### 3.5 Recall@k

**Definition**: Fraction of relevant documents retrieved in the top-k results.

**Formula**:
```
Recall@k = |R_k ∩ G| / |G|
```

**Properties**:
- Range: [0, 1]
- Measures coverage of relevant documents
- Useful for evaluating completeness of retrieval
- Complementary to Precision@k

### 3.6 Precision@k

**Definition**: Fraction of retrieved documents in top-k that are relevant.

**Formula**:
```
Precision@k = |R_k ∩ G| / |R_k|
```

**Properties**:
- Range: [0, 1]
- Measures accuracy of retrieved documents
- Useful for evaluating quality of retrieval
- Complementary to Recall@k

## 4. MoE-Specific Metrics

### 4.1 Routing Accuracy

**Definition**: Accuracy of expert routing decisions based on semantic similarity.

**Formula**:
```
Routing_Accuracy = (1/|Q|) * ∑_{q∈Q} I(route_q = optimal_expert_q)
```

Where:
- `route_q`: Expert selected for query q
- `optimal_expert_q`: Ground truth optimal expert for query q
- `I()`: Indicator function (1 if true, 0 otherwise)

### 4.2 Gate Efficiency

**Definition**: Efficiency of selective retrieval gating decisions.

**Formula**:
```
Gate_Efficiency = (Retrievals_Saved / Total_Queries) * 100%
```

Where:
- `Retrievals_Saved`: Number of queries where retrieval was correctly skipped
- `Total_Queries`: Total number of queries evaluated

### 4.3 Reranking Improvement

**Definition**: Improvement in ranking quality after reranking.

**Formula**:
```
Reranking_Improvement = ((NDCG_reranked - NDCG_baseline) / NDCG_baseline) * 100%
```

Where:
- `NDCG_reranked`: NDCG score after reranking
- `NDCG_baseline`: NDCG score before reranking

## 5. Implementation Specifications

### 5.1 Core Functions

#### Hit@k Implementation
```python
def hit_at_k(relevant_ids: Union[str, List[str]],
             predicted_ids: Union[str, List[str]],
             k: int = 10) -> float:
    """
    Calculate Hit@k metric.

    Args:
        relevant_ids: Ground truth relevant document IDs
        predicted_ids: Retrieved document IDs (ordered by relevance)
        k: Cutoff position

    Returns:
        Hit@k score (0.0 or 1.0)
    """
    rel_set = _to_set(relevant_ids)
    pred_list = _to_list(predicted_ids)[:k]  # Take top-k

    return 1.0 if any(pid in rel_set for pid in pred_list) else 0.0
```

#### NDCG@k Implementation
```python
def ndcg_at_k(relevant_ids: Union[str, List[str]],
              predicted_ids: Union[str, List[str]],
              k: int = 10,
              relevance_grades: Optional[Dict[str, float]] = None) -> float:
    """
    Calculate NDCG@k metric.

    Args:
        relevant_ids: Ground truth relevant document IDs
        predicted_ids: Retrieved document IDs (ordered by relevance)
        k: Cutoff position
        relevance_grades: Optional relevance grades for graded relevance

    Returns:
        NDCG@k score
    """
    rel_set = _to_set(relevant_ids)
    pred_list = _to_list(predicted_ids)[:k]

    # Calculate DCG
    dcg = 0.0
    for i, doc_id in enumerate(pred_list):
        relevance = _get_relevance(doc_id, rel_set, relevance_grades)
        dcg += relevance / math.log2(i + 2)

    # Calculate IDCG (Ideal DCG)
    relevant_docs = list(rel_set)
    if relevance_grades:
        # Sort by relevance grade descending
        relevant_docs.sort(key=lambda x: relevance_grades.get(x, 0), reverse=True)

    idcg = 0.0
    for i, doc_id in enumerate(relevant_docs[:k]):
        relevance = _get_relevance(doc_id, rel_set, relevance_grades)
        idcg += relevance / math.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0.0
```

#### MRR Implementation
```python
def mean_reciprocal_rank(relevant_ids_list: List[Union[str, List[str]]],
                        predicted_ids_list: List[Union[str, List[str]]]) -> float:
    """
    Calculate Mean Reciprocal Rank.

    Args:
        relevant_ids_list: List of ground truth relevant document IDs per query
        predicted_ids_list: List of retrieved document IDs per query

    Returns:
        MRR score
    """
    reciprocal_ranks = []

    for rel_ids, pred_ids in zip(relevant_ids_list, predicted_ids_list):
        rel_set = _to_set(rel_ids)
        pred_list = _to_list(pred_ids)

        # Find rank of first relevant document
        rank = None
        for i, doc_id in enumerate(pred_list):
            if doc_id in rel_set:
                rank = i + 1  # 1-based indexing
                break

        # Reciprocal rank (0 if no relevant document found)
        rr = 1.0 / rank if rank else 0.0
        reciprocal_ranks.append(rr)

    return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
```

### 5.2 Batch Processing

#### Multi-Query Evaluation
```python
def evaluate_retrieval_batch(queries: List[Dict[str, Any]],
                           retrieval_results: List[List[Dict[str, Any]]],
                           metrics: List[str] = None) -> Dict[str, float]:
    """
    Evaluate retrieval quality for a batch of queries.

    Args:
        queries: List of query objects with ground truth
        retrieval_results: List of retrieved document lists per query
        metrics: List of metrics to compute

    Returns:
        Dictionary of metric scores
    """
    if metrics is None:
        metrics = ['hit@1', 'hit@3', 'hit@5', 'ndcg@5', 'ndcg@10', 'mrr']

    results = {}

    for metric in metrics:
        if metric.startswith('hit@'):
            k = int(metric.split('@')[1])
            scores = [hit_at_k(q['relevant_docs'], [r['id'] for r in results], k)
                     for q, results in zip(queries, retrieval_results)]
            results[metric] = sum(scores) / len(scores)

        elif metric.startswith('ndcg@'):
            k = int(metric.split('@')[1])
            scores = [ndcg_at_k(q['relevant_docs'], [r['id'] for r in results], k)
                     for q, results in zip(queries, retrieval_results)]
            results[metric] = sum(scores) / len(scores)

        elif metric == 'mrr':
            relevant_lists = [q['relevant_docs'] for q in queries]
            predicted_lists = [[r['id'] for r in results] for results in retrieval_results]
            results[metric] = mean_reciprocal_rank(relevant_lists, predicted_lists)

    return results
```

## 6. Statistical Analysis

### 6.1 Confidence Intervals

For reliable evaluation, metrics should be reported with confidence intervals:

```python
def calculate_confidence_interval(scores: List[float],
                                confidence_level: float = 0.95) -> Tuple[float, float]:
    """
    Calculate confidence interval for a list of metric scores.

    Args:
        scores: List of individual query scores
        confidence_level: Desired confidence level (e.g., 0.95 for 95%)

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    n = len(scores)
    mean_score = sum(scores) / n
    std_error = statistics.stdev(scores) / math.sqrt(n)

    # t-distribution critical value (approximate for large n)
    t_value = 1.96 if confidence_level == 0.95 else 2.576  # 99% confidence

    margin_error = t_value * std_error

    return mean_score - margin_error, mean_score + margin_error
```

### 6.2 Statistical Significance Testing

```python
def statistical_significance_test(scores_a: List[float],
                                scores_b: List[float],
                                alpha: float = 0.05) -> Dict[str, float]:
    """
    Perform statistical significance test between two sets of scores.

    Args:
        scores_a: Scores from system A
        scores_b: Scores from system B
        alpha: Significance level

    Returns:
        Dictionary with test statistics and p-value
    """
    # Perform t-test
    t_stat, p_value = stats.ttest_ind(scores_a, scores_b)

    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < alpha,
        'mean_a': sum(scores_a) / len(scores_a),
        'mean_b': sum(scores_b) / len(scores_b)
    }
```

## 7. Validation and Testing

### 7.1 Known Test Cases

#### Test Case 1: Perfect Retrieval
```
Relevant: ["doc1", "doc2"]
Retrieved: ["doc1", "doc2", "doc3", "doc4"]

Expected:
- Hit@1: 1.0
- Hit@2: 1.0
- NDCG@2: 1.0
- MRR: 1.0
```

#### Test Case 2: No Relevant Documents Retrieved
```
Relevant: ["doc1", "doc2"]
Retrieved: ["doc3", "doc4", "doc5", "doc6"]

Expected:
- Hit@10: 0.0
- NDCG@10: 0.0
- MRR: 0.0
```

#### Test Case 3: Partial Retrieval
```
Relevant: ["doc1", "doc2", "doc3"]
Retrieved: ["doc4", "doc1", "doc5", "doc2"]

Expected:
- Hit@2: 1.0 (doc1 at position 2)
- Hit@1: 0.0
- NDCG@4: ≈0.693
- MRR: 0.5 (first relevant at position 2)
```

### 7.2 Edge Cases

- **Empty relevant set**: Should return 0.0 for all metrics
- **Empty retrieved set**: Should return 0.0 for all metrics
- **Single relevant document**: Test boundary conditions
- **Large k values**: Test performance with k > number of retrieved documents

## 8. Performance Considerations

### 8.1 Computational Complexity

| Metric | Time Complexity | Space Complexity |
|--------|----------------|------------------|
| Hit@k | O(k) | O(k + \|G\|) |
| NDCG@k | O(k log k) | O(k + \|G\|) |
| MRR | O(\|Q\| × k) | O(\|Q\| × k) |
| MAP@k | O(\|Q\| × k) | O(\|Q\| × k) |

### 8.2 Optimization Strategies

#### Batch Processing
```python
def batch_evaluate_metrics(queries: List[Dict], batch_size: int = 100):
    """Process metrics in batches to optimize memory usage"""
    results = []

    for i in range(0, len(queries), batch_size):
        batch = queries[i:i + batch_size]
        batch_results = evaluate_retrieval_batch(batch)
        results.extend(batch_results)

    return results
```

#### Caching for Repeated Evaluations
```python
class MetricsCache:
    """Cache computed metrics to avoid redundant calculations"""

    def __init__(self):
        self._cache = {}

    def get_cached_metric(self, query_id: str, metric: str, k: int) -> Optional[float]:
        """Retrieve cached metric value"""
        key = f"{query_id}_{metric}_{k}"
        return self._cache.get(key)

    def cache_metric(self, query_id: str, metric: str, k: int, value: float):
        """Cache metric value"""
        key = f"{query_id}_{metric}_{k}"
        self._cache[key] = value
```

## 9. Integration with Evaluation Harness

### 9.1 Metric Registration

```python
class RetrievalMetricsSuite:
    """Comprehensive suite of retrieval metrics"""

    def __init__(self):
        self.metrics = {
            'hit': self._hit_at_k,
            'ndcg': self._ndcg_at_k,
            'mrr': self._mean_reciprocal_rank,
            'map': self._mean_average_precision,
            'recall': self._recall_at_k,
            'precision': self._precision_at_k
        }

    def evaluate(self, metric_name: str, *args, **kwargs) -> float:
        """Evaluate specified metric"""
        if metric_name not in self.metrics:
            raise ValueError(f"Unknown metric: {metric_name}")

        return self.metrics[metric_name](*args, **kwargs)
```

### 9.2 Configuration Schema

```yaml
retrieval_metrics:
  enabled_metrics:
    - hit@1
    - hit@3
    - hit@5
    - hit@10
    - ndcg@5
    - ndcg@10
    - mrr
    - map@10

  graded_relevance: false  # Enable graded relevance scoring

  statistical_analysis:
    confidence_intervals: true
    significance_testing: true
    alpha_level: 0.05

  performance:
    batch_size: 100
    caching_enabled: true
    parallel_processing: true
```

## 10. Future Enhancements

### 10.1 Advanced Metrics

- **ERR@k (Expected Reciprocal Rank)**: Probabilistic ranking metric
- **rbp@k (Rank-Biased Precision)**: Patience-weighted ranking metric
- **Q-measure**: Query-level evaluation metric
- **Diversity Metrics**: Novelty and diversity in retrieved results

### 10.2 Machine Learning Integration

- **Learning-to-Rank Evaluation**: Integration with LTR model evaluation
- **Neural Ranking Metrics**: Deep learning-based relevance assessment
- **User Behavior Modeling**: Click-through rate and dwell time analysis

### 10.3 Multi-Modal Extensions

- **Image Retrieval Metrics**: Specialized metrics for visual content
- **Cross-Modal Metrics**: Evaluation of text-image retrieval systems
- **Temporal Metrics**: Time-aware retrieval evaluation

---

This specification provides comprehensive mathematical definitions and implementation guidance for retrieval quality metrics, ensuring consistent and accurate evaluation of the Personal RAG Chatbot's retrieval system.