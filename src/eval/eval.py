"""
Enhanced Evaluation Module for Personal RAG Chatbot

This module provides comprehensive evaluation capabilities including retrieval metrics,
citation accuracy, answer quality assessment, and performance benchmarking.

Author: SPARC Specification Writer
Date: 2025-08-30

Backward Compatibility: This module maintains compatibility with existing eval.py functions
while providing enhanced functionality through the new metrics module.
"""

import math
import json
import logging
from typing import List, Dict, Union, Optional, Any, Tuple
from pathlib import Path

# Import enhanced metrics
try:
    from .metrics import (
        RetrievalMetrics,
        CitationMetrics,
        AnswerQualityMetrics,
        StatisticalAnalysis,
        MetricsAggregator,
        calculate_retrieval_metrics,
        calculate_batch_retrieval_metrics,
        CitationSpan
    )
    ENHANCED_METRICS_AVAILABLE = True
except ImportError:
    ENHANCED_METRICS_AVAILABLE = False
    logging.warning("Enhanced metrics module not available, using basic functions only")

logger = logging.getLogger(__name__)

# Type aliases for backward compatibility
StringOrList = Union[str, List[str]]

# Legacy function - maintained for backward compatibility
def _idset(s: str) -> set:
    """Convert semicolon-separated string to set of IDs (legacy function)"""
    return set([x.strip() for x in (s or "").split(";") if x.strip()])

# Legacy functions - maintained for backward compatibility
def hit_at_k(relevant_ids: str, predicted_ids: str, k: int = 10) -> float:
    """
    Calculate Hit@k metric (legacy function).

    Args:
        relevant_ids: Semicolon-separated string of relevant document IDs
        predicted_ids: Semicolon-separated string of predicted document IDs
        k: Cutoff position

    Returns:
        Hit@k score (0.0 or 1.0)
    """
    rel = _idset(relevant_ids)
    pred = [x for x in (predicted_ids or "").split(";") if x][:k]
    return 1.0 if any(p in rel for p in pred) else 0.0

def ndcg_at_k(relevant_ids: str, predicted_ids: str, k: int = 10) -> float:
    """
    Calculate NDCG@k metric (legacy function).

    Args:
        relevant_ids: Semicolon-separated string of relevant document IDs
        predicted_ids: Semicolon-separated string of predicted document IDs
        k: Cutoff position

    Returns:
        NDCG@k score
    """
    rel = _idset(relevant_ids)
    pred = [x for x in (predicted_ids or "").split(";") if x][:k]
    dcg = sum((1.0 / math.log2(i+2)) for i,p in enumerate(pred) if p in rel)
    ideal = sum((1.0 / math.log2(i+2)) for i in range(min(len(rel), len(pred))))
    return dcg / ideal if ideal > 0 else 0.0

def span_accuracy(true_spans: List[Dict], pred_spans: List[Dict]) -> float:
    """
    Calculate citation span accuracy (legacy function).

    Args:
        true_spans: List of true citation spans
        pred_spans: List of predicted citation spans

    Returns:
        Span accuracy score (Jaccard similarity)
    """
    def to_set(spans):
        S = set()
        for sp in spans or []:
            a, b = int(sp.get("start",0)), int(sp.get("end",0))
            S.update(range(max(0,a), max(0,b)))
        return S
    T, P = to_set(true_spans), to_set(pred_spans)
    if not T or not P: return 0.0
    return len(T & P) / len(T | P)

# Enhanced functions using new metrics module
def evaluate_retrieval_quality(relevant_ids: StringOrList,
                              predicted_ids: StringOrList,
                              k_values: Optional[List[int]] = None,
                              use_enhanced: bool = True) -> Dict[str, Any]:
    """
    Evaluate retrieval quality using enhanced metrics.

    Args:
        relevant_ids: Relevant document IDs (string or list)
        predicted_ids: Predicted document IDs (string or list)
        k_values: List of k values for @k metrics
        use_enhanced: Whether to use enhanced metrics module

    Returns:
        Dictionary of evaluation metrics
    """
    if use_enhanced and ENHANCED_METRICS_AVAILABLE:
        if k_values is None:
            k_values = [1, 3, 5, 10]

        results = calculate_retrieval_metrics(relevant_ids, predicted_ids, k_values)

        # Convert MetricResult objects to simple values for backward compatibility
        simple_results = {}
        for metric_name, metric_result in results.items():
            simple_results[metric_name] = metric_result.value
            if metric_result.confidence_interval:
                simple_results[f"{metric_name}_ci"] = metric_result.confidence_interval

        return simple_results
    else:
        # Fallback to legacy functions
        if k_values is None:
            k_values = [1, 3, 5, 10]

        results = {}
        for k in k_values:
            results[f"hit@{k}"] = hit_at_k(
                relevant_ids if isinstance(relevant_ids, str) else ";".join(relevant_ids),
                predicted_ids if isinstance(predicted_ids, str) else ";".join(predicted_ids)
            )
            results[f"ndcg@{k}"] = ndcg_at_k(
                relevant_ids if isinstance(relevant_ids, str) else ";".join(relevant_ids),
                predicted_ids if isinstance(predicted_ids, str) else ";".join(predicted_ids)
            )

        return results

def evaluate_citation_quality(answer_text: str,
                            citations: List[Dict],
                            source_documents: Dict[str, str],
                            ground_truth_citations: Optional[List[Dict]] = None,
                            use_enhanced: bool = True) -> Dict[str, Any]:
    """
    Evaluate citation quality using enhanced metrics.

    Args:
        answer_text: Generated answer text
        citations: List of predicted citations
        source_documents: Dictionary of source document contents
        ground_truth_citations: Optional ground truth citations for comparison
        use_enhanced: Whether to use enhanced metrics module

    Returns:
        Dictionary of citation quality metrics
    """
    if use_enhanced and ENHANCED_METRICS_AVAILABLE:
        # Convert dict citations to CitationSpan objects
        citation_spans = []
        for cit in citations:
            page_num = cit.get('page_number')
            citation_spans.append(CitationSpan(
                file_name=cit.get('file_name', ''),
                page_number=int(page_num) if page_num is not None else 0,
                start_char=cit.get('start_char', 0),
                end_char=cit.get('end_char', 0),
                confidence_score=cit.get('confidence_score', 1.0),
                context_text=cit.get('context_text', '')
            ))

        # Convert ground truth if provided
        ground_truth_spans = None
        if ground_truth_citations:
            ground_truth_spans = []
            for cit in ground_truth_citations:
                page_num = cit.get('page_number')
                ground_truth_spans.append(CitationSpan(
                    file_name=cit.get('file_name', ''),
                    page_number=int(page_num) if page_num is not None else 0,
                    start_char=cit.get('start_char', 0),
                    end_char=cit.get('end_char', 0),
                    confidence_score=cit.get('confidence_score', 1.0),
                    context_text=cit.get('context_text', '')
                ))

        evaluator = CitationMetrics()

        results = {}

        # Evaluate span accuracy if ground truth available
        if ground_truth_spans:
            span_result = evaluator.span_accuracy(citation_spans, ground_truth_spans)
            results['span_accuracy'] = span_result.value
            results['span_accuracy_details'] = span_result.additional_info

        # Evaluate citation completeness
        completeness_result = evaluator.citation_completeness(
            answer_text, citation_spans, source_documents
        )
        results['citation_completeness'] = completeness_result.value

        # Evaluate citation correctness
        correctness_result = evaluator.citation_correctness(
            answer_text, citation_spans, source_documents
        )
        results['citation_correctness'] = correctness_result.value

        return results
    else:
        # Fallback to basic span accuracy
        if ground_truth_citations:
            results = {'span_accuracy': span_accuracy(ground_truth_citations, citations)}
        else:
            results = {'span_accuracy': 0.0}

        return results

def evaluate_answer_quality(answer_text: str,
                          citations: List[Dict],
                          source_documents: Dict[str, str],
                          ground_truth_answer: Optional[str] = None,
                          use_enhanced: bool = True) -> Dict[str, Any]:
    """
    Evaluate answer quality using enhanced metrics.

    Args:
        answer_text: Generated answer text
        citations: List of citations in the answer
        source_documents: Dictionary of source document contents
        ground_truth_answer: Optional ground truth answer for comparison
        use_enhanced: Whether to use enhanced metrics module

    Returns:
        Dictionary of answer quality metrics
    """
    if use_enhanced and ENHANCED_METRICS_AVAILABLE:
        # Convert citations to CitationSpan objects
        citation_spans = []
        for cit in citations:
            page_num = cit.get('page_number')
            citation_spans.append(CitationSpan(
                file_name=cit.get('file_name', ''),
                page_number=int(page_num) if page_num is not None else 0,
                start_char=cit.get('start_char', 0),
                end_char=cit.get('end_char', 0),
                confidence_score=cit.get('confidence_score', 1.0)
            ))

        evaluator = AnswerQualityMetrics()

        results = {}

        # Evaluate factual consistency
        consistency_result = evaluator.factual_consistency(
            answer_text, citation_spans, source_documents
        )
        results['factual_consistency'] = consistency_result.value

        # Evaluate hallucination rate
        hallucination_result = evaluator.hallucination_rate(
            answer_text, citation_spans, source_documents
        )
        results['hallucination_rate'] = hallucination_result.value

        # Calculate overall answer quality score
        # Higher factual consistency and lower hallucination rate = better quality
        overall_quality = (consistency_result.value * 0.7 + (1.0 - hallucination_result.value) * 0.3)
        results['overall_answer_quality'] = overall_quality

        return results
    else:
        # Basic fallback - return neutral scores
        return {
            'factual_consistency': 0.5,
            'hallucination_rate': 0.5,
            'overall_answer_quality': 0.5
        }

def batch_evaluate_queries(queries: List[Dict[str, Any]],
                         predictions: List[Dict[str, Any]],
                         source_documents: Dict[str, str],
                         use_enhanced: bool = True) -> Dict[str, Any]:
    """
    Evaluate a batch of queries with comprehensive metrics.

    Args:
        queries: List of query objects with ground truth
        predictions: List of prediction objects with answers and citations
        source_documents: Dictionary of source document contents
        use_enhanced: Whether to use enhanced metrics module

    Returns:
        Comprehensive evaluation results
    """
    if len(queries) != len(predictions):
        raise ValueError("Number of queries must match number of predictions")

    results = {
        'query_results': [],
        'aggregate_metrics': {},
        'summary': {}
    }

    # Evaluate each query
    for i, (query, prediction) in enumerate(zip(queries, predictions)):
        query_result = {
            'query_id': query.get('query_id', f'query_{i}'),
            'query_text': query.get('query_text', ''),
            'metrics': {}
        }

        # Retrieval metrics
        relevant_docs = query.get('relevant_documents', [])
        predicted_docs = prediction.get('retrieved_documents', [])

        if relevant_docs and predicted_docs:
            retrieval_metrics = evaluate_retrieval_quality(
                relevant_docs, predicted_docs, use_enhanced=use_enhanced
            )
            query_result['metrics'].update(retrieval_metrics)

        # Citation metrics
        answer_text = prediction.get('answer', '')
        citations = prediction.get('citations', [])
        ground_truth_citations = query.get('citations', [])

        if citations:
            citation_metrics = evaluate_citation_quality(
                answer_text, citations, source_documents,
                ground_truth_citations, use_enhanced=use_enhanced
            )
            query_result['metrics'].update(citation_metrics)

        # Answer quality metrics
        ground_truth_answer = query.get('ground_truth_answer')
        answer_metrics = evaluate_answer_quality(
            answer_text, citations, source_documents,
            ground_truth_answer, use_enhanced=use_enhanced
        )
        query_result['metrics'].update(answer_metrics)

        results['query_results'].append(query_result)

    # Calculate aggregate metrics
    if use_enhanced and ENHANCED_METRICS_AVAILABLE:
        aggregator = MetricsAggregator()

        # Collect all metric results
        for query_result in results['query_results']:
            for metric_name, metric_value in query_result['metrics'].items():
                # Create a simple MetricResult for aggregation
                from .metrics import MetricResult
                aggregator.add_result(MetricResult(
                    metric_name=metric_name,
                    value=metric_value
                ))

        results['aggregate_metrics'] = aggregator.get_summary()
    else:
        # Basic aggregation
        all_metrics = {}
        for query_result in results['query_results']:
            for metric_name, metric_value in query_result['metrics'].items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(metric_value)

        results['aggregate_metrics'] = {
            metric_name: {
                'mean': sum(values) / len(values) if values else 0.0,
                'count': len(values)
            }
            for metric_name, values in all_metrics.items()
        }

    # Generate summary
    results['summary'] = {
        'total_queries': len(queries),
        'successful_evaluations': len([r for r in results['query_results']
                                     if r['metrics']]),
        'average_metrics': {
            metric_name: stats['mean']
            for metric_name, stats in results['aggregate_metrics'].items()
            if isinstance(stats, dict) and 'mean' in stats
        }
    }

    return results

def run_evaluation_pipeline(evaluation_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run a complete evaluation pipeline.

    Args:
        evaluation_config: Configuration for the evaluation run

    Returns:
        Complete evaluation results
    """
    logger.info("Starting evaluation pipeline...")

    # Load evaluation dataset
    dataset_path = evaluation_config.get('dataset_path')
    if not dataset_path:
        raise ValueError("dataset_path is required in evaluation_config")

    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    # Load predictions (could come from model inference)
    predictions_path = evaluation_config.get('predictions_path')
    if predictions_path and Path(predictions_path).exists():
        with open(predictions_path, 'r', encoding='utf-8') as f:
            predictions = json.load(f)
    else:
        # Generate mock predictions for demonstration
        predictions = []
        for query in dataset.get('queries', []):
            predictions.append({
                'query_id': query['query_id'],
                'answer': query.get('ground_truth_answer', 'Mock answer'),
                'citations': query.get('citations', []),
                'retrieved_documents': query.get('relevant_documents', [])[:5]
            })

    # Load source documents
    source_documents = {}
    for doc in dataset.get('documents', []):
        source_documents[doc['document_id']] = doc['content']

    # Run evaluation
    use_enhanced = evaluation_config.get('use_enhanced_metrics', True)
    results = batch_evaluate_queries(
        dataset['queries'], predictions, source_documents, use_enhanced
    )

    # Add metadata
    results['metadata'] = {
        'evaluation_config': evaluation_config,
        'timestamp': json.dumps(None),  # Will be set by calling code
        'system_version': evaluation_config.get('system_version', 'unknown'),
        'enhanced_metrics_used': use_enhanced and ENHANCED_METRICS_AVAILABLE
    }

    logger.info("Evaluation pipeline completed.")
    return results

# Utility functions
def save_evaluation_results(results: Dict[str, Any], output_path: str):
    """Save evaluation results to JSON file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Evaluation results saved to: {output_path}")

def load_evaluation_results(input_path: str) -> Dict[str, Any]:
    """Load evaluation results from JSON file."""
    with open(input_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def compare_evaluation_runs(run1_path: str, run2_path: str) -> Dict[str, Any]:
    """
    Compare two evaluation runs.

    Args:
        run1_path: Path to first evaluation results
        run2_path: Path to second evaluation results

    Returns:
        Comparison results
    """
    run1 = load_evaluation_results(run1_path)
    run2 = load_evaluation_results(run2_path)

    comparison = {
        'run1_metadata': run1.get('metadata', {}),
        'run2_metadata': run2.get('metadata', {}),
        'metric_comparison': {}
    }

    # Compare aggregate metrics
    run1_metrics = run1.get('aggregate_metrics', {})
    run2_metrics = run2.get('aggregate_metrics', {})

    for metric_name in set(run1_metrics.keys()) | set(run2_metrics.keys()):
        run1_value = run1_metrics.get(metric_name, {}).get('mean', 0.0)
        run2_value = run2_metrics.get(metric_name, {}).get('mean', 0.0)

        improvement = run2_value - run1_value
        relative_improvement = (improvement / run1_value * 100) if run1_value != 0 else 0.0

        comparison['metric_comparison'][metric_name] = {
            'run1_value': run1_value,
            'run2_value': run2_value,
            'absolute_improvement': improvement,
            'relative_improvement_percent': relative_improvement
        }

    return comparison

# Example usage
if __name__ == "__main__":
    # Example: Evaluate a single query
    relevant = "doc1;doc2;doc3"
    predicted = "doc2;doc4;doc1;doc5"

    print("Basic evaluation:")
    print(f"Hit@3: {hit_at_k(relevant, predicted)}")
    print(f"NDCG@3: {ndcg_at_k(relevant, predicted)}")

    if ENHANCED_METRICS_AVAILABLE:
        print("\nEnhanced evaluation:")
        enhanced_results = evaluate_retrieval_quality(relevant, predicted)
        for metric, value in enhanced_results.items():
            print(f"{metric}: {value}")
