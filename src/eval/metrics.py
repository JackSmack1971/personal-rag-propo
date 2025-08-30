"""
Enhanced Metrics Module for Personal RAG Chatbot Evaluation

This module provides comprehensive evaluation metrics for retrieval quality,
citation accuracy, answer quality, and performance benchmarking.

Author: SPARC Specification Writer
Date: 2025-08-30
"""

import math
import statistics
import time
from typing import List, Dict, Any, Union, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


# Type definitions
StringOrList = Union[str, List[str]]


@dataclass
class CitationSpan:
    """Represents a single citation span"""
    file_name: str
    page_number: int
    start_char: int
    end_char: int
    confidence_score: float = 1.0
    extraction_method: str = "llm"
    context_text: str = ""
    relevance_score: float = 1.0


@dataclass
class MetricResult:
    """Result of a metric calculation"""
    metric_name: str
    value: float
    confidence_interval: Optional[Tuple[float, float]] = None
    additional_info: Optional[Dict[str, Any]] = None


class RetrievalMetrics:
    """Comprehensive retrieval quality metrics"""

    @staticmethod
    def hit_at_k(relevant_ids: StringOrList,
                 predicted_ids: StringOrList,
                 k: int = 10) -> MetricResult:
        """
        Calculate Hit@k metric.

        Args:
            relevant_ids: Ground truth relevant document IDs
            predicted_ids: Retrieved document IDs (ordered by relevance)
            k: Cutoff position

        Returns:
            MetricResult with Hit@k score
        """
        rel_set = _to_set(relevant_ids)
        pred_list = _to_list(predicted_ids)[:k]

        hit_score = 1.0 if any(pid in rel_set for pid in pred_list) else 0.0

        return MetricResult(
            metric_name=f"hit@{k}",
            value=hit_score,
            additional_info={
                "relevant_count": len(rel_set),
                "retrieved_count": len(pred_list),
                "hit": hit_score > 0
            }
        )

    @staticmethod
    def ndcg_at_k(relevant_ids: StringOrList,
                  predicted_ids: StringOrList,
                  k: int = 10,
                  relevance_grades: Optional[Dict[str, float]] = None) -> MetricResult:
        """
        Calculate NDCG@k metric.

        Args:
            relevant_ids: Ground truth relevant document IDs
            predicted_ids: Retrieved document IDs (ordered by relevance)
            k: Cutoff position
            relevance_grades: Optional relevance grades for graded relevance

        Returns:
            MetricResult with NDCG@k score
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
            relevant_docs.sort(key=lambda x: relevance_grades.get(x, 0), reverse=True)

        idcg = 0.0
        for i, doc_id in enumerate(relevant_docs[:k]):
            relevance = _get_relevance(doc_id, rel_set, relevance_grades)
            idcg += relevance / math.log2(i + 2)

        ndcg_score = dcg / idcg if idcg > 0 else 0.0

        return MetricResult(
            metric_name=f"ndcg@{k}",
            value=ndcg_score,
            additional_info={
                "dcg": dcg,
                "idcg": idcg,
                "relevant_count": len(rel_set),
                "retrieved_count": len(pred_list)
            }
        )

    @staticmethod
    def mean_reciprocal_rank(relevant_ids_list: List[StringOrList],
                           predicted_ids_list: List[StringOrList]) -> MetricResult:
        """
        Calculate Mean Reciprocal Rank.

        Args:
            relevant_ids_list: List of ground truth relevant document IDs per query
            predicted_ids_list: List of retrieved document IDs per query

        Returns:
            MetricResult with MRR score
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

        mrr_score = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0

        return MetricResult(
            metric_name="mrr",
            value=mrr_score,
            additional_info={
                "query_count": len(reciprocal_ranks),
                "average_reciprocal_rank": mrr_score,
                "max_reciprocal_rank": max(reciprocal_ranks) if reciprocal_ranks else 0
            }
        )

    @staticmethod
    def mean_average_precision(relevant_ids_list: List[StringOrList],
                             predicted_ids_list: List[StringOrList],
                             k: int = 10) -> MetricResult:
        """
        Calculate Mean Average Precision at k.

        Args:
            relevant_ids_list: List of ground truth relevant document IDs per query
            predicted_ids_list: List of retrieved document IDs per query
            k: Cutoff position

        Returns:
            MetricResult with MAP@k score
        """
        average_precisions = []

        for rel_ids, pred_ids in zip(relevant_ids_list, predicted_ids_list):
            rel_set = _to_set(rel_ids)
            pred_list = _to_list(pred_ids)[:k]

            if not rel_set:
                average_precisions.append(0.0)
                continue

            # Calculate Average Precision
            num_relevant = 0
            precision_sum = 0.0

            for i, doc_id in enumerate(pred_list):
                if doc_id in rel_set:
                    num_relevant += 1
                    precision_at_i = num_relevant / (i + 1)
                    precision_sum += precision_at_i

            ap = precision_sum / len(rel_set) if rel_set else 0.0
            average_precisions.append(ap)

        map_score = sum(average_precisions) / len(average_precisions) if average_precisions else 0.0

        return MetricResult(
            metric_name=f"map@{k}",
            value=map_score,
            additional_info={
                "query_count": len(average_precisions),
                "average_precisions": average_precisions
            }
        )

    @staticmethod
    def recall_at_k(relevant_ids: StringOrList,
                   predicted_ids: StringOrList,
                   k: int = 10) -> MetricResult:
        """
        Calculate Recall@k metric.

        Args:
            relevant_ids: Ground truth relevant document IDs
            predicted_ids: Retrieved document IDs
            k: Cutoff position

        Returns:
            MetricResult with Recall@k score
        """
        rel_set = _to_set(relevant_ids)
        pred_set = set(_to_list(predicted_ids)[:k])

        if not rel_set:
            return MetricResult(metric_name=f"recall@{k}", value=0.0)

        recall_score = len(rel_set & pred_set) / len(rel_set)

        return MetricResult(
            metric_name=f"recall@{k}",
            value=recall_score,
            additional_info={
                "relevant_count": len(rel_set),
                "retrieved_relevant": len(rel_set & pred_set),
                "total_retrieved": len(pred_set)
            }
        )

    @staticmethod
    def precision_at_k(relevant_ids: StringOrList,
                      predicted_ids: StringOrList,
                      k: int = 10) -> MetricResult:
        """
        Calculate Precision@k metric.

        Args:
            relevant_ids: Ground truth relevant document IDs
            predicted_ids: Retrieved document IDs
            k: Cutoff position

        Returns:
            MetricResult with Precision@k score
        """
        rel_set = _to_set(relevant_ids)
        pred_list = _to_list(predicted_ids)[:k]

        if not pred_list:
            return MetricResult(metric_name=f"precision@{k}", value=0.0)

        precision_score = sum(1 for pid in pred_list if pid in rel_set) / len(pred_list)

        return MetricResult(
            metric_name=f"precision@{k}",
            value=precision_score,
            additional_info={
                "relevant_count": len(rel_set),
                "retrieved_count": len(pred_list),
                "correct_retrievals": sum(1 for pid in pred_list if pid in rel_set)
            }
        )


class CitationMetrics:
    """Citation accuracy and quality metrics"""

    @staticmethod
    def span_accuracy(predicted_spans: List[CitationSpan],
                     true_spans: List[CitationSpan],
                     tolerance_chars: int = 10) -> MetricResult:
        """
        Calculate citation span accuracy with tolerance.

        Args:
            predicted_spans: Predicted citation spans
            true_spans: Ground truth citation spans
            tolerance_chars: Character tolerance for matching

        Returns:
            MetricResult with span accuracy metrics
        """
        if not true_spans or not predicted_spans:
            return MetricResult(
                metric_name="span_accuracy",
                value=0.0,
                additional_info={"reason": "empty_spans"}
            )

        # Create character-level sets for each span
        true_char_sets = [_span_to_char_set(span) for span in true_spans]
        pred_char_sets = [_span_to_char_set(span) for span in predicted_spans]

        # Calculate pairwise similarities
        similarities = []
        for true_set in true_char_sets:
            for pred_set in pred_char_sets:
                # Exact match
                exact_overlap = len(true_set & pred_set)
                exact_union = len(true_set | pred_set)
                exact_jaccard = exact_overlap / exact_union if exact_union > 0 else 0

                # Tolerance match
                expanded_pred = set()
                for char_pos in pred_set:
                    expanded_pred.update(range(
                        max(0, char_pos - tolerance_chars),
                        char_pos + tolerance_chars + 1
                    ))

                tolerance_overlap = len(true_set & expanded_pred)
                tolerance_union = len(true_set | expanded_pred)
                tolerance_jaccard = tolerance_overlap / tolerance_union if tolerance_union > 0 else 0

                similarities.append({
                    'exact_jaccard': exact_jaccard,
                    'tolerance_jaccard': tolerance_jaccard,
                    'overlap_chars': exact_overlap
                })

        if not similarities:
            return MetricResult(
                metric_name="span_accuracy",
                value=0.0,
                additional_info={"reason": "no_similarities"}
            )

        # Calculate aggregate metrics
        best_matches = [max(sims, key=lambda x: x['tolerance_jaccard']) for sims in [similarities]]

        mean_exact_jaccard = sum(s['exact_jaccard'] for s in best_matches) / len(best_matches)
        mean_tolerance_jaccard = sum(s['tolerance_jaccard'] for s in best_matches) / len(best_matches)

        perfect_matches = sum(1 for s in best_matches if s['exact_jaccard'] == 1.0)
        good_matches = sum(1 for s in best_matches if s['tolerance_jaccard'] >= 0.8)

        return MetricResult(
            metric_name="span_accuracy",
            value=mean_tolerance_jaccard,
            additional_info={
                "mean_exact_jaccard": mean_exact_jaccard,
                "mean_tolerance_jaccard": mean_tolerance_jaccard,
                "perfect_matches": perfect_matches,
                "good_matches": good_matches,
                "total_predictions": len(predicted_spans),
                "total_ground_truth": len(true_spans)
            }
        )

    @staticmethod
    def citation_completeness(answer_text: str,
                            citations: List[CitationSpan],
                            source_documents: Dict[str, str]) -> MetricResult:
        """
        Calculate citation completeness - fraction of claims that are cited.

        Args:
            answer_text: Generated answer text
            citations: List of citation spans
            source_documents: Source document contents

        Returns:
            MetricResult with citation completeness score
        """
        # Extract claims from answer (simplified - in practice would use NLP)
        claims = _extract_claims_from_text(answer_text)

        if not claims:
            return MetricResult(
                metric_name="citation_completeness",
                value=0.0,
                additional_info={"reason": "no_claims_found"}
            )

        cited_claims = 0

        for claim in claims:
            # Check if claim is supported by any citation
            claim_cited = False

            for citation in citations:
                doc_content = source_documents.get(citation.file_name, "")
                if not doc_content:
                    continue

                cited_text = doc_content[citation.start_char:citation.end_char]

                # Simple text overlap check (in practice would use semantic similarity)
                if _text_overlap(claim, cited_text) > 0.3:
                    claim_cited = True
                    break

            if claim_cited:
                cited_claims += 1

        completeness_score = cited_claims / len(claims)

        return MetricResult(
            metric_name="citation_completeness",
            value=completeness_score,
            additional_info={
                "total_claims": len(claims),
                "cited_claims": cited_claims,
                "uncited_claims": len(claims) - cited_claims
            }
        )

    @staticmethod
    def citation_correctness(answer_text: str,
                           citations: List[CitationSpan],
                           source_documents: Dict[str, str]) -> MetricResult:
        """
        Calculate citation correctness - fraction of citations that support their claims.

        Args:
            answer_text: Generated answer text
            citations: List of citation spans
            source_documents: Source document contents

        Returns:
            MetricResult with citation correctness score
        """
        if not citations:
            return MetricResult(
                metric_name="citation_correctness",
                value=0.0,
                additional_info={"reason": "no_citations"}
            )

        correct_citations = 0

        for citation in citations:
            doc_content = source_documents.get(citation.file_name, "")
            if not doc_content:
                continue

            cited_text = doc_content[citation.start_char:citation.end_char]

            # Find associated claim (simplified)
            associated_claim = _find_associated_claim(answer_text, citation)

            if associated_claim:
                # Check if citation supports the claim (simplified text overlap)
                if _text_overlap(associated_claim, cited_text) > 0.3:
                    correct_citations += 1

        correctness_score = correct_citations / len(citations)

        return MetricResult(
            metric_name="citation_correctness",
            value=correctness_score,
            additional_info={
                "total_citations": len(citations),
                "correct_citations": correct_citations,
                "incorrect_citations": len(citations) - correct_citations
            }
        )


class AnswerQualityMetrics:
    """Answer quality and relevance metrics"""

    @staticmethod
    def factual_consistency(answer: str,
                          citations: List[CitationSpan],
                          source_documents: Dict[str, str]) -> MetricResult:
        """
        Calculate factual consistency between answer and cited sources.

        Args:
            answer: Generated answer text
            citations: List of citation spans
            source_documents: Source document contents

        Returns:
            MetricResult with factual consistency score
        """
        if not citations:
            return MetricResult(
                metric_name="factual_consistency",
                value=0.0,
                additional_info={"reason": "no_citations"}
            )

        consistency_scores = []

        for citation in citations:
            doc_content = source_documents.get(citation.file_name, "")
            if not doc_content:
                continue

            cited_text = doc_content[citation.start_char:citation.end_char]
            associated_claim = _find_associated_claim(answer, citation)

            if associated_claim:
                # Calculate semantic similarity (simplified - would use embeddings)
                consistency = _calculate_text_similarity(associated_claim, cited_text)
                consistency_scores.append(consistency)

        if not consistency_scores:
            return MetricResult(
                metric_name="factual_consistency",
                value=0.0,
                additional_info={"reason": "no_associated_claims"}
            )

        mean_consistency = sum(consistency_scores) / len(consistency_scores)

        return MetricResult(
            metric_name="factual_consistency",
            value=mean_consistency,
            additional_info={
                "citation_count": len(consistency_scores),
                "consistency_scores": consistency_scores,
                "min_consistency": min(consistency_scores),
                "max_consistency": max(consistency_scores)
            }
        )

    @staticmethod
    def hallucination_rate(answer: str,
                         citations: List[CitationSpan],
                         source_documents: Dict[str, str]) -> MetricResult:
        """
        Calculate hallucination rate - claims not supported by sources.

        Args:
            answer: Generated answer text
            citations: List of citation spans
            source_documents: Source document contents

        Returns:
            MetricResult with hallucination rate
        """
        claims = _extract_claims_from_text(answer)

        if not claims:
            return MetricResult(
                metric_name="hallucination_rate",
                value=0.0,
                additional_info={"reason": "no_claims_found"}
            )

        hallucinated_claims = 0

        for claim in claims:
            claim_supported = False

            for citation in citations:
                doc_content = source_documents.get(citation.file_name, "")
                if not doc_content:
                    continue

                cited_text = doc_content[citation.start_char:citation.end_char]

                if _text_overlap(claim, cited_text) > 0.3:
                    claim_supported = True
                    break

            if not claim_supported:
                hallucinated_claims += 1

        hallucination_rate = hallucinated_claims / len(claims)

        return MetricResult(
            metric_name="hallucination_rate",
            value=hallucination_rate,
            additional_info={
                "total_claims": len(claims),
                "hallucinated_claims": hallucinated_claims,
                "supported_claims": len(claims) - hallucinated_claims
            }
        )


class StatisticalAnalysis:
    """Statistical analysis utilities for evaluation"""

    @staticmethod
    def calculate_confidence_interval(values: List[float],
                                    confidence_level: float = 0.95) -> Tuple[float, float]:
        """
        Calculate confidence interval for a list of values.

        Args:
            values: List of metric values
            confidence_level: Desired confidence level

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if len(values) < 2:
            mean_val = values[0] if values else 0.0
            return (mean_val, mean_val)

        mean_val = statistics.mean(values)
        std_error = statistics.stdev(values) / math.sqrt(len(values))

        # z-score for confidence level
        z_score = 1.96 if confidence_level == 0.95 else 2.576  # 99% confidence

        margin_error = z_score * std_error

        return (mean_val - margin_error, mean_val + margin_error)

    @staticmethod
    def statistical_significance_test(values_a: List[float],
                                    values_b: List[float],
                                    alpha: float = 0.05) -> Dict[str, Any]:
        """
        Perform statistical significance test between two sets of values.

        Args:
            values_a: Values from system A
            values_b: Values from system B
            alpha: Significance level

        Returns:
            Dictionary with test results
        """
        if len(values_a) < 2 or len(values_b) < 2:
            return {
                'significant': False,
                'reason': 'insufficient_samples',
                'mean_a': sum(values_a) / len(values_a) if values_a else 0,
                'mean_b': sum(values_b) / len(values_b) if values_b else 0
            }

        # Perform t-test
        try:
            from scipy import stats
            t_stat, p_value = stats.ttest_ind(values_a, values_b)
        except ImportError:
            # Fallback to simple t-test calculation
            mean_a = statistics.mean(values_a)
            mean_b = statistics.mean(values_b)
            var_a = statistics.variance(values_a) if len(values_a) > 1 else 0
            var_b = statistics.variance(values_b) if len(values_b) > 1 else 0

            if var_a + var_b == 0:
                return {'significant': False, 'reason': 'no_variance'}

            t_stat = (mean_a - mean_b) / math.sqrt(var_a/len(values_a) + var_b/len(values_b))
            # Approximate p-value (simplified)
            p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(t_stat) / math.sqrt(2))))

        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < alpha,
            'effect_size': abs(statistics.mean(values_a) - statistics.mean(values_b)),
            'mean_a': statistics.mean(values_a),
            'mean_b': statistics.mean(values_b),
            'std_a': statistics.stdev(values_a) if len(values_a) > 1 else 0,
            'std_b': statistics.stdev(values_b) if len(values_b) > 1 else 0
        }


class MetricsAggregator:
    """Aggregates and analyzes multiple metric results"""

    def __init__(self):
        self.results = []

    def add_result(self, result: MetricResult):
        """Add a metric result to the aggregator"""
        self.results.append(result)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for all results"""

        if not self.results:
            return {}

        # Group by metric type
        metric_groups = defaultdict(list)

        for result in self.results:
            metric_groups[result.metric_name].append(result.value)

        summary = {}

        for metric_name, values in metric_groups.items():
            summary[metric_name] = {
                'count': len(values),
                'mean': statistics.mean(values),
                'median': statistics.median(values),
                'std': statistics.stdev(values) if len(values) > 1 else 0,
                'min': min(values),
                'max': max(values),
                'confidence_interval': StatisticalAnalysis.calculate_confidence_interval(values)
            }

        return summary

    def compare_systems(self, system_a_results: List[MetricResult],
                       system_b_results: List[MetricResult]) -> Dict[str, Any]:
        """
        Compare two systems across all metrics.

        Args:
            system_a_results: Results from system A
            system_b_results: Results from system B

        Returns:
            Comparison results
        """
        # Group results by metric
        system_a_metrics = defaultdict(list)
        system_b_metrics = defaultdict(list)

        for result in system_a_results:
            system_a_metrics[result.metric_name].append(result.value)

        for result in system_b_results:
            system_b_metrics[result.metric_name].append(result.value)

        comparison = {}

        all_metrics = set(system_a_metrics.keys()) | set(system_b_metrics.keys())

        for metric in all_metrics:
            values_a = system_a_metrics.get(metric, [])
            values_b = system_b_metrics.get(metric, [])

            if values_a and values_b:
                comparison[metric] = StatisticalAnalysis.statistical_significance_test(
                    values_a, values_b
                )

        return comparison


# Utility functions

def _to_set(ids: StringOrList) -> set:
    """Convert string or list to set of IDs"""
    if isinstance(ids, str):
        return set(x.strip() for x in ids.split(";") if x.strip())
    elif isinstance(ids, list):
        return set(str(x).strip() for x in ids if x)
    else:
        return set()


def _to_list(ids: StringOrList) -> list:
    """Convert string or list to list of IDs"""
    if isinstance(ids, str):
        return [x.strip() for x in ids.split(";") if x.strip()]
    elif isinstance(ids, list):
        return [str(x).strip() for x in ids if x]
    else:
        return []


def _get_relevance(doc_id: str, relevant_set: set,
                  relevance_grades: Optional[Dict[str, float]] = None) -> float:
    """Get relevance score for a document"""
    if relevance_grades and doc_id in relevance_grades:
        return relevance_grades[doc_id]
    elif doc_id in relevant_set:
        return 1.0
    else:
        return 0.0


def _span_to_char_set(span: CitationSpan) -> set:
    """Convert citation span to character position set"""
    return set(range(span.start_char, span.end_char + 1))


def _extract_claims_from_text(text: str) -> List[str]:
    """Extract factual claims from text (simplified implementation)"""
    # This is a simplified implementation
    # In practice, would use NLP to identify factual claims
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    return sentences


def _find_associated_claim(answer_text: str, citation: CitationSpan) -> Optional[str]:
    """Find the claim associated with a citation (simplified)"""
    # This is a simplified implementation
    # In practice, would use more sophisticated NLP
    sentences = [s.strip() for s in answer_text.split('.') if s.strip()]

    # Simple heuristic: find sentence closest to citation position
    # This would need actual position mapping in practice
    return sentences[0] if sentences else None


def _text_overlap(text1: str, text2: str) -> float:
    """Calculate text overlap between two strings (simplified)"""
    if not text1 or not text2:
        return 0.0

    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    if not words1 or not words2:
        return 0.0

    intersection = words1 & words2
    union = words1 | words2

    return len(intersection) / len(union)


def _calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate text similarity (simplified implementation)"""
    # This is a simplified implementation
    # In practice, would use embeddings or more sophisticated similarity measures
    return _text_overlap(text1, text2)


# Convenience functions for common metric calculations

def calculate_retrieval_metrics(relevant_ids: StringOrList,
                              predicted_ids: StringOrList,
                              k_values: Optional[List[int]] = None) -> Dict[str, MetricResult]:
    """
    Calculate all retrieval metrics for a single query.

    Args:
        relevant_ids: Ground truth relevant document IDs
        predicted_ids: Retrieved document IDs (ordered by relevance)
        k_values: List of k values for @k metrics

    Returns:
        Dictionary of metric results
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]

    metrics = RetrievalMetrics()
    results = {}

    # Hit@k for all k values
    for k in k_values:
        results[f"hit@{k}"] = metrics.hit_at_k(relevant_ids, predicted_ids, k)

    # NDCG@k for all k values
    for k in k_values:
        results[f"ndcg@{k}"] = metrics.ndcg_at_k(relevant_ids, predicted_ids, k)

    # Recall@k and Precision@k
    for k in k_values:
        results[f"recall@{k}"] = metrics.recall_at_k(relevant_ids, predicted_ids, k)
        results[f"precision@{k}"] = metrics.precision_at_k(relevant_ids, predicted_ids, k)

    return results


def calculate_batch_retrieval_metrics(relevant_ids_list: List[StringOrList],
                                    predicted_ids_list: List[StringOrList],
                                    k_values: Optional[List[int]] = None) -> Dict[str, MetricResult]:
    """
    Calculate retrieval metrics for a batch of queries.

    Args:
        relevant_ids_list: List of ground truth relevant document IDs per query
        predicted_ids_list: List of retrieved document IDs per query
        k_values: List of k values for @k metrics

    Returns:
        Dictionary of aggregated metric results
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]

    metrics = RetrievalMetrics()
    results = {}

    # MRR
    results["mrr"] = metrics.mean_reciprocal_rank(relevant_ids_list, predicted_ids_list)

    # MAP@k
    for k in k_values:
        results[f"map@{k}"] = metrics.mean_average_precision(relevant_ids_list, predicted_ids_list, k)

    # Aggregate Hit@k, NDCG@k, etc. across queries
    for k in k_values:
        hit_scores = []
        ndcg_scores = []
        recall_scores = []
        precision_scores = []

        for rel_ids, pred_ids in zip(relevant_ids_list, predicted_ids_list):
            hit_result = metrics.hit_at_k(rel_ids, pred_ids, k)
            ndcg_result = metrics.ndcg_at_k(rel_ids, pred_ids, k)
            recall_result = metrics.recall_at_k(rel_ids, pred_ids, k)
            precision_result = metrics.precision_at_k(rel_ids, pred_ids, k)

            hit_scores.append(hit_result.value)
            ndcg_scores.append(ndcg_result.value)
            recall_scores.append(recall_result.value)
            precision_scores.append(precision_result.value)

        # Calculate means and confidence intervals
        results[f"hit@{k}_mean"] = MetricResult(
            metric_name=f"hit@{k}_mean",
            value=sum(hit_scores) / len(hit_scores) if hit_scores else 0.0,
            confidence_interval=StatisticalAnalysis.calculate_confidence_interval(hit_scores)
        )

        results[f"ndcg@{k}_mean"] = MetricResult(
            metric_name=f"ndcg@{k}_mean",
            value=sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0,
            confidence_interval=StatisticalAnalysis.calculate_confidence_interval(ndcg_scores)
        )

        results[f"recall@{k}_mean"] = MetricResult(
            metric_name=f"recall@{k}_mean",
            value=sum(recall_scores) / len(recall_scores) if recall_scores else 0.0,
            confidence_interval=StatisticalAnalysis.calculate_confidence_interval(recall_scores)
        )

        results[f"precision@{k}_mean"] = MetricResult(
            metric_name=f"precision@{k}_mean",
            value=sum(precision_scores) / len(precision_scores) if precision_scores else 0.0,
            confidence_interval=StatisticalAnalysis.calculate_confidence_interval(precision_scores)
        )

    return results