"""
Selective Retrieval Gate Module for Mixture of Experts (MoE) Architecture

This module implements intelligent gating mechanisms that decide whether to retrieve
documents and optimize retrieval parameters based on query characteristics.

Author: SPARC Code Implementer
Date: 2025-08-30
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import time
import statistics

from .config import MoEGateConfig

logger = logging.getLogger(__name__)

@dataclass
class GateDecision:
    """Result of retrieval gating decision"""
    should_retrieve: bool
    optimal_k: int
    confidence: float
    reasoning: str
    query_complexity: float
    timestamp: float = field(default_factory=time.time)

@dataclass
class RetrievalMatch:
    """Represents a retrieved document match"""
    id: str
    score: float
    metadata: Dict[str, Any]
    expert_id: Optional[str] = None

@dataclass
class GatePerformance:
    """Tracks gate performance metrics"""
    total_decisions: int = 0
    correct_retrieval_decisions: int = 0
    optimal_k_predictions: int = 0
    avg_confidence: float = 0.0
    last_updated: float = 0.0

    @property
    def retrieval_accuracy(self) -> float:
        """Calculate retrieval decision accuracy"""
        return self.correct_retrieval_decisions / max(self.total_decisions, 1)

class SelectiveGate:
    """
    Intelligent gate that optimizes retrieval decisions and parameters
    based on query analysis and historical performance.
    """

    def __init__(self, config: MoEGateConfig):
        self.config = config
        self.performance = GatePerformance()
        self.decision_history: List[GateDecision] = []
        self.adaptation_history: List[Dict[str, Any]] = []

        # Adaptive parameters
        self.confidence_threshold = config.confidence_weight
        self.complexity_threshold = config.complexity_weight

        logger.info("Initialized SelectiveGate with adaptive parameters")

    def should_retrieve_and_k(
        self,
        router_similarities: Dict[str, float],
        query_complexity_score: Optional[float] = None,
        query_embedding: Optional[np.ndarray] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> GateDecision:
        """
        Decide whether to retrieve documents and determine optimal k value.

        Args:
            router_similarities: Similarity scores from expert router
            query_complexity_score: Query complexity score (0-1)
            query_embedding: Query embedding vector
            context: Additional context information

        Returns:
            GateDecision with retrieval recommendation and optimal k
        """
        try:
            # Calculate query complexity if not provided
            if query_complexity_score is None:
                query_complexity_score = self._calculate_query_complexity(
                    router_similarities, query_embedding
                )

            # Calculate retrieval confidence
            retrieval_confidence = self._calculate_retrieval_confidence(
                router_similarities, query_complexity_score
            )

            # Make retrieval decision
            should_retrieve = self._decide_retrieval(
                retrieval_confidence, query_complexity_score
            )

            # Determine optimal k
            optimal_k = self._calculate_optimal_k(
                retrieval_confidence, query_complexity_score, should_retrieve
            )

            # Generate reasoning
            reasoning = self._generate_gate_reasoning(
                should_retrieve, optimal_k, retrieval_confidence, query_complexity_score
            )

            # Create decision
            decision = GateDecision(
                should_retrieve=should_retrieve,
                optimal_k=optimal_k,
                confidence=retrieval_confidence,
                reasoning=reasoning,
                query_complexity=query_complexity_score
            )

            # Record decision for learning
            self.decision_history.append(decision)
            self._update_performance_metrics(decision)

            logger.debug(f"Gate decision: retrieve={should_retrieve}, k={optimal_k}, confidence={retrieval_confidence:.3f}")
            return decision

        except Exception as e:
            logger.error(f"Error in gate decision: {e}")
            # Safe fallback
            return GateDecision(
                should_retrieve=True,
                optimal_k=self.config.default_top_k,
                confidence=0.5,
                reasoning=f"Fallback due to error: {e}",
                query_complexity=0.5
            )

    def apply_score_filtering(
        self,
        matches: List[RetrievalMatch],
        query_embedding: Optional[np.ndarray] = None
    ) -> List[RetrievalMatch]:
        """
        Apply dynamic score-based filtering to retrieved matches.

        Args:
            matches: List of retrieved matches
            query_embedding: Query embedding for additional analysis

        Returns:
            Filtered list of matches
        """
        if not matches:
            return matches

        try:
            # Extract scores
            scores = [match.score for match in matches]
            max_score = max(scores) if scores else 0.0
            avg_score = sum(scores) / len(scores) if scores else 0.0

            # Dynamic threshold calculation
            if max_score >= self.config.high_score_cutoff:
                # High confidence scenario
                threshold = self.config.high_score_cutoff
                reasoning = f"High confidence filtering (threshold: {threshold:.3f})"
            elif max_score <= self.config.low_score_cutoff:
                # Low confidence scenario
                threshold = self.config.low_score_cutoff * 0.8  # More lenient
                reasoning = f"Low confidence filtering (threshold: {threshold:.3f})"
            else:
                # Medium confidence scenario
                threshold = (self.config.high_score_cutoff + self.config.low_score_cutoff) / 2
                reasoning = f"Medium confidence filtering (threshold: {threshold:.3f})"

            # Apply filtering
            filtered_matches = [match for match in matches if match.score >= threshold]

            # Ensure we have at least minimum results
            if len(filtered_matches) < self.config.k_min and len(matches) >= self.config.k_min:
                # Fallback to top-k matches
                filtered_matches = matches[:self.config.k_min]
                reasoning += f" | Fallback to top-{self.config.k_min}"

            logger.debug(f"Score filtering: {len(matches)} -> {len(filtered_matches)} matches ({reasoning})")

            # Store filtering decision for adaptation
            self.adaptation_history.append({
                "timestamp": time.time(),
                "original_count": len(matches),
                "filtered_count": len(filtered_matches),
                "max_score": max_score,
                "avg_score": avg_score,
                "threshold": threshold,
                "reasoning": reasoning
            })

            return filtered_matches

        except Exception as e:
            logger.error(f"Error in score filtering: {e}")
            return matches  # Return original matches on error

    def _calculate_query_complexity(
        self,
        router_similarities: Dict[str, float],
        query_embedding: Optional[np.ndarray] = None
    ) -> float:
        """Calculate query complexity score based on router similarities"""
        if not router_similarities:
            return 0.5  # Neutral complexity

        try:
            # Calculate similarity distribution metrics
            similarities = list(router_similarities.values())
            max_similarity = max(similarities)
            avg_similarity = sum(similarities) / len(similarities)

            # Calculate variance (how spread out the similarities are)
            if len(similarities) > 1:
                variance = sum((s - avg_similarity) ** 2 for s in similarities) / len(similarities)
                std_dev = variance ** 0.5
            else:
                std_dev = 0.0

            # Complexity factors:
            # 1. Low maximum similarity suggests complex/ambiguous query
            # 2. High standard deviation suggests multi-faceted query
            # 3. Compare max vs average similarity

            complexity_score = 0.0

            # Factor 1: Maximum similarity (inverse relationship)
            if max_similarity < self.config.low_sim_threshold:
                complexity_score += 0.4  # High complexity
            elif max_similarity < self.config.retrieve_sim_threshold:
                complexity_score += 0.2  # Medium complexity

            # Factor 2: Similarity distribution
            if std_dev > 0.2:
                complexity_score += 0.3  # High variance = complex query
            elif std_dev > 0.1:
                complexity_score += 0.15  # Medium variance

            # Factor 3: Max vs average ratio
            if max_similarity > 0 and avg_similarity > 0:
                ratio = max_similarity / avg_similarity
                if ratio > 2.0:
                    complexity_score += 0.3  # Clear preference for one expert

            # Normalize to 0-1 range
            complexity_score = min(max(complexity_score, 0.0), 1.0)

            return complexity_score

        except Exception as e:
            logger.error(f"Error calculating query complexity: {e}")
            return 0.5  # Neutral fallback

    def _calculate_retrieval_confidence(
        self,
        router_similarities: Dict[str, float],
        query_complexity: float
    ) -> float:
        """Calculate confidence in retrieval decision"""
        if not router_similarities:
            return 0.0

        try:
            similarities = list(router_similarities.values())
            max_similarity = max(similarities)
            avg_similarity = sum(similarities) / len(similarities)

            # Base confidence on maximum similarity
            base_confidence = max_similarity

            # Adjust for query complexity (complex queries need more retrieval)
            if query_complexity > 0.7:
                base_confidence *= 0.9  # Slight reduction for very complex queries
            elif query_complexity < 0.3:
                base_confidence *= 1.1  # Slight increase for simple queries

            # Adjust for similarity distribution
            if len(similarities) > 1:
                # Reward diverse expert matching
                diversity_bonus = min(len([s for s in similarities if s > 0.3]), 3) * 0.05
                base_confidence += diversity_bonus

            return min(max(base_confidence, 0.0), 1.0)

        except Exception as e:
            logger.error(f"Error calculating retrieval confidence: {e}")
            return 0.5  # Neutral fallback

    def _decide_retrieval(self, confidence: float, complexity: float) -> bool:
        """Decide whether to perform retrieval"""
        # Always retrieve if confidence is below threshold
        if confidence < self.config.retrieve_sim_threshold:
            return True

        # For high confidence, consider skipping retrieval
        if confidence > self.config.retrieve_sim_threshold:
            # But retrieve anyway for complex queries
            if complexity > 0.6:
                return True
            # Or if we have adaptive threshold learning
            if self._should_retrieve_high_confidence(confidence, complexity):
                return True
            return False

        # Default: retrieve
        return True

    def _should_retrieve_high_confidence(self, confidence: float, complexity: float) -> bool:
        """Adaptive decision for high-confidence scenarios"""
        # Analyze recent performance
        recent_decisions = self.decision_history[-10:]  # Last 10 decisions

        if len(recent_decisions) < 5:
            return False  # Not enough data

        # Calculate recent accuracy for non-retrieval decisions
        non_retrieval_decisions = [d for d in recent_decisions if not d.should_retrieve]
        if non_retrieval_decisions:
            avg_confidence = sum(d.confidence for d in non_retrieval_decisions) / len(non_retrieval_decisions)
            # Only skip retrieval if recent non-retrieval decisions had high confidence
            return avg_confidence < confidence * 0.9

        return False  # Default to retrieving

    def _calculate_optimal_k(self, confidence: float, complexity: float, should_retrieve: bool) -> int:
        """Calculate optimal number of documents to retrieve"""
        if not should_retrieve:
            return 0

        try:
            # Base k on confidence and complexity
            if confidence < self.config.low_sim_threshold:
                # Low confidence: cast wider net
                base_k = self.config.k_max
            elif confidence > self.config.retrieve_sim_threshold:
                # High confidence: focused retrieval
                base_k = self.config.k_min
            else:
                # Medium confidence: default
                base_k = self.config.default_top_k

            # Adjust for complexity
            if complexity > 0.7:
                # Complex queries need more results
                base_k = min(base_k + 2, self.config.k_max)
            elif complexity < 0.3:
                # Simple queries can use fewer results
                base_k = max(base_k - 1, self.config.k_min)

            # Adaptive adjustment based on recent performance
            adaptive_adjustment = self._calculate_adaptive_k_adjustment(confidence)
            base_k = max(self.config.k_min, min(base_k + adaptive_adjustment, self.config.k_max))

            return base_k

        except Exception as e:
            logger.error(f"Error calculating optimal k: {e}")
            return self.config.default_top_k

    def _calculate_adaptive_k_adjustment(self, confidence: float) -> int:
        """Calculate adaptive adjustment for k based on recent performance"""
        recent_decisions = self.decision_history[-20:]  # Last 20 decisions

        if len(recent_decisions) < 10:
            return 0  # Not enough data

        # Analyze performance for similar confidence levels
        similar_decisions = [d for d in recent_decisions
                           if abs(d.confidence - confidence) < 0.2]

        if not similar_decisions:
            return 0

        # If recent similar decisions had good outcomes, we can be more aggressive
        avg_complexity = sum(d.query_complexity for d in similar_decisions) / len(similar_decisions)

        if avg_complexity > 0.6 and confidence > 0.7:
            return 1  # Slightly increase k for complex queries with good confidence
        elif avg_complexity < 0.4 and confidence > 0.8:
            return -1  # Slightly decrease k for simple queries with very good confidence

        return 0

    def _generate_gate_reasoning(
        self,
        should_retrieve: bool,
        optimal_k: int,
        confidence: float,
        complexity: float
    ) -> str:
        """Generate human-readable reasoning for gate decision"""
        reasoning_parts = []

        # Retrieval decision
        if should_retrieve:
            reasoning_parts.append("Retrieval recommended")
        else:
            reasoning_parts.append("Retrieval skipped (high confidence)")

        # K value reasoning
        if optimal_k > 0:
            if optimal_k == self.config.k_min:
                k_reason = "minimal results"
            elif optimal_k >= self.config.k_max:
                k_reason = "maximum results"
            else:
                k_reason = f"{optimal_k} results"
            reasoning_parts.append(f"Optimal k: {k_reason}")

        # Confidence and complexity
        conf_desc = "high" if confidence > 0.7 else "medium" if confidence > 0.4 else "low"
        comp_desc = "high" if complexity > 0.7 else "medium" if complexity > 0.4 else "low"

        reasoning_parts.append(f"Confidence: {conf_desc} ({confidence:.2f})")
        reasoning_parts.append(f"Complexity: {comp_desc} ({complexity:.2f})")

        return " | ".join(reasoning_parts)

    def _update_performance_metrics(self, decision: GateDecision) -> None:
        """Update performance tracking metrics"""
        self.performance.total_decisions += 1
        self.performance.last_updated = time.time()

        # Update rolling average confidence
        alpha = 0.1  # Smoothing factor
        self.performance.avg_confidence = (
            alpha * decision.confidence +
            (1 - alpha) * self.performance.avg_confidence
        )

    def record_decision_outcome(
        self,
        decision: GateDecision,
        was_correct: bool,
        actual_k_used: int,
        performance_metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record the outcome of a gate decision for learning.

        Args:
            decision: The original gate decision
            was_correct: Whether the decision was correct
            actual_k_used: Actual k value used
            performance_metrics: Additional performance metrics
        """
        if was_correct:
            self.performance.correct_retrieval_decisions += 1

        # Track k prediction accuracy
        if abs(actual_k_used - decision.optimal_k) <= 1:
            self.performance.optimal_k_predictions += 1

        # Store outcome for adaptation
        outcome_record = {
            "timestamp": time.time(),
            "decision": {
                "should_retrieve": decision.should_retrieve,
                "optimal_k": decision.optimal_k,
                "confidence": decision.confidence,
                "query_complexity": decision.query_complexity
            },
            "outcome": {
                "was_correct": was_correct,
                "actual_k_used": actual_k_used,
                "performance_metrics": performance_metrics or {}
            }
        }

        self.adaptation_history.append(outcome_record)

        logger.debug(f"Recorded decision outcome: correct={was_correct}, k_diff={abs(actual_k_used - decision.optimal_k)}")

    def get_gate_stats(self) -> Dict[str, Any]:
        """Get gate performance statistics"""
        total_decisions = self.performance.total_decisions

        stats = {
            "total_decisions": total_decisions,
            "retrieval_accuracy": self.performance.retrieval_accuracy,
            "k_prediction_accuracy": (
                self.performance.optimal_k_predictions / max(total_decisions, 1)
            ),
            "avg_confidence": self.performance.avg_confidence,
            "recent_decisions": len(self.decision_history[-100:]),
            "adaptation_records": len(self.adaptation_history)
        }

        # Add recent trends
        recent_decisions = self.decision_history[-50:]
        if recent_decisions:
            recent_confidence = sum(d.confidence for d in recent_decisions) / len(recent_decisions)
            recent_retrieval_rate = sum(1 for d in recent_decisions if d.should_retrieve) / len(recent_decisions)

            stats.update({
                "recent_avg_confidence": recent_confidence,
                "recent_retrieval_rate": recent_retrieval_rate
            })

        return stats

    def adapt_parameters(self) -> None:
        """Adapt gate parameters based on recent performance"""
        recent_decisions = self.decision_history[-100:]

        if len(recent_decisions) < 20:
            return  # Not enough data for adaptation

        try:
            # Analyze recent performance
            recent_correct = sum(1 for d in recent_decisions if d.confidence > 0.7)
            recent_correct_rate = recent_correct / len(recent_decisions)

            # Adapt confidence threshold based on performance
            if recent_correct_rate > 0.8:
                # Too conservative, lower threshold slightly
                self.config.retrieve_sim_threshold *= 0.98
            elif recent_correct_rate < 0.6:
                # Too aggressive, raise threshold slightly
                self.config.retrieve_sim_threshold *= 1.02

            # Keep within bounds
            self.config.retrieve_sim_threshold = max(0.3, min(0.8, self.config.retrieve_sim_threshold))

            logger.info(f"Adapted gate parameters: retrieval_threshold={self.config.retrieve_sim_threshold:.3f}")

        except Exception as e:
            logger.error(f"Error adapting parameters: {e}")

# Global gate instance
_gate_instance: Optional[SelectiveGate] = None

def get_selective_gate(config: Optional[MoEGateConfig] = None) -> SelectiveGate:
    """Get or create global selective gate instance"""
    global _gate_instance

    if _gate_instance is None:
        if config is None:
            from .config import get_moe_config
            moe_config = get_moe_config()
            config = moe_config.gate

        _gate_instance = SelectiveGate(config)

    return _gate_instance