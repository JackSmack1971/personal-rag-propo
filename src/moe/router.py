"""
Expert Router Module for Mixture of Experts (MoE) Architecture

This module implements intelligent query routing to specialized expert models
based on query characteristics and historical performance.

Author: SPARC Code Implementer
Date: 2025-08-30
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import time
import json
import os

from .config import MoERouterConfig

logger = logging.getLogger(__name__)

@dataclass
class ExpertCentroid:
    """Represents an expert's centroid in embedding space"""
    expert_id: str
    centroid: np.ndarray
    document_count: int
    last_updated: float
    confidence_score: float = 0.0
    performance_score: float = 0.5
    specialization: str = "general"

@dataclass
class RoutingDecision:
    """Result of expert routing decision"""
    query: str
    chosen_experts: List[str]
    routing_scores: Dict[str, float]
    confidence: float
    reasoning: str
    timestamp: float = field(default_factory=time.time)

@dataclass
class ExpertPerformance:
    """Tracks expert performance metrics"""
    expert_id: str
    total_queries: int = 0
    successful_queries: int = 0
    avg_response_time: float = 0.0
    avg_relevance_score: float = 0.0
    last_used: float = 0.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        return self.successful_queries / max(self.total_queries, 1)

class ExpertRouter:
    """
    Intelligent router that directs queries to the most appropriate experts
    based on query characteristics and expert performance.
    """

    def __init__(self, config: MoERouterConfig):
        self.config = config
        self.centroids: Dict[str, ExpertCentroid] = {}
        self.performance: Dict[str, ExpertPerformance] = {}
        self.routing_history: List[RoutingDecision] = []
        self.last_centroid_refresh = 0

        # Initialize expert performance tracking
        for expert in config.experts:
            self.performance[expert] = ExpertPerformance(expert_id=expert)

        logger.info(f"Initialized ExpertRouter with {len(config.experts)} experts: {config.experts}")

    def route_query(
        self,
        query: str,
        query_embedding: np.ndarray,
        context: Optional[Dict[str, Any]] = None
    ) -> RoutingDecision:
        """
        Route query to most appropriate experts based on embedding similarity
        and performance metrics.

        Args:
            query: The query text
            query_embedding: Query embedding vector
            context: Additional context information

        Returns:
            RoutingDecision with chosen experts and reasoning
        """
        try:
            # Ensure centroids are up to date
            self._refresh_centroids_if_needed()

            # Calculate routing scores
            routing_scores = self._calculate_routing_scores(query_embedding)

            # Select experts based on configuration
            chosen_experts, confidence = self._select_experts(routing_scores)

            # Generate reasoning
            reasoning = self._generate_routing_reasoning(
                chosen_experts, routing_scores, confidence
            )

            # Create routing decision
            decision = RoutingDecision(
                query=query,
                chosen_experts=chosen_experts,
                routing_scores=routing_scores,
                confidence=confidence,
                reasoning=reasoning
            )

            # Record decision for learning
            self.routing_history.append(decision)
            self._update_performance_metrics(chosen_experts)

            logger.debug(f"Routed query to experts {chosen_experts} with confidence {confidence:.3f}")
            return decision

        except Exception as e:
            logger.error(f"Error routing query: {e}")
            # Fallback to all experts
            return RoutingDecision(
                query=query,
                chosen_experts=self.config.experts.copy(),
                routing_scores={},
                confidence=0.0,
                reasoning=f"Fallback routing due to error: {e}"
            )

    def _calculate_routing_scores(self, query_embedding: np.ndarray) -> Dict[str, float]:
        """Calculate routing scores for each expert based on embedding similarity"""
        scores = {}

        if not self.centroids:
            # No centroids available, use uniform scores
            for expert in self.config.experts:
                scores[expert] = 1.0 / len(self.config.experts)
            return scores

        query_norm = self._normalize_vector(query_embedding)

        for expert_id, centroid_info in self.centroids.items():
            if expert_id in self.config.experts:
                # Cosine similarity
                centroid_norm = self._normalize_vector(centroid_info.centroid)
                similarity = float(np.dot(query_norm, centroid_norm))

                # Adjust by performance and confidence
                performance = self.performance.get(expert_id, ExpertPerformance(expert_id))
                performance_factor = performance.success_rate * 0.3 + 0.7  # Weighted adjustment

                # Apply similarity threshold
                if similarity < self.config.similarity_threshold:
                    similarity *= 0.5  # Reduce score for low similarity

                scores[expert_id] = similarity * performance_factor * centroid_info.confidence_score

        # Normalize scores
        total_score = sum(scores.values())
        if total_score > 0:
            scores = {k: v / total_score for k, v in scores.items()}

        return scores

    def _select_experts(self, routing_scores: Dict[str, float]) -> Tuple[List[str], float]:
        """Select top-k experts based on routing scores"""
        if not routing_scores:
            return self.config.experts.copy(), 0.0

        # Sort experts by score
        sorted_experts = sorted(
            routing_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Select top-k experts
        chosen_experts = [expert for expert, _ in sorted_experts[:self.config.top_k_experts]]

        # Calculate confidence as average of chosen experts' scores
        chosen_scores = [routing_scores[expert] for expert in chosen_experts]
        confidence = sum(chosen_scores) / len(chosen_scores) if chosen_scores else 0.0

        return chosen_experts, confidence

    def _generate_routing_reasoning(
        self,
        chosen_experts: List[str],
        routing_scores: Dict[str, float],
        confidence: float
    ) -> str:
        """Generate human-readable reasoning for routing decision"""
        if not routing_scores:
            return "No routing scores available, using all experts"

        reasoning_parts = []

        # Add chosen experts
        reasoning_parts.append(f"Selected experts: {', '.join(chosen_experts)}")

        # Add confidence level
        if confidence > 0.8:
            confidence_desc = "high"
        elif confidence > 0.6:
            confidence_desc = "medium"
        else:
            confidence_desc = "low"
        reasoning_parts.append(f"Confidence: {confidence_desc} ({confidence:.2f})")

        # Add top scores
        top_scores = sorted(routing_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        score_strs = [f"{expert}: {score:.3f}" for expert, score in top_scores]
        reasoning_parts.append(f"Top scores: {', '.join(score_strs)}")

        return " | ".join(reasoning_parts)

    def update_centroids(self, expert_embeddings: Dict[str, List[np.ndarray]]) -> None:
        """
        Update expert centroids based on new document embeddings.

        Args:
            expert_embeddings: Dict mapping expert IDs to lists of embeddings
        """
        current_time = time.time()

        for expert_id, embeddings in expert_embeddings.items():
            if not embeddings or expert_id not in self.config.experts:
                continue

            try:
                # Calculate new centroid
                embeddings_array = np.array(embeddings)
                new_centroid = np.mean(embeddings_array, axis=0)

                # Update or create centroid
                if expert_id in self.centroids:
                    # Exponential moving average for stability
                    old_centroid = self.centroids[expert_id].centroid
                    alpha = 0.1  # Learning rate
                    updated_centroid = alpha * new_centroid + (1 - alpha) * old_centroid

                    self.centroids[expert_id].centroid = updated_centroid
                    self.centroids[expert_id].document_count += len(embeddings)
                else:
                    self.centroids[expert_id] = ExpertCentroid(
                        expert_id=expert_id,
                        centroid=new_centroid,
                        document_count=len(embeddings),
                        last_updated=current_time,
                        confidence_score=0.5  # Initial confidence
                    )

                self.centroids[expert_id].last_updated = current_time

                logger.debug(f"Updated centroid for expert {expert_id} with {len(embeddings)} documents")

            except Exception as e:
                logger.error(f"Error updating centroid for expert {expert_id}: {e}")

        self.last_centroid_refresh = current_time

    def _refresh_centroids_if_needed(self) -> None:
        """Refresh centroids if they're stale"""
        current_time = time.time()
        if current_time - self.last_centroid_refresh > self.config.centroid_refresh_interval:
            # In a real implementation, this would load centroids from persistent storage
            logger.debug("Centroids refresh needed but no persistent storage implemented yet")
            self.last_centroid_refresh = current_time

    def _update_performance_metrics(self, chosen_experts: List[str]) -> None:
        """Update performance metrics for chosen experts"""
        current_time = time.time()

        for expert_id in chosen_experts:
            if expert_id in self.performance:
                perf = self.performance[expert_id]
                perf.total_queries += 1
                perf.last_used = current_time

    def record_expert_performance(
        self,
        expert_id: str,
        response_time: float,
        relevance_score: float,
        success: bool = True
    ) -> None:
        """
        Record performance metrics for an expert after query completion.

        Args:
            expert_id: The expert that processed the query
            response_time: Time taken to respond (seconds)
            relevance_score: Relevance score (0-1)
            success: Whether the query was successful
        """
        if expert_id not in self.performance:
            self.performance[expert_id] = ExpertPerformance(expert_id=expert_id)

        perf = self.performance[expert_id]

        # Update metrics
        perf.total_queries += 1
        if success:
            perf.successful_queries += 1

        # Exponential moving average for response time
        alpha = 0.1
        perf.avg_response_time = alpha * response_time + (1 - alpha) * perf.avg_response_time
        perf.avg_relevance_score = alpha * relevance_score + (1 - alpha) * perf.avg_relevance_score

        logger.debug(f"Updated performance for expert {expert_id}: success_rate={perf.success_rate:.3f}")

    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics for monitoring"""
        total_decisions = len(self.routing_history)
        avg_confidence = sum(d.confidence for d in self.routing_history) / max(total_decisions, 1)

        expert_usage = {}
        for decision in self.routing_history[-100:]:  # Last 100 decisions
            for expert in decision.chosen_experts:
                expert_usage[expert] = expert_usage.get(expert, 0) + 1

        return {
            "total_decisions": total_decisions,
            "avg_confidence": avg_confidence,
            "expert_usage": expert_usage,
            "centroids_available": len(self.centroids),
            "performance_metrics": {
                expert_id: {
                    "success_rate": perf.success_rate,
                    "total_queries": perf.total_queries,
                    "avg_response_time": perf.avg_response_time
                }
                for expert_id, perf in self.performance.items()
            }
        }

    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """Normalize vector with numerical stability"""
        norm = np.linalg.norm(vector)
        return vector / max(norm, 1e-12)

    def save_state(self, filepath: str) -> None:
        """Save router state to file"""
        state = {
            "centroids": {
                expert_id: {
                    "centroid": centroid.centroid.tolist(),
                    "document_count": centroid.document_count,
                    "last_updated": centroid.last_updated,
                    "confidence_score": centroid.confidence_score,
                    "performance_score": centroid.performance_score,
                    "specialization": centroid.specialization
                }
                for expert_id, centroid in self.centroids.items()
            },
            "performance": {
                expert_id: {
                    "total_queries": perf.total_queries,
                    "successful_queries": perf.successful_queries,
                    "avg_response_time": perf.avg_response_time,
                    "avg_relevance_score": perf.avg_relevance_score,
                    "last_used": perf.last_used
                }
                for expert_id, perf in self.performance.items()
            },
            "last_centroid_refresh": self.last_centroid_refresh
        }

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

        logger.info(f"Saved router state to {filepath}")

    def load_state(self, filepath: str) -> None:
        """Load router state from file"""
        if not os.path.exists(filepath):
            logger.info(f"Router state file not found: {filepath}")
            return

        try:
            with open(filepath, 'r') as f:
                state = json.load(f)

            # Load centroids
            for expert_id, centroid_data in state.get("centroids", {}).items():
                self.centroids[expert_id] = ExpertCentroid(
                    expert_id=expert_id,
                    centroid=np.array(centroid_data["centroid"]),
                    document_count=centroid_data["document_count"],
                    last_updated=centroid_data["last_updated"],
                    confidence_score=centroid_data.get("confidence_score", 0.5),
                    performance_score=centroid_data.get("performance_score", 0.5),
                    specialization=centroid_data.get("specialization", "general")
                )

            # Load performance metrics
            for expert_id, perf_data in state.get("performance", {}).items():
                self.performance[expert_id] = ExpertPerformance(
                    expert_id=expert_id,
                    total_queries=perf_data["total_queries"],
                    successful_queries=perf_data["successful_queries"],
                    avg_response_time=perf_data["avg_response_time"],
                    avg_relevance_score=perf_data["avg_relevance_score"],
                    last_used=perf_data["last_used"]
                )

            self.last_centroid_refresh = state.get("last_centroid_refresh", 0)

            logger.info(f"Loaded router state from {filepath}")

        except Exception as e:
            logger.error(f"Error loading router state: {e}")

# Global router instance
_router_instance: Optional[ExpertRouter] = None

def get_expert_router(config: Optional[MoERouterConfig] = None) -> ExpertRouter:
    """Get or create global expert router instance"""
    global _router_instance

    if _router_instance is None:
        if config is None:
            from .config import get_moe_config
            moe_config = get_moe_config()
            config = moe_config.router

        _router_instance = ExpertRouter(config)

        # Try to load saved state
        state_file = os.getenv("MOE_ROUTER_STATE_FILE", "moe_router_state.json")
        if os.path.exists(state_file):
            _router_instance.load_state(state_file)

    return _router_instance

def save_router_state(filepath: Optional[str] = None) -> None:
    """Save global router state"""
    global _router_instance

    if _router_instance is not None:
        filepath = filepath or os.getenv("MOE_ROUTER_STATE_FILE", "moe_router_state.json")
        _router_instance.save_state(filepath)