"""
MoE Integration Module - Orchestrates the Complete MoE Pipeline

This module provides the main orchestration layer that integrates all MoE components
into a seamless retrieval-augmented generation pipeline with intelligent routing,
adaptive retrieval, and multi-stage reranking.

Author: SPARC Code Implementer
Date: 2025-08-30
"""

import time
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

from .config import MoEConfig, MoEIntegrationConfig
from .router import ExpertRouter, RoutingDecision, get_expert_router
from .gate import SelectiveGate, GateDecision, RetrievalMatch, get_selective_gate
from .reranker import TwoStageReranker, RerankerResult, get_two_stage_reranker

logger = logging.getLogger(__name__)

@dataclass
class MoEPipelineResult:
    """Complete result from MoE pipeline execution"""
    query: str
    routing_decision: RoutingDecision
    gate_decision: GateDecision
    retrieval_matches: List[Dict[str, Any]]
    reranker_result: Optional[RerankerResult]
    final_answer: Optional[str]
    processing_time: float
    pipeline_stats: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)

@dataclass
class PipelinePerformance:
    """Tracks overall pipeline performance"""
    total_queries: int = 0
    avg_routing_time: float = 0.0
    avg_gate_time: float = 0.0
    avg_retrieval_time: float = 0.0
    avg_reranking_time: float = 0.0
    avg_total_time: float = 0.0
    cache_hit_rate: float = 0.0
    last_updated: float = 0.0

class MoEPipeline:
    """
    Complete MoE pipeline that orchestrates routing, gating, retrieval, and reranking
    for enhanced retrieval-augmented generation.
    """

    def __init__(self, config: MoEConfig):
        self.config = config
        self.integration_config = config.integration

        # Initialize components
        self.router = get_expert_router(config.router) if config.router.enabled else None
        self.gate = get_selective_gate(config.gate) if config.gate.enabled else None
        self.reranker = get_two_stage_reranker(config.reranker) if config.reranker.enabled else None

        # Performance tracking
        self.performance = PipelinePerformance()

        # Cache for query results
        self._query_cache: Dict[str, MoEPipelineResult] = {}
        self._cache_max_size = 1000

        logger.info("Initialized MoE Pipeline")

    def process_query(
        self,
        query: str,
        query_embedding: Optional[np.ndarray] = None,
        retrieval_function: Optional[callable] = None,
        generation_function: Optional[callable] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> MoEPipelineResult:
        """
        Process a query through the complete MoE pipeline.

        Args:
            query: The user's query
            query_embedding: Pre-computed query embedding (optional)
            retrieval_function: Function to perform vector retrieval
            generation_function: Function to generate final answer
            context: Additional context information

        Returns:
            Complete pipeline result
        """
        start_time = time.time()
        pipeline_stats = {
            "stages_executed": [],
            "component_times": {},
            "cache_used": False
        }

        try:
            # Check cache first
            cache_key = self._get_cache_key(query)
            if cache_key in self._query_cache:
                cached_result = self._query_cache[cache_key]
                pipeline_stats["cache_used"] = True
                logger.debug("Returning cached pipeline result")
                return cached_result

            # Stage 1: Expert Routing
            routing_decision = None
            if self.router and self.config.router.enabled:
                stage_start = time.time()
                routing_decision = self.router.route_query(
                    query=query,
                    query_embedding=query_embedding,
                    context=context
                )
                pipeline_stats["component_times"]["routing"] = time.time() - stage_start
                pipeline_stats["stages_executed"].append("routing")
                logger.debug(f"Routing completed: {routing_decision.chosen_experts}")

            # Stage 2: Retrieval Gating
            gate_decision = None
            if self.gate and self.config.gate.enabled:
                stage_start = time.time()
                router_similarities = (
                    routing_decision.routing_scores if routing_decision else {}
                )
                gate_decision = self.gate.should_retrieve_and_k(
                    router_similarities=router_similarities,
                    query_complexity_score=None,  # Will be calculated internally
                    query_embedding=query_embedding,
                    context=context
                )
                pipeline_stats["component_times"]["gating"] = time.time() - stage_start
                pipeline_stats["stages_executed"].append("gating")
                logger.debug(f"Gating completed: retrieve={gate_decision.should_retrieve}, k={gate_decision.optimal_k}")

            # Stage 3: Vector Retrieval
            retrieval_matches = []
            if gate_decision and gate_decision.should_retrieve and retrieval_function:
                stage_start = time.time()

                # Prepare retrieval parameters
                retrieval_params = self._prepare_retrieval_params(
                    routing_decision, gate_decision, context
                )

                # Perform retrieval
                retrieval_matches = self._perform_retrieval(
                    retrieval_function, query, query_embedding, retrieval_params
                )

                pipeline_stats["component_times"]["retrieval"] = time.time() - stage_start
                pipeline_stats["stages_executed"].append("retrieval")
                logger.debug(f"Retrieval completed: {len(retrieval_matches)} matches")

            # Stage 4: Score Filtering (if gate enabled)
            if self.gate and retrieval_matches:
                stage_start = time.time()
                retrieval_matches_objects = [
                    RetrievalMatch(
                        id=match.get("id", ""),
                        score=match.get("score", 0.0),
                        metadata=match.get("metadata", {})
                    )
                    for match in retrieval_matches
                ]

                filtered_matches = self.gate.apply_score_filtering(
                    retrieval_matches_objects, query_embedding
                )

                # Convert back to dict format
                retrieval_matches = [
                    {
                        "id": match.id,
                        "score": match.score,
                        "metadata": match.metadata
                    }
                    for match in filtered_matches
                ]

                pipeline_stats["component_times"]["filtering"] = time.time() - stage_start
                pipeline_stats["stages_executed"].append("filtering")
                logger.debug(f"Filtering completed: {len(retrieval_matches)} matches after filtering")

            # Stage 5: Reranking
            reranker_result = None
            if self.reranker and self.config.reranker.enabled and retrieval_matches:
                stage_start = time.time()
                reranker_result = self.reranker.rerank(
                    query=query,
                    matches=retrieval_matches,
                    return_stage2=self.config.reranker.stage2_enabled
                )

                # Update matches with reranked results
                if reranker_result.reranked_matches:
                    retrieval_matches = reranker_result.reranked_matches

                pipeline_stats["component_times"]["reranking"] = time.time() - stage_start
                pipeline_stats["stages_executed"].append("reranking")
                logger.debug(f"Reranking completed: uncertainty={reranker_result.uncertainty_score:.3f}")

            # Stage 6: Answer Generation
            final_answer = None
            if generation_function and retrieval_matches:
                stage_start = time.time()

                # Prepare context from retrieval matches
                context_text = self._prepare_generation_context(retrieval_matches)

                # Generate answer
                final_answer = generation_function(
                    query=query,
                    context=context_text,
                    retrieval_matches=retrieval_matches
                )

                pipeline_stats["component_times"]["generation"] = time.time() - stage_start
                pipeline_stats["stages_executed"].append("generation")
                logger.debug("Answer generation completed")

            # Create comprehensive result
            result = MoEPipelineResult(
                query=query,
                routing_decision=routing_decision,
                gate_decision=gate_decision,
                retrieval_matches=retrieval_matches,
                reranker_result=reranker_result,
                final_answer=final_answer,
                processing_time=time.time() - start_time,
                pipeline_stats=pipeline_stats
            )

            # Cache result
            self._cache_result(cache_key, result)

            # Update performance metrics
            self._update_performance_metrics(result)

            logger.info(f"MoE pipeline completed in {result.processing_time:.3f}s")
            return result

        except Exception as e:
            logger.error(f"MoE pipeline failed: {e}")

            # Create error result
            error_result = MoEPipelineResult(
                query=query,
                routing_decision=None,
                gate_decision=None,
                retrieval_matches=[],
                reranker_result=None,
                final_answer=None,
                processing_time=time.time() - start_time,
                pipeline_stats={
                    "error": str(e),
                    "stages_executed": pipeline_stats["stages_executed"]
                }
            )

            return error_result

    def _prepare_retrieval_params(
        self,
        routing_decision: Optional[RoutingDecision],
        gate_decision: Optional[GateDecision],
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Prepare parameters for retrieval function"""
        params = {
            "top_k": gate_decision.optimal_k if gate_decision else 10,
        }

        # Add expert filtering if routing was performed
        if routing_decision and routing_decision.chosen_experts:
            params["expert_filter"] = routing_decision.chosen_experts

        # Add any additional context
        if context:
            params.update(context.get("retrieval_params", {}))

        return params

    def _perform_retrieval(
        self,
        retrieval_function: callable,
        query: str,
        query_embedding: Optional[np.ndarray],
        params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Perform vector retrieval with error handling"""
        try:
            # Call the retrieval function
            result = retrieval_function(
                query=query,
                query_embedding=query_embedding,
                **params
            )

            # Handle different result formats
            if isinstance(result, dict):
                matches = result.get("matches", [])
            elif isinstance(result, list):
                matches = result
            else:
                logger.warning(f"Unexpected retrieval result type: {type(result)}")
                matches = []

            return matches

        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []

    def _prepare_generation_context(self, retrieval_matches: List[Dict[str, Any]]) -> str:
        """Prepare context text from retrieval matches for generation"""
        if not retrieval_matches:
            return ""

        context_parts = []

        for i, match in enumerate(retrieval_matches[:5]):  # Limit to top 5
            metadata = match.get("metadata", {})
            text = metadata.get("text", "")
            file_name = metadata.get("file_name", "unknown")
            page = metadata.get("page", "?")

            if text:
                context_parts.append(f"[{i+1}] {file_name}:{page} - {text[:500]}...")

        return "\n\n".join(context_parts)

    def _get_cache_key(self, query: str) -> str:
        """Generate cache key for query"""
        import hashlib
        return hashlib.md5(query.encode()).hexdigest()

    def _cache_result(self, key: str, result: MoEPipelineResult) -> None:
        """Cache pipeline result"""
        if len(self._query_cache) >= self._cache_max_size:
            # Remove oldest entry
            oldest_key = min(
                self._query_cache.keys(),
                key=lambda k: self._query_cache[k].timestamp
            )
            del self._query_cache[oldest_key]

        self._query_cache[key] = result

    def _update_performance_metrics(self, result: MoEPipelineResult) -> None:
        """Update performance tracking"""
        self.performance.total_queries += 1
        self.performance.last_updated = time.time()

        # Update timing averages
        alpha = 0.1  # Smoothing factor

        component_times = result.pipeline_stats.get("component_times", {})

        if "routing" in component_times:
            self.performance.avg_routing_time = (
                alpha * component_times["routing"] +
                (1 - alpha) * self.performance.avg_routing_time
            )

        if "gating" in component_times:
            self.performance.avg_gate_time = (
                alpha * component_times["gating"] +
                (1 - alpha) * self.performance.avg_gate_time
            )

        if "retrieval" in component_times:
            self.performance.avg_retrieval_time = (
                alpha * component_times["retrieval"] +
                (1 - alpha) * self.performance.avg_retrieval_time
            )

        if "reranking" in component_times:
            self.performance.avg_reranking_time = (
                alpha * component_times["reranking"] +
                (1 - alpha) * self.performance.avg_reranking_time
            )

        self.performance.avg_total_time = (
            alpha * result.processing_time +
            (1 - alpha) * self.performance.avg_total_time
        )

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics"""
        cache_size = len(self._query_cache)

        stats = {
            "total_queries": self.performance.total_queries,
            "avg_total_time": self.performance.avg_total_time,
            "avg_routing_time": self.performance.avg_routing_time,
            "avg_gate_time": self.performance.avg_gate_time,
            "avg_retrieval_time": self.performance.avg_retrieval_time,
            "avg_reranking_time": self.performance.avg_reranking_time,
            "cache_size": cache_size,
            "cache_hit_rate": self.performance.cache_hit_rate,
            "components_enabled": {
                "router": self.config.router.enabled,
                "gate": self.config.gate.enabled,
                "reranker": self.config.reranker.enabled
            }
        }

        # Add component-specific stats
        if self.router:
            stats["router_stats"] = self.router.get_routing_stats()

        if self.gate:
            stats["gate_stats"] = self.gate.get_gate_stats()

        if self.reranker:
            stats["reranker_stats"] = self.reranker.get_performance_stats()

        return stats

    def clear_cache(self) -> None:
        """Clear pipeline cache"""
        self._query_cache.clear()
        if self.reranker:
            self.reranker.clear_cache()
        logger.info("Cleared MoE pipeline cache")

    def preload_models(self) -> Dict[str, bool]:
        """Preload all ML models for faster inference"""
        results = {}

        try:
            if self.reranker and self.config.reranker.stage1_enabled:
                results["cross_encoder"] = self.reranker.preload_model()
            else:
                results["cross_encoder"] = True  # Not needed

            logger.info("Model preloading completed")
            return results

        except Exception as e:
            logger.error(f"Model preloading failed: {e}")
            return {"error": str(e)}

    def update_expert_centroids(
        self,
        expert_embeddings: Dict[str, List[np.ndarray]]
    ) -> bool:
        """Update expert centroids for improved routing"""
        try:
            if self.router:
                self.router.update_centroids(expert_embeddings)
                logger.info(f"Updated centroids for {len(expert_embeddings)} experts")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to update centroids: {e}")
            return False

# Global pipeline instance
_pipeline_instance: Optional[MoEPipeline] = None

def get_moe_pipeline(config: Optional[MoEConfig] = None) -> MoEPipeline:
    """Get or create global MoE pipeline instance"""
    global _pipeline_instance

    if _pipeline_instance is None:
        if config is None:
            from .config import get_moe_config
            config = get_moe_config()

        _pipeline_instance = MoEPipeline(config)

    return _pipeline_instance

def process_query_with_moe(
    query: str,
    query_embedding: Optional[np.ndarray] = None,
    retrieval_function: Optional[callable] = None,
    generation_function: Optional[callable] = None,
    config: Optional[MoEConfig] = None
) -> MoEPipelineResult:
    """
    Convenience function to process a query through the MoE pipeline.

    Args:
        query: The user's query
        query_embedding: Pre-computed query embedding
        retrieval_function: Function to perform retrieval
        generation_function: Function to generate answers
        config: MoE configuration (optional)

    Returns:
        Complete pipeline result
    """
    pipeline = get_moe_pipeline(config)
    return pipeline.process_query(
        query=query,
        query_embedding=query_embedding,
        retrieval_function=retrieval_function,
        generation_function=generation_function
    )