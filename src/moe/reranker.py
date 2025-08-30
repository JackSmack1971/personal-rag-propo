"""
Two-Stage Reranker Module for Mixture of Experts (MoE) Architecture

This module implements a sophisticated two-stage reranking pipeline that combines
cross-encoder reranking with optional LLM-based reranking for improved retrieval quality.

Author: SPARC Code Implementer
Date: 2025-08-30
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import time
import hashlib
import json

from .config import MoERerankerConfig

logger = logging.getLogger(__name__)

@dataclass
class RerankerResult:
    """Result from reranking operation"""
    original_matches: List[Dict[str, Any]]
    reranked_matches: List[Dict[str, Any]]
    stage1_scores: List[float]
    stage2_scores: Optional[List[float]] = None
    uncertainty_score: float = 0.0
    processing_time: float = 0.0
    timestamp: float = field(default_factory=time.time)

@dataclass
class RerankerPerformance:
    """Tracks reranker performance metrics"""
    total_reranks: int = 0
    stage1_improvements: int = 0
    stage2_improvements: int = 0
    avg_processing_time: float = 0.0
    cache_hit_rate: float = 0.0
    last_updated: float = 0.0

class CrossEncoderReranker:
    """
    Stage 1 reranker using cross-encoder models for precise relevance scoring.
    """

    def __init__(self, config: MoERerankerConfig):
        self.config = config
        self._cross_encoder = None
        self._model_loaded = False

        # Performance tracking
        self.performance = RerankerPerformance()

        logger.info("Initialized CrossEncoderReranker")

    @property
    def cross_encoder(self):
        """Lazy load cross-encoder model"""
        if self._cross_encoder is None:
            try:
                from sentence_transformers import CrossEncoder
                logger.info(f"Loading cross-encoder model: {self.config.cross_encoder_model}")
                self._cross_encoder = CrossEncoder(
                    self.config.cross_encoder_model,
                    max_length=512,  # Limit input length for efficiency
                    trust_remote_code=False
                )
                self._model_loaded = True
                logger.info("Cross-encoder model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load cross-encoder model: {e}")
                raise
        return self._cross_encoder

    def rerank(
        self,
        query: str,
        matches: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """
        Rerank matches using cross-encoder.

        Args:
            query: The search query
            matches: List of matches to rerank
            top_k: Limit number of results to rerank

        Returns:
            Tuple of (reranked_matches, scores)
        """
        if not matches:
            return [], []

        start_time = time.time()

        try:
            # Limit matches for efficiency
            if top_k and len(matches) > top_k:
                matches = matches[:top_k]

            # Prepare query-passage pairs
            query_passage_pairs = []
            for match in matches:
                # Extract text from match
                text = self._extract_text_from_match(match)
                if text:
                    query_passage_pairs.append([query, text])

            if not query_passage_pairs:
                logger.warning("No valid text found in matches for reranking")
                return matches, [0.0] * len(matches)

            # Perform cross-encoder scoring
            logger.debug(f"Cross-encoder scoring {len(query_passage_pairs)} pairs")
            scores = self.cross_encoder.predict(
                query_passage_pairs,
                batch_size=self.config.batch_size,
                show_progress_bar=False
            )

            # Convert to list if needed
            if isinstance(scores, np.ndarray):
                scores = scores.tolist()

            # Sort matches by scores (descending)
            scored_matches = list(zip(matches, scores))
            scored_matches.sort(key=lambda x: x[1], reverse=True)

            reranked_matches = [match for match, _ in scored_matches]
            final_scores = [score for _, score in scored_matches]

            # Update performance metrics
            processing_time = time.time() - start_time
            self._update_performance_metrics(processing_time, len(matches))

            logger.debug(f"Cross-encoder reranking completed in {processing_time:.3f}s")
            return reranked_matches, final_scores

        except Exception as e:
            logger.error(f"Cross-encoder reranking failed: {e}")
            # Return original order with zero scores
            return matches, [0.0] * len(matches)

    def _extract_text_from_match(self, match: Dict[str, Any]) -> Optional[str]:
        """Extract text content from a match dictionary"""
        try:
            # Try different possible text fields
            text_sources = [
                match.get('metadata', {}).get('text', ''),
                match.get('text', ''),
                match.get('content', ''),
                match.get('body', ''),
                match.get('description', '')
            ]

            # Find first non-empty text
            for text in text_sources:
                if text and isinstance(text, str) and text.strip():
                    # Truncate for efficiency
                    return text.strip()[:1000]  # Limit to 1000 chars

            return None

        except Exception as e:
            logger.error(f"Error extracting text from match: {e}")
            return None

    def _update_performance_metrics(self, processing_time: float, num_matches: int) -> None:
        """Update performance tracking metrics"""
        self.performance.total_reranks += 1
        self.performance.last_updated = time.time()

        # Update average processing time
        alpha = 0.1  # Smoothing factor
        self.performance.avg_processing_time = (
            alpha * processing_time +
            (1 - alpha) * self.performance.avg_processing_time
        )

class LLMReranker:
    """
    Stage 2 reranker using LLM for complex reasoning-based reranking.
    Only used when cross-encoder uncertainty is high.
    """

    def __init__(self, config: MoERerankerConfig):
        self.config = config
        self._llm_client = None

    def rerank(
        self,
        query: str,
        matches: List[Dict[str, Any]],
        cross_encoder_scores: List[float]
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """
        Perform LLM-based reranking for complex queries.

        Args:
            query: The search query
            matches: Matches from stage 1 reranking
            cross_encoder_scores: Scores from cross-encoder

        Returns:
            Tuple of (reranked_matches, combined_scores)
        """
        if not matches or len(matches) < 2:
            return matches, cross_encoder_scores

        try:
            # Prepare LLM prompt
            prompt = self._create_reranking_prompt(query, matches)

            # Get LLM response
            llm_response = self._call_llm(prompt)

            # Parse ranking from response
            new_order = self._parse_llm_ranking(llm_response, len(matches))

            if new_order:
                # Reorder matches and scores
                reranked_matches = [matches[i] for i in new_order]
                reranked_scores = [cross_encoder_scores[i] for i in new_order]

                # Combine scores (weighted average)
                combined_scores = []
                for i, (ce_score, new_pos) in enumerate(zip(cross_encoder_scores, new_order)):
                    # Give more weight to LLM ranking for final position
                    position_weight = 1.0 - (new_pos / len(matches))  # Higher for better positions
                    combined_score = 0.7 * ce_score + 0.3 * position_weight
                    combined_scores.append(combined_score)

                logger.debug("LLM reranking completed successfully")
                return reranked_matches, combined_scores
            else:
                logger.warning("Failed to parse LLM ranking, returning original order")
                return matches, cross_encoder_scores

        except Exception as e:
            logger.error(f"LLM reranking failed: {e}")
            return matches, cross_encoder_scores

    def _create_reranking_prompt(self, query: str, matches: List[Dict[str, Any]]) -> str:
        """Create prompt for LLM reranking"""
        prompt = f"""
You are an expert at ranking document passages by relevance to a query.

Query: {query}

Please rank the following passages by how well they answer the query.
Consider both direct relevance and contextual usefulness.

Passages:
"""

        for i, match in enumerate(matches):
            text = self._extract_text_from_match(match) or "No text available"
            # Truncate for token efficiency
            truncated_text = text[:500] + "..." if len(text) > 500 else text
            prompt += f"[{i}] {truncated_text}\n\n"

        prompt += """
Instructions:
1. Rank the passages from most relevant (0) to least relevant
2. Consider both direct answers and supporting context
3. Return only the ranking as comma-separated indices
4. Example: 2,0,1,3 (meaning passage 2 is best, then 0, then 1, then 3)

Ranking:"""

        return prompt

    def _call_llm(self, prompt: str) -> str:
        """Call LLM for reranking decision"""
        # This would integrate with your existing LLM client
        # For now, return a mock response
        logger.warning("LLM reranking not fully implemented - using mock response")

        # Mock response - in practice, this would call your LLM API
        num_passages = prompt.count('[')
        mock_ranking = list(range(num_passages))
        np.random.shuffle(mock_ranking)  # Random for demonstration

        return ','.join(map(str, mock_ranking))

    def _parse_llm_ranking(self, response: str, num_passages: int) -> Optional[List[int]]:
        """Parse ranking indices from LLM response"""
        try:
            # Extract ranking from response
            ranking_text = response.strip()
            if ',' in ranking_text:
                indices = [int(x.strip()) for x in ranking_text.split(',')]
            else:
                # Try space-separated
                indices = [int(x.strip()) for x in ranking_text.split()]

            # Validate ranking
            if len(indices) == num_passages and set(indices) == set(range(num_passages)):
                return indices

            logger.warning(f"Invalid LLM ranking: {indices}")
            return None

        except Exception as e:
            logger.error(f"Error parsing LLM ranking: {e}")
            return None

    def _extract_text_from_match(self, match: Dict[str, Any]) -> Optional[str]:
        """Extract text content from match (reuse from CrossEncoderReranker)"""
        try:
            text_sources = [
                match.get('metadata', {}).get('text', ''),
                match.get('text', ''),
                match.get('content', ''),
                match.get('body', '')
            ]

            for text in text_sources:
                if text and isinstance(text, str) and text.strip():
                    return text.strip()[:1000]

            return None

        except Exception:
            return None

class TwoStageReranker:
    """
    Complete two-stage reranking pipeline combining cross-encoder and LLM reranking.
    """

    def __init__(self, config: MoERerankerConfig):
        self.config = config

        # Initialize reranker components
        self.stage1_reranker = CrossEncoderReranker(config)
        self.stage2_reranker = LLMReranker(config) if config.stage2_enabled else None

        # Caching for performance
        self._cache: Dict[str, RerankerResult] = {}
        self._cache_max_size = config.cache_size

        # Performance tracking
        self.performance = RerankerPerformance()

        logger.info("Initialized TwoStageReranker")

    def rerank(
        self,
        query: str,
        matches: List[Dict[str, Any]],
        return_stage2: bool = True
    ) -> RerankerResult:
        """
        Perform complete two-stage reranking.

        Args:
            query: The search query
            matches: Initial matches to rerank
            return_stage2: Whether to perform stage 2 if conditions met

        Returns:
            RerankerResult with reranked matches and metadata
        """
        start_time = time.time()

        try:
            # Check cache first
            cache_key = self._get_cache_key(query, matches)
            if cache_key in self._cache:
                cached_result = self._cache[cache_key]
                logger.debug("Returning cached reranking result")
                return cached_result

            original_matches = matches.copy()

            # Stage 1: Cross-encoder reranking (always performed if enabled)
            stage1_matches = matches
            stage1_scores = []

            if self.config.stage1_enabled and matches:
                stage1_matches, stage1_scores = self.stage1_reranker.rerank(query, matches)
                logger.debug(f"Stage 1 reranking completed: {len(stage1_matches)} matches")

            # Calculate uncertainty for stage 2 decision
            uncertainty = self._calculate_uncertainty(stage1_scores)

            # Stage 2: LLM reranking (conditional)
            final_matches = stage1_matches
            stage2_scores = None

            if (self.config.stage2_enabled and
                return_stage2 and
                uncertainty >= self.config.uncertainty_threshold and
                len(stage1_matches) >= 3):  # Only for sufficient matches

                logger.debug(f"Performing stage 2 reranking (uncertainty: {uncertainty:.3f})")
                final_matches, stage2_scores = self.stage2_reranker.rerank(
                    query, stage1_matches, stage1_scores
                )

            # Create result
            result = RerankerResult(
                original_matches=original_matches,
                reranked_matches=final_matches,
                stage1_scores=stage1_scores,
                stage2_scores=stage2_scores,
                uncertainty_score=uncertainty,
                processing_time=time.time() - start_time
            )

            # Cache result
            self._cache_result(cache_key, result)

            # Update performance metrics
            self._update_performance_metrics(result)

            logger.info(f"Two-stage reranking completed in {result.processing_time:.3f}s")
            return result

        except Exception as e:
            logger.error(f"Two-stage reranking failed: {e}")
            # Return original matches on failure
            return RerankerResult(
                original_matches=matches,
                reranked_matches=matches,
                stage1_scores=[],
                uncertainty_score=1.0,
                processing_time=time.time() - start_time
            )

    def _calculate_uncertainty(self, scores: List[float]) -> float:
        """Calculate uncertainty score from cross-encoder scores"""
        if not scores or len(scores) < 2:
            return 1.0  # Maximum uncertainty

        try:
            # Calculate coefficient of variation (CV)
            mean_score = sum(scores) / len(scores)
            if mean_score == 0:
                return 1.0

            variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
            std_dev = variance ** 0.5
            cv = std_dev / abs(mean_score)

            # Calculate score range
            score_range = max(scores) - min(scores)

            # Combine metrics for uncertainty
            uncertainty = min(cv * 0.7 + (score_range / max(scores)) * 0.3, 1.0)

            return uncertainty

        except Exception as e:
            logger.error(f"Error calculating uncertainty: {e}")
            return 1.0

    def _get_cache_key(self, query: str, matches: List[Dict[str, Any]]) -> str:
        """Generate cache key for reranking result"""
        # Create hash of query and match IDs
        match_ids = [str(match.get('id', '')) for match in matches]
        content = f"{query}|{','.join(match_ids)}"
        return hashlib.md5(content.encode()).hexdigest()

    def _cache_result(self, key: str, result: RerankerResult) -> None:
        """Cache reranking result"""
        if len(self._cache) >= self._cache_max_size:
            # Remove oldest entry (simple LRU)
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k].timestamp)
            del self._cache[oldest_key]

        self._cache[key] = result

    def _update_performance_metrics(self, result: RerankerResult) -> None:
        """Update performance tracking"""
        self.performance.total_reranks += 1
        self.performance.last_updated = time.time()

        # Update cache hit rate
        total_cache_requests = len(self._cache)
        if total_cache_requests > 0:
            # Estimate hit rate (this is simplified)
            self.performance.cache_hit_rate = 0.1  # Placeholder

        # Track processing time
        alpha = 0.1
        self.performance.avg_processing_time = (
            alpha * result.processing_time +
            (1 - alpha) * self.performance.avg_processing_time
        )

    def clear_cache(self) -> None:
        """Clear reranking cache"""
        self._cache.clear()
        logger.info("Cleared reranking cache")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get reranker performance statistics"""
        cache_size = len(self._cache)

        stats = {
            "total_reranks": self.performance.total_reranks,
            "avg_processing_time": self.performance.avg_processing_time,
            "cache_size": cache_size,
            "cache_hit_rate": self.performance.cache_hit_rate,
            "stage1_enabled": self.config.stage1_enabled,
            "stage2_enabled": self.config.stage2_enabled,
            "uncertainty_threshold": self.config.uncertainty_threshold
        }

        # Add cache efficiency metrics
        if self.performance.total_reranks > 0:
            cache_efficiency = cache_size / self.performance.total_reranks
            stats["cache_efficiency"] = cache_efficiency

        return stats

    def preload_model(self) -> bool:
        """Preload cross-encoder model for faster first inference"""
        try:
            if self.config.stage1_enabled:
                # Trigger lazy loading
                _ = self.stage1_reranker.cross_encoder
                logger.info("Cross-encoder model preloaded successfully")
                return True
        except Exception as e:
            logger.error(f"Failed to preload model: {e}")
            return False

        return False

# Global reranker instance
_reranker_instance: Optional[TwoStageReranker] = None

def get_two_stage_reranker(config: Optional[MoERerankerConfig] = None) -> TwoStageReranker:
    """Get or create global two-stage reranker instance"""
    global _reranker_instance

    if _reranker_instance is None:
        if config is None:
            from .config import get_moe_config
            moe_config = get_moe_config()
            config = moe_config.reranker

        _reranker_instance = TwoStageReranker(config)

    return _reranker_instance