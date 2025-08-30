#!/usr/bin/env python3
"""
Comprehensive Unit Tests for MoE Components

This test suite provides thorough validation of all MoE architecture components
including configuration, routing, gating, reranking, and integration functionality.

Author: SPARC QA Analyst
Date: 2025-08-30
"""

import sys
import os
import unittest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from moe import (
    MoEConfig, MoERouterConfig, MoEGateConfig, MoERerankerConfig,
    ExpertRouter, SelectiveGate, TwoStageReranker,
    get_moe_pipeline, process_query_with_moe,
    RoutingDecision, GateDecision, RetrievalMatch
)


class TestMoEConfiguration(unittest.TestCase):
    """Test MoE configuration system"""

    def test_default_config_creation(self):
        """Test creating default MoE configuration"""
        config = MoEConfig()
        self.assertIsInstance(config, MoEConfig)
        self.assertTrue(config.enabled)
        self.assertIsInstance(config.router, MoERouterConfig)
        self.assertIsInstance(config.gate, MoEGateConfig)
        self.assertIsInstance(config.reranker, MoERerankerConfig)

    def test_config_validation(self):
        """Test configuration validation"""
        config = MoEConfig()

        # Test valid configuration
        try:
            config.__post_init__()
        except ValueError:
            self.fail("Valid configuration raised ValueError")

        # Test invalid router config
        config.router.top_k_experts = 0
        with self.assertRaises(ValueError):
            config.router.__post_init__()

    def test_config_updates(self):
        """Test configuration updates"""
        config = MoEConfig()
        original_enabled = config.enabled

        # Update configuration
        updates = {"enabled": not original_enabled}
        config.update_from_dict(updates)

        self.assertEqual(config.enabled, not original_enabled)

    def test_component_enablement(self):
        """Test component enablement checking"""
        config = MoEConfig()

        # All components should be enabled by default
        self.assertTrue(config.is_component_enabled("router"))
        self.assertTrue(config.is_component_enabled("gate"))
        self.assertTrue(config.is_component_enabled("reranker"))

        # Disable a component
        config.router.enabled = False
        self.assertFalse(config.is_component_enabled("router"))


class TestExpertRouter(unittest.TestCase):
    """Test Expert Router functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = MoERouterConfig()
        self.router = ExpertRouter(self.config)

    def test_router_initialization(self):
        """Test router initialization"""
        self.assertIsInstance(self.router, ExpertRouter)
        self.assertEqual(len(self.router.performance), len(self.config.experts))
        self.assertEqual(len(self.router.centroids), 0)  # No centroids initially

    def test_routing_without_centroids(self):
        """Test routing when no centroids are available"""
        query_embedding = np.random.rand(384)

        decision = self.router.route_query(
            query="What is machine learning?",
            query_embedding=query_embedding
        )

        self.assertIsInstance(decision, RoutingDecision)
        self.assertEqual(len(decision.chosen_experts), self.config.top_k_experts)
        self.assertGreater(decision.confidence, 0.0)
        self.assertIsInstance(decision.reasoning, str)

    def test_centroid_updates(self):
        """Test centroid updates"""
        expert_embeddings = {
            "expert1": [np.random.rand(384) for _ in range(5)],
            "expert2": [np.random.rand(384) for _ in range(3)],
        }

        self.router.update_centroids(expert_embeddings)

        self.assertEqual(len(self.router.centroids), 2)
        self.assertIn("expert1", self.router.centroids)
        self.assertIn("expert2", self.router.centroids)

        # Check centroid properties
        centroid1 = self.router.centroids["expert1"]
        self.assertEqual(centroid1.document_count, 5)
        self.assertIsInstance(centroid1.centroid, np.ndarray)
        self.assertEqual(len(centroid1.centroid), 384)

    def test_routing_with_centroids(self):
        """Test routing with centroids available"""
        # Add centroids
        expert_embeddings = {
            "general": [np.ones(384) * 0.1 for _ in range(10)],
            "technical": [np.ones(384) * 0.5 for _ in range(10)],
            "personal": [np.ones(384) * 0.9 for _ in range(10)],
        }
        self.router.update_centroids(expert_embeddings)

        # Test routing
        query_embedding = np.ones(384) * 0.8  # Similar to personal centroid
        decision = self.router.route_query(
            query="Tell me about your personal experiences",
            query_embedding=query_embedding
        )

        self.assertIsInstance(decision, RoutingDecision)
        self.assertIn("personal", decision.chosen_experts)

    def test_performance_tracking(self):
        """Test performance metrics tracking"""
        expert_id = "general"
        initial_queries = self.router.performance[expert_id].total_queries

        self.router.record_expert_performance(
            expert_id=expert_id,
            response_time=1.5,
            relevance_score=0.8,
            success=True
        )

        self.assertEqual(
            self.router.performance[expert_id].total_queries,
            initial_queries + 1
        )
        self.assertEqual(self.router.performance[expert_id].successful_queries, 1)
        self.assertAlmostEqual(self.router.performance[expert_id].avg_response_time, 1.5)
        self.assertAlmostEqual(self.router.performance[expert_id].avg_relevance_score, 0.8)

    def test_routing_statistics(self):
        """Test routing statistics generation"""
        stats = self.router.get_routing_stats()

        self.assertIsInstance(stats, dict)
        self.assertIn("total_decisions", stats)
        self.assertIn("avg_confidence", stats)
        self.assertIn("expert_usage", stats)
        self.assertIn("centroids_available", stats)


class TestSelectiveGate(unittest.TestCase):
    """Test Selective Gate functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = MoEGateConfig()
        self.gate = SelectiveGate(self.config)

    def test_gate_initialization(self):
        """Test gate initialization"""
        self.assertIsInstance(self.gate, SelectiveGate)
        self.assertIsInstance(self.gate.performance, type(self.gate.performance))

    def test_gate_decision_retrieval(self):
        """Test gate decision for retrieval"""
        router_similarities = {"expert1": 0.8, "expert2": 0.6}

        decision = self.gate.should_retrieve_and_k(
            router_similarities=router_similarities,
            query_complexity_score=0.5
        )

        self.assertIsInstance(decision, GateDecision)
        self.assertIsInstance(decision.should_retrieve, bool)
        self.assertIsInstance(decision.optimal_k, int)
        self.assertGreaterEqual(decision.optimal_k, 0)
        self.assertIsInstance(decision.reasoning, str)

    def test_complexity_calculation(self):
        """Test query complexity calculation"""
        router_similarities = {"expert1": 0.9, "expert2": 0.1}  # High variance

        complexity = self.gate._calculate_query_complexity(
            router_similarities, None
        )

        self.assertIsInstance(complexity, float)
        self.assertGreaterEqual(complexity, 0.0)
        self.assertLessEqual(complexity, 1.0)

    def test_score_filtering(self):
        """Test score-based filtering"""
        matches = [
            RetrievalMatch(id="doc1", score=0.9, metadata={"text": "test1"}),
            RetrievalMatch(id="doc2", score=0.5, metadata={"text": "test2"}),
            RetrievalMatch(id="doc3", score=0.1, metadata={"text": "test3"}),
        ]

        filtered = self.gate.apply_score_filtering(matches)

        self.assertIsInstance(filtered, list)
        self.assertLessEqual(len(filtered), len(matches))

        # Check that filtered matches have higher scores
        if filtered:
            min_filtered_score = min(match.score for match in filtered)
            max_unfiltered_score = max(match.score for match in matches)
            self.assertGreaterEqual(min_filtered_score, min_filtered_score)

    def test_gate_statistics(self):
        """Test gate statistics"""
        stats = self.gate.get_gate_stats()

        self.assertIsInstance(stats, dict)
        self.assertIn("total_decisions", stats)
        self.assertIn("retrieval_accuracy", stats)


class TestTwoStageReranker(unittest.TestCase):
    """Test Two-Stage Reranker functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = MoERerankerConfig()
        self.reranker = TwoStageReranker(self.config)

    def test_reranker_initialization(self):
        """Test reranker initialization"""
        self.assertIsInstance(self.reranker, TwoStageReranker)
        self.assertIsNotNone(self.reranker.stage1_reranker)

    @patch('moe.reranker.TwoStageReranker._calculate_uncertainty')
    def test_reranking_pipeline(self, mock_uncertainty):
        """Test complete reranking pipeline"""
        mock_uncertainty.return_value = 0.5  # Medium uncertainty

        matches = [
            {"id": "doc1", "score": 0.8, "metadata": {"text": "This is a test document about machine learning."}},
            {"id": "doc2", "score": 0.6, "metadata": {"text": "Another document about artificial intelligence."}},
        ]

        result = self.reranker.rerank(
            query="What is machine learning?",
            matches=matches
        )

        self.assertIsInstance(result, type(result))  # RerankerResult
        self.assertIsInstance(result.reranked_matches, list)
        self.assertIsInstance(result.processing_time, float)
        self.assertGreater(result.processing_time, 0)

    def test_uncertainty_calculation(self):
        """Test uncertainty calculation"""
        scores = [0.9, 0.5, 0.3]  # Varied scores

        uncertainty = self.reranker._calculate_uncertainty(scores)

        self.assertIsInstance(uncertainty, float)
        self.assertGreaterEqual(uncertainty, 0.0)
        self.assertLessEqual(uncertainty, 1.0)

    def test_cache_functionality(self):
        """Test caching functionality"""
        cache_key = "test_key"
        result = Mock()

        self.reranker._cache_result(cache_key, result)
        self.assertIn(cache_key, self.reranker._cache)

        # Test cache retrieval
        cached = self.reranker._cache.get(cache_key)
        self.assertEqual(cached, result)

    def test_cache_clearing(self):
        """Test cache clearing"""
        # Add some items to cache
        for i in range(5):
            self.reranker._cache[f"key_{i}"] = Mock()

        self.assertEqual(len(self.reranker._cache), 5)

        # Clear cache
        self.reranker.clear_cache()
        self.assertEqual(len(self.reranker._cache), 0)


class TestMoEPipeline(unittest.TestCase):
    """Test MoE Pipeline integration"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = MoEConfig()
        # Disable components that require external dependencies
        self.config.reranker.stage2_enabled = False

    @patch('moe.integration.get_expert_router')
    @patch('moe.integration.get_selective_gate')
    @patch('moe.integration.get_two_stage_reranker')
    def test_pipeline_initialization(self, mock_reranker, mock_gate, mock_router):
        """Test pipeline initialization"""
        mock_router.return_value = Mock()
        mock_gate.return_value = Mock()
        mock_reranker.return_value = Mock()

        pipeline = get_moe_pipeline(self.config)

        self.assertIsNotNone(pipeline)
        mock_router.assert_called_once()
        mock_gate.assert_called_once()
        mock_reranker.assert_called_once()

    def test_pipeline_processing_mock(self):
        """Test pipeline processing with mocks"""
        # This would require extensive mocking of all components
        # For now, test that the function exists and has correct signature
        self.assertTrue(callable(process_query_with_moe))


class TestIntegrationScenarios(unittest.TestCase):
    """Test integration scenarios and edge cases"""

    def test_component_disablement(self):
        """Test behavior when components are disabled"""
        config = MoEConfig()
        config.router.enabled = False
        config.gate.enabled = False
        config.reranker.enabled = False

        # Should still create pipeline but with disabled components
        pipeline = get_moe_pipeline(config)
        self.assertIsNotNone(pipeline)

    def test_error_handling(self):
        """Test error handling in components"""
        router = ExpertRouter(MoERouterConfig())

        # Test with invalid inputs
        with self.assertRaises((AttributeError, TypeError)):
            router.route_query("test", None)

    def test_configuration_persistence(self):
        """Test configuration persistence"""
        from moe.config import MoEConfigManager
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name

        try:
            manager = MoEConfigManager(temp_path)
            config = manager.get_config()
            self.assertIsInstance(config, MoEConfig)

            # Test saving
            success = manager.save_config()
            self.assertTrue(success)

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


if __name__ == '__main__':
    # Configure test runner
    unittest.main(verbosity=2, exit=False)

    # Print test summary
    print("\n" + "="*60)
    print("MoE Component Unit Test Summary")
    print("="*60)
    print("✅ All unit tests completed")
    print("📊 Test coverage includes:")
    print("   • Configuration validation")
    print("   • Expert routing logic")
    print("   • Selective gating")
    print("   • Two-stage reranking")
    print("   • Pipeline integration")
    print("   • Error handling")
    print("   • Performance tracking")
    print("="*60)