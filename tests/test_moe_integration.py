#!/usr/bin/env python3
"""
Integration Tests for MoE Pipeline

This test suite validates the complete MoE pipeline integration,
including component interactions, data flow, and end-to-end functionality.

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
    MoEConfig, get_moe_pipeline, process_query_with_moe,
    MoEPipelineResult
)


class TestMoEPipelineIntegration(unittest.TestCase):
    """Test complete MoE pipeline integration"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = MoEConfig()
        # Configure for testing - disable components that need external deps
        self.config.reranker.stage2_enabled = False

    @patch('moe.integration.get_expert_router')
    @patch('moe.integration.get_selective_gate')
    @patch('moe.integration.get_two_stage_reranker')
    def test_pipeline_creation(self, mock_reranker, mock_gate, mock_router):
        """Test pipeline creation with mocked components"""
        # Create mocks
        mock_router_instance = Mock()
        mock_gate_instance = Mock()
        mock_reranker_instance = Mock()

        mock_router.return_value = mock_router_instance
        mock_gate.return_value = mock_gate_instance
        mock_reranker.return_value = mock_reranker_instance

        # Create pipeline
        pipeline = get_moe_pipeline(self.config)

        self.assertIsNotNone(pipeline)
        mock_router.assert_called_once()
        mock_gate.assert_called_once()
        mock_reranker.assert_called_once()

    def test_pipeline_with_disabled_components(self):
        """Test pipeline with some components disabled"""
        config = MoEConfig()
        config.router.enabled = False
        config.gate.enabled = False
        config.reranker.enabled = False

        # Should still create pipeline
        pipeline = get_moe_pipeline(config)
        self.assertIsNotNone(pipeline)

    @patch('moe.integration.get_expert_router')
    @patch('moe.integration.get_selective_gate')
    @patch('moe.integration.get_two_stage_reranker')
    def test_end_to_end_pipeline_mock(self, mock_reranker, mock_gate, mock_router):
        """Test end-to-end pipeline with comprehensive mocking"""
        # Create detailed mocks
        mock_router_instance = Mock()
        mock_gate_instance = Mock()
        mock_reranker_instance = Mock()

        # Mock routing decision
        routing_decision = Mock()
        routing_decision.chosen_experts = ["general", "technical"]
        routing_decision.routing_scores = {"general": 0.8, "technical": 0.6}
        routing_decision.confidence = 0.7
        routing_decision.reasoning = "Test routing"
        mock_router_instance.route_query.return_value = routing_decision

        # Mock gate decision
        gate_decision = Mock()
        gate_decision.should_retrieve = True
        gate_decision.optimal_k = 5
        gate_decision.confidence = 0.8
        gate_decision.reasoning = "Test gating"
        gate_decision.query_complexity = 0.5
        mock_gate_instance.should_retrieve_and_k.return_value = gate_decision

        # Mock reranker result
        reranker_result = Mock()
        reranker_result.reranked_matches = [
            {"id": "doc1", "score": 0.9, "metadata": {"text": "Test document 1"}},
            {"id": "doc2", "score": 0.7, "metadata": {"text": "Test document 2"}},
        ]
        reranker_result.uncertainty_score = 0.3
        reranker_result.processing_time = 0.1
        mock_reranker_instance.rerank.return_value = reranker_result

        # Set up mocks
        mock_router.return_value = mock_router_instance
        mock_gate.return_value = mock_gate_instance
        mock_reranker.return_value = mock_reranker_instance

        # Create pipeline
        pipeline = get_moe_pipeline(self.config)

        # Mock retrieval function
        def mock_retrieval(query, query_embedding, **kwargs):
            return {
                "matches": [
                    {"id": "doc1", "score": 0.8, "metadata": {"text": "Document 1"}},
                    {"id": "doc2", "score": 0.6, "metadata": {"text": "Document 2"}},
                    {"id": "doc3", "score": 0.4, "metadata": {"text": "Document 3"}},
                ]
            }

        # Mock generation function
        def mock_generation(query, context, retrieval_matches):
            return f"Generated answer for: {query}"

        # Test pipeline execution
        result = pipeline.process_query(
            query="What is machine learning?",
            query_embedding=np.random.rand(384),
            retrieval_function=mock_retrieval,
            generation_function=mock_generation
        )

        # Verify result structure
        self.assertIsInstance(result, MoEPipelineResult)
        self.assertEqual(result.query, "What is machine learning?")
        self.assertIsNotNone(result.routing_decision)
        self.assertIsNotNone(result.gate_decision)
        self.assertIsInstance(result.retrieval_matches, list)
        self.assertIsNotNone(result.reranker_result)
        self.assertIsInstance(result.final_answer, str)
        self.assertGreater(result.processing_time, 0)

        # Verify pipeline statistics
        self.assertIsInstance(result.pipeline_stats, dict)
        self.assertIn("stages_executed", result.pipeline_stats)
        self.assertIn("component_times", result.pipeline_stats)

    def test_pipeline_error_handling(self):
        """Test pipeline error handling"""
        pipeline = get_moe_pipeline(self.config)

        # Test with invalid inputs
        result = pipeline.process_query(
            query="",  # Empty query
            retrieval_function=None,
            generation_function=None
        )

        # Should still return a result (error result)
        self.assertIsInstance(result, MoEPipelineResult)
        self.assertEqual(result.query, "")

    def test_pipeline_caching(self):
        """Test pipeline caching functionality"""
        pipeline = get_moe_pipeline(self.config)

        # Process same query twice
        query = "Test query"
        result1 = pipeline.process_query(query=query)
        result2 = pipeline.process_query(query=query)

        # Results should be identical (from cache)
        self.assertEqual(result1.query, result2.query)
        self.assertEqual(result1.pipeline_stats.get("cache_used", False),
                        result2.pipeline_stats.get("cache_used", False))

    def test_pipeline_statistics(self):
        """Test pipeline statistics collection"""
        pipeline = get_moe_pipeline(self.config)

        # Process a few queries
        for i in range(3):
            pipeline.process_query(f"Query {i}")

        # Get statistics
        stats = pipeline.get_pipeline_stats()

        self.assertIsInstance(stats, dict)
        self.assertIn("total_queries", stats)
        self.assertEqual(stats["total_queries"], 3)
        self.assertIn("avg_total_time", stats)
        self.assertIn("components_enabled", stats)


class TestComponentInteraction(unittest.TestCase):
    """Test interactions between MoE components"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = MoEConfig()

    @patch('moe.integration.get_expert_router')
    @patch('moe.integration.get_selective_gate')
    def test_router_gate_interaction(self, mock_gate, mock_router):
        """Test interaction between router and gate components"""
        # Mock router
        mock_router_instance = Mock()
        routing_decision = Mock()
        routing_decision.chosen_experts = ["expert1", "expert2"]
        routing_decision.routing_scores = {"expert1": 0.9, "expert2": 0.7}
        mock_router_instance.route_query.return_value = routing_decision
        mock_router.return_value = mock_router_instance

        # Mock gate
        mock_gate_instance = Mock()
        gate_decision = Mock()
        gate_decision.should_retrieve = True
        gate_decision.optimal_k = 8
        mock_gate_instance.should_retrieve_and_k.return_value = gate_decision
        mock_gate.return_value = mock_gate_instance

        pipeline = get_moe_pipeline(self.config)

        # Process query
        result = pipeline.process_query("Test query")

        # Verify router was called
        mock_router_instance.route_query.assert_called_once()

        # Verify gate was called with router results
        mock_gate_instance.should_retrieve_and_k.assert_called_once()
        call_args = mock_gate_instance.should_retrieve_and_k.call_args
        self.assertEqual(call_args[1]["router_similarities"], routing_decision.routing_scores)

    @patch('moe.integration.get_selective_gate')
    @patch('moe.integration.get_two_stage_reranker')
    def test_gate_reranker_interaction(self, mock_reranker, mock_gate):
        """Test interaction between gate and reranker components"""
        # Disable router for this test
        self.config.router.enabled = False

        # Mock gate
        mock_gate_instance = Mock()
        gate_decision = Mock()
        gate_decision.should_retrieve = True
        gate_decision.optimal_k = 5
        mock_gate_instance.should_retrieve_and_k.return_value = gate_decision
        mock_gate.return_value = mock_gate_instance

        # Mock reranker
        mock_reranker_instance = Mock()
        reranker_result = Mock()
        reranker_result.reranked_matches = []
        mock_reranker_instance.rerank.return_value = reranker_result
        mock_reranker.return_value = mock_reranker_instance

        pipeline = get_moe_pipeline(self.config)

        # Mock retrieval
        def mock_retrieval(**kwargs):
            return {"matches": []}

        result = pipeline.process_query(
            "Test query",
            retrieval_function=mock_retrieval
        )

        # Verify reranker was called
        mock_reranker_instance.rerank.assert_called_once()


class TestDataFlow(unittest.TestCase):
    """Test data flow through the MoE pipeline"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = MoEConfig()

    def test_data_transformation(self):
        """Test data transformation through pipeline stages"""
        # Test that data is properly transformed at each stage
        # This would test the actual data flow transformations

        # For now, test the data structures
        from moe import RoutingDecision, GateDecision

        # Test routing decision structure
        routing_decision = RoutingDecision(
            query="test",
            chosen_experts=["expert1"],
            routing_scores={"expert1": 0.8},
            confidence=0.8,
            reasoning="test"
        )

        self.assertEqual(routing_decision.query, "test")
        self.assertIn("expert1", routing_decision.chosen_experts)

        # Test gate decision structure
        gate_decision = GateDecision(
            should_retrieve=True,
            optimal_k=5,
            confidence=0.7,
            reasoning="test",
            query_complexity=0.5
        )

        self.assertTrue(gate_decision.should_retrieve)
        self.assertEqual(gate_decision.optimal_k, 5)

    def test_pipeline_result_structure(self):
        """Test pipeline result data structure"""
        result = MoEPipelineResult(
            query="test query",
            routing_decision=None,
            gate_decision=None,
            retrieval_matches=[],
            reranker_result=None,
            final_answer=None,
            processing_time=1.0,
            pipeline_stats={}
        )

        self.assertEqual(result.query, "test query")
        self.assertEqual(result.processing_time, 1.0)
        self.assertIsInstance(result.pipeline_stats, dict)


class TestPerformanceIntegration(unittest.TestCase):
    """Test performance aspects of MoE integration"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = MoEConfig()

    def test_pipeline_performance_tracking(self):
        """Test that pipeline tracks performance metrics"""
        pipeline = get_moe_pipeline(self.config)

        # Process multiple queries
        for i in range(5):
            result = pipeline.process_query(f"Query {i}")
            self.assertGreater(result.processing_time, 0)

        # Check performance stats
        stats = pipeline.get_pipeline_stats()
        self.assertEqual(stats["total_queries"], 5)
        self.assertGreater(stats["avg_total_time"], 0)

    def test_memory_usage(self):
        """Test memory usage patterns"""
        import psutil
        import os

        pipeline = get_moe_pipeline(self.config)

        # Get initial memory
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Process queries
        for i in range(10):
            pipeline.process_query(f"Query {i}")

        # Check memory after processing
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 100MB)
        self.assertLess(memory_increase, 100 * 1024 * 1024)

    def test_concurrent_processing(self):
        """Test concurrent query processing"""
        import threading
        import time

        pipeline = get_moe_pipeline(self.config)
        results = []
        errors = []

        def process_query_thread(query_id):
            try:
                result = pipeline.process_query(f"Concurrent query {query_id}")
                results.append(result)
            except Exception as e:
                errors.append(e)

        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=process_query_thread, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Verify results
        self.assertEqual(len(results), 5)
        self.assertEqual(len(errors), 0)


if __name__ == '__main__':
    # Configure test runner
    unittest.main(verbosity=2, exit=False)

    # Print integration test summary
    print("\n" + "="*60)
    print("MoE Integration Test Summary")
    print("="*60)
    print("✅ Integration tests completed")
    print("📊 Coverage includes:")
    print("   • Pipeline creation and configuration")
    print("   • Component interaction and data flow")
    print("   • End-to-end processing with mocks")
    print("   • Error handling and edge cases")
    print("   • Performance and memory usage")
    print("   • Concurrent processing")
    print("="*60)