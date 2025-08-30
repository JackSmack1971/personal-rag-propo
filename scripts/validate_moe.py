#!/usr/bin/env python3
"""
MoE Integration Validation Script

This script validates that the MoE (Mixture of Experts) architecture components
are properly integrated with the existing RAG system.
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_moe_imports():
    """Test that all MoE components can be imported successfully"""
    try:
        from moe.config import get_moe_config, MoEConfig
        from moe.router import ExpertRouter, get_expert_router
        from moe.gate import SelectiveGate, get_selective_gate
        from moe.reranker import TwoStageReranker, get_two_stage_reranker
        from moe.integration import MoEPipeline, get_moe_pipeline, process_query_with_moe

        logger.info("✅ All MoE components imported successfully")
        return True
    except ImportError as e:
        logger.error(f"❌ Failed to import MoE components: {e}")
        return False

def test_moe_config():
    """Test MoE configuration loading"""
    try:
        from moe.config import get_moe_config

        config = get_moe_config()
        logger.info(f"✅ MoE config loaded: enabled={config.enabled}")

        if config.enabled:
            logger.info(f"   Router enabled: {config.router.enabled}")
            logger.info(f"   Gate enabled: {config.gate.enabled}")
            logger.info(f"   Reranker enabled: {config.reranker.enabled}")
            logger.info(f"   Experts: {config.router.experts}")

        return True
    except Exception as e:
        logger.error(f"❌ Failed to load MoE config: {e}")
        return False

def test_rag_integration():
    """Test RAG system integration with MoE"""
    try:
        from rag import rag_chat
        from config import AppConfig
        from embeddings import get_embedder

        # Load configuration
        cfg = AppConfig.from_env()
        embedder = get_embedder(cfg.EMBED_MODEL)

        # Test traditional RAG (should always work)
        test_message = "What is machine learning?"
        result = rag_chat(cfg, embedder, test_message, [])
        logger.info("✅ Traditional RAG test passed")

        # Test MoE integration (if enabled)
        if cfg.moe_enabled:
            logger.info("🧠 Testing MoE-enhanced RAG...")
            moe_result = rag_chat(cfg, embedder, test_message, [])
            logger.info("✅ MoE-enhanced RAG test passed")
        else:
            logger.info("⏳ MoE disabled, skipping MoE-specific tests")

        return True
    except Exception as e:
        logger.error(f"❌ RAG integration test failed: {e}")
        return False

def test_moe_pipeline():
    """Test the complete MoE pipeline"""
    try:
        from moe.integration import get_moe_pipeline
        from moe.config import get_moe_config

        config = get_moe_config()
        if not config.enabled:
            logger.info("⏳ MoE disabled, skipping pipeline test")
            return True

        pipeline = get_moe_pipeline()

        # Mock functions for testing
        def mock_retrieval(query=None, query_embedding=None, top_k=5, **kwargs):
            return {
                "matches": [
                    {
                        "id": "test_1",
                        "score": 0.8,
                        "metadata": {
                            "text": "This is a test document about machine learning.",
                            "file": "test.pdf",
                            "page": 1
                        }
                    }
                ]
            }

        def mock_generation(query=None, context=None, retrieval_matches=None):
            return f"Mock response for query: {query}"

        # Test pipeline
        result = pipeline.process_query(
            query="What is machine learning?",
            retrieval_function=mock_retrieval,
            generation_function=mock_generation
        )

        logger.info("✅ MoE pipeline test passed")
        logger.info(f"   Pipeline stages executed: {result.pipeline_stats['stages_executed']}")
        answer_preview = result.final_answer[:100] if result.final_answer else "None"
        logger.info(f"   Final answer: {answer_preview}...")

        return True
    except Exception as e:
        logger.error(f"❌ MoE pipeline test failed: {e}")
        return False

def main():
    """Run all MoE integration validation tests"""
    logger.info("🚀 Starting MoE Integration Validation")
    logger.info("=" * 50)

    tests = [
        ("MoE Imports", test_moe_imports),
        ("MoE Configuration", test_moe_config),
        ("RAG Integration", test_rag_integration),
        ("MoE Pipeline", test_moe_pipeline),
    ]

    results = []
    for test_name, test_func in tests:
        logger.info(f"\n🔍 Running {test_name} test...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))

    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📊 VALIDATION SUMMARY")
    logger.info("=" * 50)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{status} - {test_name}")
        if result:
            passed += 1

    logger.info("-" * 50)
    logger.info(f"Overall: {passed}/{total} tests passed")

    if passed == total:
        logger.info("🎉 All MoE integration tests passed!")
        return 0
    else:
        logger.error(f"⚠️  {total - passed} test(s) failed")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)