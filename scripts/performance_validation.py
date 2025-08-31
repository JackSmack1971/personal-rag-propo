#!/usr/bin/env python3
"""
Performance Validation Script for Phase 5 Implementation
Validates OpenVINO quantization, caching, and benchmarking improvements.
"""

import sys
import os
import time
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import AppConfig
from embeddings import get_embedder, encode_optimized, get_performance_metrics
from caching import embedding_cache, query_cache, warm_up_manager, get_cache_stats
from performance_benchmark import benchmarker, benchmark_embedding_generation

def validate_openvino_quantization():
    """Validate OpenVINO quantization performance"""
    print("=== Validating OpenVINO Quantization ===")

    cfg = AppConfig.from_env()
    cfg.SENTENCE_TRANSFORMERS_BACKEND = "openvino"

    embedder = get_embedder(cfg.EMBED_MODEL, backend="openvino")
    test_sentences = ["This is a test sentence for performance validation."] * 10

    # Benchmark OpenVINO encoding
    result = benchmark_embedding_generation(embedder, test_sentences)
    print(f"OpenVINO encoding: {result.duration:.3f}s for {len(test_sentences)} sentences")
    print(".3f")

    return result.duration

def validate_caching_system():
    """Validate embedding and query caching"""
    print("\n=== Validating Caching System ===")

    cfg = AppConfig.from_env()
    embedder = get_embedder(cfg.EMBED_MODEL)

    test_query = "What is machine learning?"

    # First encoding (should cache)
    start_time = time.time()
    embedding1 = encode_optimized(embedder, [test_query])[0]
    first_duration = time.time() - start_time

    # Second encoding (should use cache)
    start_time = time.time()
    embedding2 = encode_optimized(embedder, [test_query])[0]
    second_duration = time.time() - start_time

    speedup = first_duration / second_duration if second_duration > 0 else float('inf')
    print(f"Cache speedup: {speedup:.1f}x (first: {first_duration:.3f}s, cached: {second_duration:.3f}s)")

    # Check cache stats
    cache_stats = get_cache_stats()
    print(f"Embedding cache hit rate: {cache_stats['embedding_cache']['hit_rate']:.1%}")
    print(f"Cache size: {cache_stats['embedding_cache']['total_size_mb']:.1f}MB")

    return speedup > 1.5  # Expect at least 1.5x speedup

def validate_batch_processing():
    """Validate batch processing optimization"""
    print("\n=== Validating Batch Processing ===")

    cfg = AppConfig.from_env()
    embedder = get_embedder(cfg.EMBED_MODEL)

    sentences = [f"Test sentence number {i} for batch processing validation." for i in range(20)]

    # Test different batch sizes
    batch_sizes = [1, 5, 10, 20]

    results = {}
    for batch_size in batch_sizes:
        start_time = time.time()
        embeddings = encode_optimized(embedder, sentences, batch_size=batch_size)
        duration = time.time() - start_time

        results[batch_size] = duration
        print(f"Batch size {batch_size}: {duration:.3f}s for {len(sentences)} sentences")

    # Find optimal batch size
    optimal_batch = min(results.keys(), key=lambda x: results[x])
    print(f"Optimal batch size: {optimal_batch}")

    return optimal_batch > 1

def validate_memory_optimization():
    """Validate memory usage optimization"""
    print("\n=== Validating Memory Optimization ===")

    cfg = AppConfig.from_env()
    embedder = get_embedder(cfg.EMBED_MODEL)

    # Get initial memory metrics
    initial_metrics = get_performance_metrics()

    # Process multiple batches
    sentences = [f"Memory test sentence {i}" for i in range(50)]
    embeddings = encode_optimized(embedder, sentences, batch_size=10)

    # Get final memory metrics
    final_metrics = get_performance_metrics()

    memory_increase = final_metrics['system_memory'] - initial_metrics['system_memory']
    print(f"Memory increase: {memory_increase:.1f}MB")
    print(f"Embeddings generated: {len(embeddings)}")

    # Memory per embedding should be reasonable (< 1MB per 10 embeddings)
    memory_per_10_embeddings = memory_increase / (len(embeddings) / 10)
    print(f"Memory per 10 embeddings: {memory_per_10_embeddings:.2f}MB")

    return memory_per_10_embeddings < 2.0  # Reasonable threshold

def validate_warm_up_mechanism():
    """Validate model warm-up mechanism"""
    print("\n=== Validating Warm-up Mechanism ===")

    cfg = AppConfig.from_env()
    embedder = get_embedder(cfg.EMBED_MODEL)

    model_name = cfg.EMBED_MODEL
    backend = cfg.SENTENCE_TRANSFORMERS_BACKEND

    # Warm up model
    warm_up_time = warm_up_manager.warm_up_model(embedder, model_name, backend)
    print(f"Model warm-up time: {warm_up_time:.3f}s")

    # Check if model is marked as warmed up
    is_warmed = warm_up_manager.is_warmed_up(model_name, backend)
    print(f"Model warmed up status: {is_warmed}")

    return is_warmed and warm_up_time > 0

def run_performance_validation():
    """Run complete performance validation suite"""
    print("🚀 Starting Phase 5 Performance Validation\n")

    results = {}

    try:
        # Validate OpenVINO quantization
        openvino_time = validate_openvino_quantization()
        results['openvino'] = openvino_time < 2.0  # Should be fast

        # Validate caching system
        cache_working = validate_caching_system()
        results['caching'] = cache_working

        # Validate batch processing
        batch_working = validate_batch_processing()
        results['batch_processing'] = batch_working

        # Validate memory optimization
        memory_optimized = validate_memory_optimization()
        results['memory'] = memory_optimized

        # Validate warm-up mechanism
        warm_up_working = validate_warm_up_mechanism()
        results['warm_up'] = warm_up_working

    except Exception as e:
        print(f"❌ Validation failed with error: {e}")
        return False

    # Summary
    print("\n" + "="*50)
    print("📊 PERFORMANCE VALIDATION SUMMARY")
    print("="*50)

    all_passed = True
    for test, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print("15")
        if not passed:
            all_passed = False

    print(f"\n🎯 Overall Result: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")

    # Export performance report
    try:
        report = benchmarker.get_performance_report()
        report_file = Path("performance_validation_report.json")
        with open(report_file, 'w') as f:
            import json
            json.dump(report, f, indent=2, default=str)
        print(f"📄 Detailed report saved to: {report_file}")
    except Exception as e:
        print(f"⚠️  Failed to save report: {e}")

    return all_passed

if __name__ == "__main__":
    success = run_performance_validation()
    sys.exit(0 if success else 1)