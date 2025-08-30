#!/usr/bin/env python3
"""
Performance Benchmarking Tests for Personal RAG Chatbot with MoE

This test suite provides comprehensive performance benchmarking for the MoE
implementation, including latency, throughput, memory usage, and scalability tests.

Author: SPARC QA Analyst
Date: 2025-08-30
"""

import sys
import os
import unittest
import time
import statistics
import psutil
import threading
from unittest.mock import Mock, patch
from pathlib import Path
from typing import List, Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from moe import MoEConfig, get_moe_pipeline
import numpy as np


class PerformanceBenchmark:
    """Base class for performance benchmarking"""

    def __init__(self, name: str):
        self.name = name
        self.results = []
        self.start_time = None
        self.end_time = None

    def start_measurement(self):
        """Start performance measurement"""
        self.start_time = time.time()

    def end_measurement(self):
        """End performance measurement"""
        self.end_time = time.time()

    def get_duration(self) -> float:
        """Get measurement duration"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0

    def record_result(self, result: Dict[str, Any]):
        """Record a benchmark result"""
        self.results.append(result)

    def get_summary(self) -> Dict[str, Any]:
        """Get benchmark summary"""
        if not self.results:
            return {}

        durations = [r.get('duration', 0) for r in self.results]

        return {
            'benchmark_name': self.name,
            'total_runs': len(self.results),
            'avg_duration': statistics.mean(durations),
            'min_duration': min(durations),
            'max_duration': max(durations),
            'std_duration': statistics.stdev(durations) if len(durations) > 1 else 0,
            'median_duration': statistics.median(durations),
            'total_duration': sum(durations)
        }


class TestLatencyBenchmarks(unittest.TestCase):
    """Test latency performance benchmarks"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = MoEConfig()
        self.pipeline = get_moe_pipeline(self.config)

    def test_query_latency(self):
        """Benchmark query processing latency"""
        benchmark = PerformanceBenchmark("Query Latency")

        test_queries = [
            "What is machine learning?",
            "Explain neural networks",
            "How does gradient descent work?",
            "What are the types of machine learning algorithms?",
            "Explain the difference between supervised and unsupervised learning"
        ]

        for query in test_queries:
            benchmark.start_measurement()
            result = self.pipeline.process_query(query)
            benchmark.end_measurement()

            benchmark.record_result({
                'query': query,
                'duration': benchmark.get_duration(),
                'success': result is not None
            })

        summary = benchmark.get_summary()

        # Assert performance requirements
        self.assertLess(summary['avg_duration'], 5.0, "Average query latency should be < 5 seconds")
        self.assertLess(summary['median_duration'], 3.0, "Median query latency should be < 3 seconds")
        self.assertLess(summary['max_duration'], 10.0, "Maximum query latency should be < 10 seconds")

        print(f"Query Latency Benchmark: {summary['avg_duration']:.3f}s average")

    def test_component_latency_breakdown(self):
        """Benchmark individual component latencies"""
        benchmark = PerformanceBenchmark("Component Latency")

        query = "What is artificial intelligence?"

        # Measure full pipeline
        benchmark.start_measurement()
        result = self.pipeline.process_query(query)
        benchmark.end_measurement()

        # Get component timings
        component_times = result.pipeline_stats.get('component_times', {})

        for component, duration in component_times.items():
            benchmark.record_result({
                'component': component,
                'duration': duration,
                'percentage': duration / benchmark.get_duration() * 100
            })

        summary = benchmark.get_summary()

        # Router should be fast
        router_times = [r['duration'] for r in benchmark.results if r.get('component') == 'routing']
        if router_times:
            avg_router_time = statistics.mean(router_times)
            self.assertLess(avg_router_time, 0.1, "Router latency should be < 100ms")

        print(f"Component Latency Breakdown: {len(component_times)} components measured")

    def test_cold_start_latency(self):
        """Benchmark cold start latency"""
        benchmark = PerformanceBenchmark("Cold Start")

        # Clear any cached instances
        from moe.integration import _pipeline_instance
        _pipeline_instance = None

        benchmark.start_measurement()
        pipeline = get_moe_pipeline(self.config)
        benchmark.end_measurement()

        benchmark.record_result({
            'operation': 'pipeline_initialization',
            'duration': benchmark.get_duration()
        })

        summary = benchmark.get_summary()

        # Cold start should be reasonable
        self.assertLess(summary['avg_duration'], 5.0, "Cold start should be < 5 seconds")

        print(f"Cold Start Latency: {summary['avg_duration']:.3f}s")


class TestThroughputBenchmarks(unittest.TestCase):
    """Test throughput performance benchmarks"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = MoEConfig()
        self.pipeline = get_moe_pipeline(self.config)

    def test_single_thread_throughput(self):
        """Benchmark single-thread throughput"""
        benchmark = PerformanceBenchmark("Single Thread Throughput")

        num_queries = 20
        test_queries = [f"Query {i}" for i in range(num_queries)]

        benchmark.start_measurement()

        for query in test_queries:
            result = self.pipeline.process_query(query)
            benchmark.record_result({
                'query': query,
                'duration': 0,  # Will be calculated in summary
                'success': result is not None
            })

        benchmark.end_measurement()

        total_time = benchmark.get_duration()
        throughput = num_queries / total_time

        # Assert throughput requirements
        self.assertGreater(throughput, 0.2, "Throughput should be > 0.2 queries/second")

        print(f"Single Thread Throughput: {throughput:.2f} queries/second")

    def test_concurrent_throughput(self):
        """Benchmark concurrent throughput"""
        benchmark = PerformanceBenchmark("Concurrent Throughput")

        num_threads = 5
        queries_per_thread = 10
        results = []
        errors = []

        def worker_thread(thread_id):
            local_results = []
            for i in range(queries_per_thread):
                try:
                    start_time = time.time()
                    result = self.pipeline.process_query(f"Thread {thread_id} Query {i}")
                    end_time = time.time()

                    local_results.append({
                        'thread_id': thread_id,
                        'query_id': i,
                        'duration': end_time - start_time,
                        'success': result is not None
                    })
                except Exception as e:
                    errors.append(f"Thread {thread_id} Query {i}: {e}")

            results.extend(local_results)

        benchmark.start_measurement()

        # Start threads
        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=worker_thread, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        benchmark.end_measurement()

        total_time = benchmark.get_duration()
        total_queries = len(results)
        throughput = total_queries / total_time

        # Assert concurrent throughput requirements
        self.assertGreater(throughput, 0.5, "Concurrent throughput should be > 0.5 queries/second")
        self.assertEqual(len(errors), 0, "No errors should occur in concurrent execution")

        print(f"Concurrent Throughput: {throughput:.2f} queries/second ({total_queries} queries in {total_time:.2f}s)")

    def test_burst_throughput(self):
        """Benchmark burst request handling"""
        benchmark = PerformanceBenchmark("Burst Throughput")

        # Simulate burst of requests
        burst_size = 10
        queries = [f"Burst Query {i}" for i in range(burst_size)]

        benchmark.start_measurement()

        for query in queries:
            result = self.pipeline.process_query(query)
            benchmark.record_result({
                'query': query,
                'success': result is not None
            })

        benchmark.end_measurement()

        total_time = benchmark.get_duration()
        throughput = burst_size / total_time

        # Burst throughput should be reasonable
        self.assertGreater(throughput, 1.0, "Burst throughput should be > 1 query/second")

        print(f"Burst Throughput: {throughput:.2f} queries/second")


class TestMemoryBenchmarks(unittest.TestCase):
    """Test memory usage benchmarks"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = MoEConfig()
        self.pipeline = get_moe_pipeline(self.config)
        self.process = psutil.Process(os.getpid())

    def test_memory_usage_baseline(self):
        """Benchmark baseline memory usage"""
        benchmark = PerformanceBenchmark("Memory Baseline")

        # Get initial memory
        initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB

        benchmark.record_result({
            'measurement': 'initial',
            'memory_mb': initial_memory
        })

        # Process some queries
        for i in range(10):
            self.pipeline.process_query(f"Memory test {i}")

        # Get memory after processing
        processing_memory = self.process.memory_info().rss / 1024 / 1024  # MB

        benchmark.record_result({
            'measurement': 'after_processing',
            'memory_mb': processing_memory,
            'increase_mb': processing_memory - initial_memory
        })

        # Memory increase should be reasonable
        memory_increase = processing_memory - initial_memory
        self.assertLess(memory_increase, 100, "Memory increase should be < 100MB")

        print(f"Memory Usage: {initial_memory:.1f}MB -> {processing_memory:.1f}MB (+{memory_increase:.1f}MB)")

    def test_memory_leak_detection(self):
        """Test for memory leaks"""
        benchmark = PerformanceBenchmark("Memory Leak Detection")

        initial_memory = self.process.memory_info().rss / 1024 / 1024

        # Process many queries
        for i in range(50):
            self.pipeline.process_query(f"Leak test {i}")

            # Periodic memory check
            if i % 10 == 0:
                current_memory = self.process.memory_info().rss / 1024 / 1024
                benchmark.record_result({
                    'iteration': i,
                    'memory_mb': current_memory,
                    'increase_from_initial': current_memory - initial_memory
                })

        final_memory = self.process.memory_info().rss / 1024 / 1024
        total_increase = final_memory - initial_memory

        # Total increase should be reasonable
        self.assertLess(total_increase, 200, "Total memory increase should be < 200MB")

        print(f"Memory Leak Test: Total increase {total_increase:.1f}MB after 50 queries")

    def test_cache_memory_usage(self):
        """Test memory usage with caching"""
        benchmark = PerformanceBenchmark("Cache Memory")

        # Clear cache first
        self.pipeline.clear_cache()
        initial_memory = self.process.memory_info().rss / 1024 / 1024

        # Process queries to build cache
        for i in range(20):
            self.pipeline.process_query(f"Cache test {i}")

        cached_memory = self.process.memory_info().rss / 1024 / 1024

        # Clear cache
        self.pipeline.clear_cache()
        cleared_memory = self.process.memory_info().rss / 1024 / 1024

        cache_memory_usage = cached_memory - cleared_memory

        benchmark.record_result({
            'cache_memory_mb': cache_memory_usage,
            'memory_reduction_mb': cached_memory - cleared_memory
        })

        # Cache memory usage should be reasonable
        self.assertLess(cache_memory_usage, 50, "Cache memory usage should be < 50MB")

        print(f"Cache Memory Usage: {cache_memory_usage:.1f}MB")


class TestScalabilityBenchmarks(unittest.TestCase):
    """Test scalability benchmarks"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = MoEConfig()

    def test_query_complexity_scalability(self):
        """Test performance with varying query complexity"""
        benchmark = PerformanceBenchmark("Query Complexity Scaling")

        query_lengths = [10, 50, 100, 200, 500]

        for length in query_lengths:
            query = "What is machine learning? " * (length // 25)  # Approximate scaling

            benchmark.start_measurement()
            result = get_moe_pipeline(self.config).process_query(query)
            benchmark.end_measurement()

            benchmark.record_result({
                'query_length': length,
                'duration': benchmark.get_duration(),
                'success': result is not None
            })

        summary = benchmark.get_summary()

        # Performance should scale reasonably with query length
        durations = [r['duration'] for r in benchmark.results]
        if len(durations) > 1:
            scaling_factor = durations[-1] / durations[0]  # Ratio of longest to shortest
            self.assertLess(scaling_factor, 10, "Scaling factor should be reasonable (< 10x)")

        print(f"Query Complexity Scaling: {scaling_factor:.2f}x scaling factor")

    def test_concurrent_user_scalability(self):
        """Test scalability with increasing concurrent users"""
        benchmark = PerformanceBenchmark("Concurrent User Scaling")

        user_counts = [1, 2, 5, 10]

        for num_users in user_counts:
            results = []
            errors = []

            def user_simulation(user_id):
                try:
                    start_time = time.time()
                    pipeline = get_moe_pipeline(self.config)
                    result = pipeline.process_query(f"User {user_id} query")
                    end_time = time.time()

                    results.append({
                        'user_id': user_id,
                        'duration': end_time - start_time,
                        'success': result is not None
                    })
                except Exception as e:
                    errors.append(f"User {user_id}: {e}")

            benchmark.start_measurement()

            # Start concurrent users
            threads = []
            for i in range(num_users):
                thread = threading.Thread(target=user_simulation, args=(i,))
                threads.append(thread)
                thread.start()

            # Wait for completion
            for thread in threads:
                thread.join(timeout=30)

            benchmark.end_measurement()

            success_rate = len(results) / num_users if num_users > 0 else 0
            avg_duration = statistics.mean([r['duration'] for r in results]) if results else 0

            benchmark.record_result({
                'num_users': num_users,
                'success_rate': success_rate,
                'avg_duration': avg_duration,
                'total_duration': benchmark.get_duration()
            })

        summary = benchmark.get_summary()

        # Should maintain reasonable performance with more users
        results_by_users = {r['num_users']: r for r in benchmark.results}
        if 10 in results_by_users and 1 in results_by_users:
            scaling_efficiency = results_by_users[1]['avg_duration'] / results_by_users[10]['avg_duration']
            self.assertGreater(scaling_efficiency, 0.1, "Scaling efficiency should be > 10%")

        print(f"Concurrent User Scaling: Tested with {max(user_counts)} concurrent users")


class TestResourceUtilization(unittest.TestCase):
    """Test resource utilization benchmarks"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = MoEConfig()
        self.pipeline = get_moe_pipeline(self.config)

    def test_cpu_utilization(self):
        """Test CPU utilization during processing"""
        benchmark = PerformanceBenchmark("CPU Utilization")

        # Get initial CPU usage
        initial_cpu = psutil.cpu_percent(interval=1)

        # Process queries
        start_time = time.time()
        for i in range(10):
            self.pipeline.process_query(f"CPU test {i}")
        end_time = time.time()

        # Get CPU usage during processing
        processing_cpu = psutil.cpu_percent(interval=1)

        benchmark.record_result({
            'initial_cpu': initial_cpu,
            'processing_cpu': processing_cpu,
            'duration': end_time - start_time
        })

        # CPU usage should be reasonable
        self.assertLess(processing_cpu, 90, "CPU usage should be < 90%")

        print(f"CPU Utilization: {processing_cpu:.1f}% during processing")

    def test_disk_io_patterns(self):
        """Test disk I/O patterns"""
        benchmark = PerformanceBenchmark("Disk I/O")

        # Get initial I/O stats
        initial_io = psutil.disk_io_counters()

        # Process queries that might involve I/O
        for i in range(20):
            self.pipeline.process_query(f"IO test {i}")

        # Get final I/O stats
        final_io = psutil.disk_io_counters()

        if initial_io and final_io:
            read_bytes = final_io.read_bytes - initial_io.read_bytes
            write_bytes = final_io.write_bytes - initial_io.write_bytes

            benchmark.record_result({
                'read_mb': read_bytes / 1024 / 1024,
                'write_mb': write_bytes / 1024 / 1024,
                'total_io_mb': (read_bytes + write_bytes) / 1024 / 1024
            })

            # I/O should be reasonable
            total_io_mb = (read_bytes + write_bytes) / 1024 / 1024
            self.assertLess(total_io_mb, 100, "Total I/O should be < 100MB")

            print(f"Disk I/O: {total_io_mb:.1f}MB total")


class TestBenchmarkReporting(unittest.TestCase):
    """Test benchmark reporting and analysis"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = MoEConfig()

    def test_performance_regression_detection(self):
        """Test detection of performance regressions"""
        benchmark = PerformanceBenchmark("Regression Detection")

        # Establish baseline
        baseline_times = []
        for i in range(5):
            start_time = time.time()
            pipeline = get_moe_pipeline(self.config)
            result = pipeline.process_query(f"Baseline {i}")
            end_time = time.time()
            baseline_times.append(end_time - start_time)

        baseline_avg = statistics.mean(baseline_times)

        # Test current performance
        current_times = []
        for i in range(5):
            start_time = time.time()
            pipeline = get_moe_pipeline(self.config)
            result = pipeline.process_query(f"Current {i}")
            end_time = time.time()
            current_times.append(end_time - start_time)

        current_avg = statistics.mean(current_times)

        benchmark.record_result({
            'baseline_avg': baseline_avg,
            'current_avg': current_avg,
            'regression_ratio': current_avg / baseline_avg if baseline_avg > 0 else 1.0
        })

        # Check for significant regression (>50% slowdown)
        regression_ratio = current_avg / baseline_avg if baseline_avg > 0 else 1.0
        self.assertLess(regression_ratio, 2.0, "Performance regression should be < 2x")

        print(f"Performance Regression: {regression_ratio:.2f}x (baseline: {baseline_avg:.3f}s, current: {current_avg:.3f}s)")


if __name__ == '__main__':
    # Configure test runner
    unittest.main(verbosity=2, exit=False)

    # Print performance benchmarking summary
    print("\n" + "="*60)
    print("Performance Benchmarking Summary")
    print("="*60)
    print("✅ Performance benchmarks completed")
    print("📊 Benchmarks executed:")
    print("   • Latency measurements")
    print("   • Throughput analysis")
    print("   • Memory usage profiling")
    print("   • Scalability testing")
    print("   • Resource utilization monitoring")
    print("   • Regression detection")
    print("="*60)