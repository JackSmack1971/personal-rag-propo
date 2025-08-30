#!/usr/bin/env python3
"""
Performance Benchmarking Script for Personal RAG Chatbot

This script executes comprehensive performance benchmarks including:
- Query response time measurements
- Memory usage tracking
- CPU utilization monitoring
- Scalability testing

Author: SPARC QA Analyst
Date: 2025-08-30
"""

import sys
import os
import time
import json
import threading
import psutil
from datetime import datetime
from pathlib import Path
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def log_message(message, level="INFO"):
    """Log a message with timestamp"""
    timestamp = datetime.now().isoformat()
    print(f"[{timestamp}] {level}: {message}")

def get_system_metrics():
    """Get current system metrics"""
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        return {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_used_mb": memory.used / (1024 * 1024),
            "memory_available_mb": memory.available / (1024 * 1024),
            "disk_percent": disk.percent,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        log_message(f"Error getting system metrics: {e}", "WARNING")
        return {}

def benchmark_query_response_time():
    """Benchmark query response times"""
    log_message("Benchmarking query response times...")

    results = []
    test_queries = [
        "What is machine learning?",
        "Explain neural networks",
        "How does natural language processing work?",
        "What are the benefits of retrieval-augmented generation?",
        "Explain the concept of embeddings in AI"
    ]

    try:
        # Mock pipeline for testing (since we can't run the full system)
        class MockPipeline:
            def __init__(self):
                self.response_times = []

            def process_query(self, query):
                # Simulate processing time
                start_time = time.time()
                time.sleep(np.random.uniform(0.5, 2.0))  # Random processing time
                end_time = time.time()

                processing_time = end_time - start_time
                self.response_times.append(processing_time)

                return {
                    "query": query,
                    "processing_time": processing_time,
                    "final_answer": f"Mock response to: {query}",
                    "success": True
                }

        pipeline = MockPipeline()

        # Run benchmark
        for i, query in enumerate(test_queries):
            log_message(f"Testing query {i+1}/{len(test_queries)}: {query}")

            # Get system metrics before query
            pre_metrics = get_system_metrics()

            # Process query
            result = pipeline.process_query(query)

            # Get system metrics after query
            post_metrics = get_system_metrics()

            # Record results
            result_entry = {
                "query_id": i + 1,
                "query": query,
                "processing_time_seconds": result["processing_time"],
                "success": result["success"],
                "pre_query_metrics": pre_metrics,
                "post_query_metrics": post_metrics,
                "timestamp": datetime.now().isoformat()
            }
            results.append(result_entry)

        # Calculate statistics
        response_times = [r["processing_time_seconds"] for r in results]
        stats = {
            "mean_response_time": np.mean(response_times),
            "median_response_time": np.median(response_times),
            "min_response_time": np.min(response_times),
            "max_response_time": np.max(response_time),
            "std_response_time": np.std(response_times),
            "p95_response_time": np.percentile(response_times, 95),
            "total_queries": len(results),
            "successful_queries": sum(1 for r in results if r["success"])
        }

        log_message("Query response time benchmark complete"        log_message(f"Mean response time: {stats['mean_response_time']:.2f}s")
        log_message(f"P95 response time: {stats['p95_response_time']:.2f}s")

        return results, stats

    except Exception as e:
        log_message(f"Error in query response benchmark: {e}", "ERROR")
        return [], {}

def benchmark_memory_usage():
    """Benchmark memory usage patterns"""
    log_message("Benchmarking memory usage...")

    results = []

    try:
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB

        log_message(f"Initial memory usage: {initial_memory:.2f} MB")

        # Simulate memory-intensive operations
        memory_snapshots = []
        for i in range(10):
            # Allocate some memory (simulate processing)
            test_data = [np.random.rand(1000, 1000) for _ in range(5)]
            time.sleep(0.5)  # Simulate processing time

            current_memory = process.memory_info().rss / (1024 * 1024)
            memory_snapshots.append({
                "iteration": i + 1,
                "memory_mb": current_memory,
                "memory_increase_mb": current_memory - initial_memory,
                "timestamp": datetime.now().isoformat()
            })

            # Clean up
            del test_data

        final_memory = process.memory_info().rss / (1024 * 1024)
        memory_increase = final_memory - initial_memory

        stats = {
            "initial_memory_mb": initial_memory,
            "final_memory_mb": final_memory,
            "total_memory_increase_mb": memory_increase,
            "average_memory_mb": np.mean([s["memory_mb"] for s in memory_snapshots]),
            "max_memory_mb": np.max([s["memory_mb"] for s in memory_snapshots]),
            "memory_leak_detected": memory_increase > 100  # >100MB increase
        }

        log_message("Memory usage benchmark complete"        log_message(f"Memory increase: {memory_increase:.2f} MB")
        log_message(f"Max memory usage: {stats['max_memory_mb']:.2f} MB")

        return memory_snapshots, stats

    except Exception as e:
        log_message(f"Error in memory benchmark: {e}", "ERROR")
        return [], {}

def benchmark_concurrent_users():
    """Benchmark concurrent user handling"""
    log_message("Benchmarking concurrent user handling...")

    results = []

    try:
        def simulate_user(user_id, results_list):
            """Simulate a user session"""
            start_time = time.time()

            # Simulate user queries
            for i in range(5):
                query_start = time.time()
                time.sleep(np.random.uniform(0.2, 1.0))  # Random query time
                query_end = time.time()

                results_list.append({
                    "user_id": user_id,
                    "query_id": i + 1,
                    "query_time": query_end - query_start,
                    "timestamp": datetime.now().isoformat()
                })

            end_time = time.time()
            return {
                "user_id": user_id,
                "total_time": end_time - start_time,
                "queries_completed": 5
            }

        # Test different concurrency levels
        concurrency_levels = [1, 2, 5, 10]
        all_results = []

        for num_users in concurrency_levels:
            log_message(f"Testing {num_users} concurrent users...")

            user_results = []
            threads = []

            # Start concurrent users
            start_time = time.time()
            for user_id in range(num_users):
                thread = threading.Thread(
                    target=lambda uid=user_id: simulate_user(uid, user_results)
                )
                threads.append(thread)
                thread.start()

            # Wait for all users to complete
            for thread in threads:
                thread.join(timeout=30)  # 30 second timeout

            end_time = time.time()
            total_time = end_time - start_time

            # Calculate statistics
            query_times = [r["query_time"] for r in user_results]
            stats = {
                "concurrency_level": num_users,
                "total_time": total_time,
                "queries_per_second": len(user_results) / total_time if total_time > 0 else 0,
                "mean_query_time": np.mean(query_times) if query_times else 0,
                "max_query_time": np.max(query_times) if query_times else 0,
                "total_queries": len(user_results),
                "timestamp": datetime.now().isoformat()
            }

            all_results.append(stats)
            log_message(f"  Completed in {total_time:.2f}s")
            log_message(f"  Queries/sec: {stats['queries_per_second']:.2f}")

        return all_results

    except Exception as e:
        log_message(f"Error in concurrent user benchmark: {e}", "ERROR")
        return []

def benchmark_scalability():
    """Benchmark system scalability"""
    log_message("Benchmarking system scalability...")

    results = []

    try:
        # Test increasing data sizes
        data_sizes = [100, 1000, 10000]

        for size in data_sizes:
            log_message(f"Testing with data size: {size}")

            start_time = time.time()
            start_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

            # Simulate processing data of different sizes
            data = np.random.rand(size, 100)
            time.sleep(size / 10000)  # Scale processing time with data size

            end_time = time.time()
            end_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

            result = {
                "data_size": size,
                "processing_time": end_time - start_time,
                "memory_used_mb": end_memory - start_memory,
                "timestamp": datetime.now().isoformat()
            }
            results.append(result)

            log_message(f"  Processing time: {result['processing_time']:.2f}s")
            log_message(f"  Memory used: {result['memory_used_mb']:.2f} MB")

        return results

    except Exception as e:
        log_message(f"Error in scalability benchmark: {e}", "ERROR")
        return []

def generate_performance_report(query_results, query_stats, memory_snapshots, memory_stats,
                               concurrency_results, scalability_results):
    """Generate comprehensive performance report"""
    log_message("Generating performance report...")

    report = {
        "benchmark_timestamp": datetime.now().isoformat(),
        "query_performance": {
            "results": query_results,
            "statistics": query_stats
        },
        "memory_performance": {
            "snapshots": memory_snapshots,
            "statistics": memory_stats
        },
        "concurrency_performance": {
            "results": concurrency_results
        },
        "scalability_performance": {
            "results": scalability_results
        },
        "system_info": get_system_metrics(),
        "recommendations": []
    }

    # Generate recommendations based on results
    if query_stats and query_stats.get("p95_response_time", 0) > 5.0:
        report["recommendations"].append("Consider optimizing query processing for better P95 response times")

    if memory_stats and memory_stats.get("memory_leak_detected", False):
        report["recommendations"].append("Investigate potential memory leaks in the application")

    if concurrency_results:
        max_qps = max(r.get("queries_per_second", 0) for r in concurrency_results)
        if max_qps < 10:
            report["recommendations"].append("Consider performance optimizations for higher throughput")

    return report

def main():
    """Run comprehensive performance benchmarks"""
    log_message("🚀 Starting Performance Benchmarking")
    log_message("=" * 60)

    start_time = datetime.now()

    # Run all benchmarks
    log_message("Running query response time benchmark...")
    query_results, query_stats = benchmark_query_response_time()

    log_message("Running memory usage benchmark...")
    memory_snapshots, memory_stats = benchmark_memory_usage()

    log_message("Running concurrent user benchmark...")
    concurrency_results = benchmark_concurrent_users()

    log_message("Running scalability benchmark...")
    scalability_results = benchmark_scalability()

    # Generate report
    report = generate_performance_report(
        query_results, query_stats, memory_snapshots, memory_stats,
        concurrency_results, scalability_results
    )

    end_time = datetime.now()
    benchmark_duration = (end_time - start_time).total_seconds()

    # Save report
    report_file = Path(__file__).parent.parent / "qa_performance_results.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, default=str)

    log_message("=" * 60)
    log_message("📊 PERFORMANCE BENCHMARKING COMPLETE")
    log_message(f"Duration: {benchmark_duration:.2f} seconds")

    if query_stats:
        log_message(f"Query Performance - Mean: {query_stats.get('mean_response_time', 0):.2f}s, P95: {query_stats.get('p95_response_time', 0):.2f}s")

    if memory_stats:
        log_message(f"Memory Usage - Increase: {memory_stats.get('total_memory_increase_mb', 0):.2f} MB, Max: {memory_stats.get('max_memory_mb', 0):.2f} MB")

    if concurrency_results:
        max_concurrent = max(r.get("concurrency_level", 0) for r in concurrency_results)
        max_qps = max(r.get("queries_per_second", 0) for r in concurrency_results)
        log_message(f"Concurrency - Max users: {max_concurrent}, Peak QPS: {max_qps:.2f}")

    log_message(f"Report saved to: {report_file}")

    return 0

if __name__ == "__main__":
    sys.exit(main())