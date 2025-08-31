"""
Performance Benchmarking Framework for Personal RAG
Implements comprehensive benchmarking, regression detection, and optimization impact measurement.
"""

import time
import logging
import psutil
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import numpy as np
from datetime import datetime
import statistics

logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """Result of a benchmark run"""
    operation: str
    timestamp: float
    duration: float
    memory_usage: float
    memory_delta: float
    cpu_percent: float
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceBaseline:
    """Performance baseline for comparison"""
    operation: str
    mean_duration: float
    std_duration: float
    mean_memory: float
    sample_count: int
    last_updated: float
    thresholds: Dict[str, float] = field(default_factory=dict)

class PerformanceBenchmarker:
    """Comprehensive performance benchmarking framework"""

    def __init__(self, baseline_file: str = "performance_baselines.json"):
        self.baseline_file = Path(baseline_file)
        self.baselines: Dict[str, PerformanceBaseline] = {}
        self.current_results: List[BenchmarkResult] = []
        self._load_baselines()
        self._monitor_thread: Optional[threading.Thread] = None
        self._monitoring = False

    def _load_baselines(self):
        """Load performance baselines from file"""
        if self.baseline_file.exists():
            try:
                with open(self.baseline_file, 'r') as f:
                    data = json.load(f)
                    for op, baseline_data in data.items():
                        self.baselines[op] = PerformanceBaseline(**baseline_data)
                logger.info(f"Loaded {len(self.baselines)} performance baselines")
            except Exception as e:
                logger.error(f"Failed to load baselines: {e}")

    def _save_baselines(self):
        """Save performance baselines to file"""
        try:
            data = {op: {
                "operation": b.operation,
                "mean_duration": b.mean_duration,
                "std_duration": b.std_duration,
                "mean_memory": b.mean_memory,
                "sample_count": b.sample_count,
                "last_updated": b.last_updated,
                "thresholds": b.thresholds
            } for op, b in self.baselines.items()}

            with open(self.baseline_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved {len(self.baselines)} performance baselines")
        except Exception as e:
            logger.error(f"Failed to save baselines: {e}")

    def benchmark_operation(
        self,
        operation_name: str,
        operation_func: Callable,
        *args,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> BenchmarkResult:
        """
        Benchmark a specific operation with comprehensive metrics.

        Args:
            operation_name: Name of the operation being benchmarked
            operation_func: Function to benchmark
            metadata: Additional metadata to store
            *args, **kwargs: Arguments to pass to operation_func

        Returns:
            BenchmarkResult with comprehensive metrics
        """
        metadata = metadata or {}

        # Pre-operation metrics
        start_time = time.time()
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        initial_cpu = process.cpu_percent(interval=None)

        success = False
        error_message = None
        result = None

        try:
            # Execute operation
            result = operation_func(*args, **kwargs)
            success = True
        except Exception as e:
            error_message = str(e)
            logger.error(f"Benchmark operation failed: {e}")

        # Post-operation metrics
        end_time = time.time()
        duration = end_time - start_time
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_delta = final_memory - initial_memory
        final_cpu = process.cpu_percent(interval=None)

        benchmark_result = BenchmarkResult(
            operation=operation_name,
            timestamp=end_time,
            duration=duration,
            memory_usage=final_memory,
            memory_delta=memory_delta,
            cpu_percent=final_cpu,
            success=success,
            error_message=error_message,
            metadata=metadata
        )

        self.current_results.append(benchmark_result)

        # Check for regression
        self._check_regression(benchmark_result)

        logger.info(f"Benchmarked {operation_name}: {duration:.3f}s, {memory_delta:.1f}MB, success={success}")

        return benchmark_result

    def _check_regression(self, result: BenchmarkResult):
        """Check for performance regression against baseline"""
        if result.operation not in self.baselines:
            return

        baseline = self.baselines[result.operation]

        # Check duration regression
        duration_threshold = baseline.thresholds.get("duration_regression_threshold", 1.5)  # 50% slower
        if result.duration > baseline.mean_duration * duration_threshold:
            logger.warning(
                f"Performance regression detected for {result.operation}: "
                f"{result.duration:.3f}s vs baseline {baseline.mean_duration:.3f}s "
                f"({result.duration/baseline.mean_duration:.1f}x slower)"
            )

        # Check memory regression
        memory_threshold = baseline.thresholds.get("memory_regression_threshold", 1.2)  # 20% more memory
        if result.memory_delta > baseline.mean_memory * memory_threshold:
            logger.warning(
                f"Memory regression detected for {result.operation}: "
                f"{result.memory_delta:.1f}MB vs baseline {baseline.mean_memory:.1f}MB"
            )

    def update_baseline(self, operation: str, results: List[BenchmarkResult]):
        """Update performance baseline for an operation"""
        if not results:
            return

        durations = [r.duration for r in results if r.success]
        memory_deltas = [r.memory_delta for r in results if r.success]

        if not durations:
            logger.warning(f"No successful results for {operation} baseline update")
            return

        baseline = PerformanceBaseline(
            operation=operation,
            mean_duration=statistics.mean(durations),
            std_duration=statistics.stdev(durations) if len(durations) > 1 else 0,
            mean_memory=statistics.mean(memory_deltas) if memory_deltas else 0,
            sample_count=len(durations),
            last_updated=time.time(),
            thresholds={
                "duration_regression_threshold": 1.5,
                "memory_regression_threshold": 1.2,
                "min_samples_for_baseline": 5
            }
        )

        self.baselines[operation] = baseline
        self._save_baselines()

        logger.info(f"Updated baseline for {operation}: mean={baseline.mean_duration:.3f}s, std={baseline.std_duration:.3f}s")

    def start_resource_monitoring(self, interval: float = 1.0):
        """Start background resource monitoring"""
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_resources,
            args=(interval,),
            daemon=True
        )
        self._monitor_thread.start()
        logger.info("Started resource monitoring")

    def stop_resource_monitoring(self):
        """Stop background resource monitoring"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
        logger.info("Stopped resource monitoring")

    def _monitor_resources(self, interval: float):
        """Background resource monitoring thread"""
        process = psutil.Process()
        while self._monitoring:
            try:
                cpu_percent = process.cpu_percent(interval=interval)
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024

                # Store monitoring data
                monitoring_result = BenchmarkResult(
                    operation="resource_monitoring",
                    timestamp=time.time(),
                    duration=interval,
                    memory_usage=memory_mb,
                    memory_delta=0,  # Not applicable for monitoring
                    cpu_percent=cpu_percent,
                    success=True,
                    metadata={"monitoring_interval": interval}
                )

                self.current_results.append(monitoring_result)

            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                break

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_operations": len(self.current_results),
                "successful_operations": len([r for r in self.current_results if r.success]),
                "failed_operations": len([r for r in self.current_results if not r.success]),
                "total_duration": sum(r.duration for r in self.current_results),
                "average_memory_delta": statistics.mean([r.memory_delta for r in self.current_results]) if self.current_results else 0
            },
            "baselines": {
                op: {
                    "mean_duration": b.mean_duration,
                    "std_duration": b.std_duration,
                    "mean_memory": b.mean_memory,
                    "sample_count": b.sample_count,
                    "last_updated": datetime.fromtimestamp(b.last_updated).isoformat()
                } for op, b in self.baselines.items()
            },
            "recent_results": [
                {
                    "operation": r.operation,
                    "duration": r.duration,
                    "memory_delta": r.memory_delta,
                    "success": r.success,
                    "timestamp": datetime.fromtimestamp(r.timestamp).isoformat()
                } for r in self.current_results[-10:]  # Last 10 results
            ]
        }

        return report

    def export_report(self, filename: str):
        """Export performance report to file"""
        report = self.get_performance_report()
        try:
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Exported performance report to {filename}")
        except Exception as e:
            logger.error(f"Failed to export report: {e}")

# Global benchmarker instance
benchmarker = PerformanceBenchmarker()

def benchmark_embedding_generation(embedder, sentences: List[str], **kwargs) -> BenchmarkResult:
    """Benchmark embedding generation specifically"""
    def operation():
        return embedder.encode(sentences, **kwargs)

    return benchmarker.benchmark_operation(
        "embedding_generation",
        operation,
        metadata={
            "sentence_count": len(sentences),
            "model_name": getattr(embedder, '_model_name', 'unknown'),
            "backend": getattr(embedder, '_backend', 'unknown')
        }
    )

def benchmark_vector_search(query_embedding: np.ndarray, index, top_k: int = 10) -> BenchmarkResult:
    """Benchmark vector search operation"""
    def operation():
        return index.query(query_embedding.tolist(), top_k=top_k)

    return benchmarker.benchmark_operation(
        "vector_search",
        operation,
        metadata={"top_k": top_k, "embedding_dim": len(query_embedding)}
    )

def benchmark_rag_pipeline(pipeline_func: Callable, query: str, **kwargs) -> BenchmarkResult:
    """Benchmark complete RAG pipeline"""
    def operation():
        return pipeline_func(query, **kwargs)

    return benchmarker.benchmark_operation(
        "rag_pipeline",
        operation,
        metadata={"query_length": len(query)}
    )