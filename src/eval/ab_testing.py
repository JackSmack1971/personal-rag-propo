"""
A/B Testing Framework for Personal RAG Chatbot Evaluation

This module provides a comprehensive A/B testing framework for comparing
different variants of the RAG system, with support for automated traffic
allocation, statistical analysis, and experiment management.

Author: SPARC Specification Writer
Date: 2025-08-30
"""

import random
import time
import uuid
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import json
from collections import defaultdict
import statistics
import math

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for an A/B testing experiment"""
    experiment_id: str
    name: str
    description: str

    # Variants
    variants: List['VariantConfig'] = field(default_factory=list)

    # Traffic allocation
    traffic_allocation: Dict[str, float] = field(default_factory=dict)

    # Statistical parameters
    statistical_test: str = "t_test"
    alpha: float = 0.05  # Significance level
    power: float = 0.80  # Statistical power

    # Duration and sample size
    min_sample_size: int = 1000
    max_duration_days: int = 30
    auto_stop_enabled: bool = True

    # Metrics to track
    primary_metric: str = "ndcg@5"
    secondary_metrics: List[str] = field(default_factory=list)

    # Feature flags
    feature_flags: Dict[str, Any] = field(default_factory=dict)

    # Status
    status: str = "created"
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    ended_at: Optional[float] = None


@dataclass
class VariantConfig:
    """Configuration for a single experiment variant"""
    variant_id: str
    name: str
    description: str

    # System configuration
    config_overrides: Dict[str, Any] = field(default_factory=dict)

    # Feature toggles
    features_enabled: List[str] = field(default_factory=list)
    features_disabled: List[str] = field(default_factory=list)

    # Resource allocation
    resource_limits: Dict[str, float] = field(default_factory=dict)


@dataclass
class ExperimentResults:
    """Results from an A/B testing experiment"""
    experiment_id: str
    timestamp: float

    # Sample information
    total_samples: int = 0
    variant_samples: Dict[str, int] = field(default_factory=dict)

    # Metric results
    primary_metric_results: Dict[str, 'MetricResult'] = field(default_factory=dict)
    secondary_metric_results: Dict[str, 'MetricResult'] = field(default_factory=dict)

    # Statistical analysis
    statistical_tests: Dict[str, 'StatisticalTestResult'] = field(default_factory=dict)

    # Experiment status
    status: str = "running"
    winner: Optional[str] = None
    confidence_level: float = 0.0

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricResult:
    """Result for a single metric"""
    metric_name: str
    variant_results: Dict[str, List[float]] = field(default_factory=dict)

    # Summary statistics
    means: Dict[str, float] = field(default_factory=dict)
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    # Statistical comparison
    p_value: float = 1.0
    effect_size: float = 0.0
    significant: bool = False

    # Additional info
    additional_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StatisticalTestResult:
    """Result of a statistical significance test"""
    test_type: str
    variant_a: str
    variant_b: str
    p_value: float
    significant: bool
    effect_size: float
    confidence_interval: Tuple[float, float]
    test_statistics: Dict[str, float] = field(default_factory=dict)


class TrafficAllocator:
    """Handles traffic allocation for A/B testing experiments"""

    def __init__(self, allocation_strategy: str = "uniform"):
        self.allocation_strategy = allocation_strategy
        self.session_assignments = {}  # session_id -> (experiment_id, variant_id)

    def allocate_traffic(self, experiment: ExperimentConfig,
                        session_id: Optional[str] = None) -> str:
        """
        Allocate traffic to a variant for the given experiment.

        Args:
            experiment: Experiment configuration
            session_id: Optional session identifier for sticky allocation

        Returns:
            Selected variant ID
        """
        # Check for existing session assignment
        if session_id and session_id in self.session_assignments:
            exp_id, variant_id = self.session_assignments[session_id]
            if exp_id == experiment.experiment_id:
                return variant_id

        # Allocate new variant
        variant_id = self._allocate_new_variant(experiment)

        # Store session assignment for stickiness
        if session_id:
            self.session_assignments[session_id] = (experiment.experiment_id, variant_id)

        return variant_id

    def _allocate_new_variant(self, experiment: ExperimentConfig) -> str:
        """Allocate traffic to a new variant"""

        if self.allocation_strategy == "uniform":
            return self._uniform_allocation(experiment)
        elif self.allocation_strategy == "weighted":
            return self._weighted_allocation(experiment)
        else:
            raise ValueError(f"Unknown allocation strategy: {self.allocation_strategy}")

    def _uniform_allocation(self, experiment: ExperimentConfig) -> str:
        """Uniform random allocation across variants"""
        variants = list(experiment.traffic_allocation.keys())
        return random.choice(variants)

    def _weighted_allocation(self, experiment: ExperimentConfig) -> str:
        """Weighted allocation based on traffic percentages"""
        variants = list(experiment.traffic_allocation.keys())
        weights = list(experiment.traffic_allocation.values())

        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            return random.choice(variants)

        normalized_weights = [w / total_weight for w in weights]

        # Weighted random selection
        r = random.random()
        cumulative = 0.0

        for variant, weight in zip(variants, normalized_weights):
            cumulative += weight
            if r <= cumulative:
                return variant

        # Fallback (should not reach here normally)
        return variants[-1]

    def get_allocation_stats(self, experiment: ExperimentConfig) -> Dict[str, Any]:
        """Get allocation statistics for an experiment"""
        total_sessions = len([s for s in self.session_assignments.values()
                            if s[0] == experiment.experiment_id])

        variant_counts = defaultdict(int)
        for exp_id, variant_id in self.session_assignments.values():
            if exp_id == experiment.experiment_id:
                variant_counts[variant_id] += 1

        stats = {
            'total_sessions': total_sessions,
            'variant_counts': dict(variant_counts),
            'target_allocation': experiment.traffic_allocation
        }

        # Calculate actual allocation percentages
        if total_sessions > 0:
            actual_allocation = {}
            for variant, count in variant_counts.items():
                actual_allocation[variant] = count / total_sessions
            stats['actual_allocation'] = actual_allocation

        return stats


class StatisticalAnalyzer:
    """Performs statistical analysis for A/B testing"""

    def __init__(self):
        self.confidence_levels = {
            0.95: 1.96,  # 95% confidence
            0.99: 2.576  # 99% confidence
        }

    def analyze_experiment(self, experiment: ExperimentConfig,
                          results_data: Dict[str, Any]) -> ExperimentResults:
        """
        Perform comprehensive statistical analysis of experiment results.

        Args:
            experiment: Experiment configuration
            results_data: Raw experiment results data

        Returns:
            Analyzed experiment results
        """
        # Extract variant results
        variant_results = results_data.get('variant_results', {})
        total_samples = sum(len(results) for results in variant_results.values())

        # Analyze primary metric
        primary_results = self._analyze_metric(
            experiment.primary_metric,
            variant_results,
            experiment.alpha
        )

        # Analyze secondary metrics
        secondary_results = {}
        for metric in experiment.secondary_metrics:
            secondary_results[metric] = self._analyze_metric(
                metric,
                variant_results,
                experiment.alpha
            )

        # Perform pairwise statistical tests
        statistical_tests = self._perform_statistical_tests(
            experiment.primary_metric,
            variant_results,
            experiment.alpha
        )

        # Determine winner
        winner = self._determine_winner(primary_results, experiment.alpha)

        # Calculate overall confidence
        confidence_level = self._calculate_overall_confidence(primary_results)

        return ExperimentResults(
            experiment_id=experiment.experiment_id,
            timestamp=time.time(),
            total_samples=total_samples,
            variant_samples={v: len(variant_results.get(v, []))
                           for v in experiment.traffic_allocation.keys()},
            primary_metric_results={experiment.primary_metric: primary_results},
            secondary_metric_results=secondary_results,
            statistical_tests=statistical_tests,
            status="completed",
            winner=winner,
            confidence_level=confidence_level
        )

    def _analyze_metric(self, metric_name: str,
                       variant_results: Dict[str, List[float]],
                       alpha: float) -> MetricResult:
        """Analyze a single metric across variants"""

        # Calculate summary statistics
        means = {}
        confidence_intervals = {}

        for variant, values in variant_results.items():
            if values:
                mean_val = statistics.mean(values)
                means[variant] = mean_val

                # Calculate confidence interval
                if len(values) > 1:
                    std_error = statistics.stdev(values) / math.sqrt(len(values))
                    margin_error = self.confidence_levels[0.95] * std_error
                    confidence_intervals[variant] = (
                        mean_val - margin_error,
                        mean_val + margin_error
                    )
                else:
                    confidence_intervals[variant] = (mean_val, mean_val)

        # Perform statistical comparison (simplified - between first two variants)
        variants = list(variant_results.keys())
        if len(variants) >= 2:
            variant_a, variant_b = variants[0], variants[1]
            values_a = variant_results.get(variant_a, [])
            values_b = variant_results.get(variant_b, [])

            if values_a and values_b:
                test_result = self._perform_t_test(values_a, values_b, alpha)
                p_value = test_result['p_value']
                effect_size = test_result['effect_size']
                significant = test_result['significant']
            else:
                p_value, effect_size, significant = 1.0, 0.0, False
        else:
            p_value, effect_size, significant = 1.0, 0.0, False

        return MetricResult(
            metric_name=metric_name,
            variant_results=variant_results,
            means=means,
            confidence_intervals=confidence_intervals,
            p_value=p_value,
            effect_size=effect_size,
            significant=significant
        )

    def _perform_t_test(self, values_a: List[float], values_b: List[float],
                       alpha: float) -> Dict[str, Any]:
        """Perform two-sample t-test"""

        if len(values_a) < 2 or len(values_b) < 2:
            return {
                'p_value': 1.0,
                'significant': False,
                'effect_size': 0.0,
                't_statistic': 0.0
            }

        try:
            # Use scipy if available, otherwise fallback to manual calculation
            from scipy import stats
            t_stat, p_value = stats.ttest_ind(values_a, values_b)
        except ImportError:
            # Manual t-test calculation (simplified)
            mean_a = statistics.mean(values_a)
            mean_b = statistics.mean(values_b)
            var_a = statistics.variance(values_a) if len(values_a) > 1 else 0
            var_b = statistics.variance(values_b) if len(values_b) > 1 else 0

            if var_a + var_b == 0:
                t_stat, p_value = 0.0, 1.0
            else:
                t_stat = (mean_a - mean_b) / math.sqrt(var_a/len(values_a) + var_b/len(values_b))
                # Approximate p-value (two-tailed)
                p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(t_stat) / math.sqrt(2))))

        # Calculate effect size (Cohen's d)
        pooled_std = math.sqrt(
            ((len(values_a) - 1) * statistics.variance(values_a) +
             (len(values_b) - 1) * statistics.variance(values_b)) /
            (len(values_a) + len(values_b) - 2)
        ) if len(values_a) > 1 and len(values_b) > 1 else 1.0

        effect_size = abs(statistics.mean(values_a) - statistics.mean(values_b)) / pooled_std

        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < alpha,
            'effect_size': effect_size,
            'mean_a': statistics.mean(values_a),
            'mean_b': statistics.mean(values_b)
        }

    def _perform_statistical_tests(self, metric_name: str,
                                 variant_results: Dict[str, List[float]],
                                 alpha: float) -> Dict[str, StatisticalTestResult]:
        """Perform pairwise statistical tests between all variant combinations"""

        variants = list(variant_results.keys())
        tests = {}

        for i in range(len(variants)):
            for j in range(i + 1, len(variants)):
                variant_a = variants[i]
                variant_b = variants[j]

                values_a = variant_results.get(variant_a, [])
                values_b = variant_results.get(variant_b, [])

                if values_a and values_b:
                    test_result = self._perform_t_test(values_a, values_b, alpha)

                    test_key = f"{variant_a}_vs_{variant_b}"
                    tests[test_key] = StatisticalTestResult(
                        test_type="t_test",
                        variant_a=variant_a,
                        variant_b=variant_b,
                        p_value=test_result['p_value'],
                        significant=test_result['significant'],
                        effect_size=test_result['effect_size'],
                        confidence_interval=(test_result['mean_a'], test_result['mean_b']),
                        test_statistics={
                            't_statistic': test_result['t_statistic'],
                            'mean_a': test_result['mean_a'],
                            'mean_b': test_result['mean_b']
                        }
                    )

        return tests

    def _determine_winner(self, primary_results: MetricResult,
                         alpha: float) -> Optional[str]:
        """Determine the winning variant based on statistical significance"""

        if not primary_results.significant:
            return None  # No statistically significant difference

        # Find variant with highest mean
        best_variant = max(primary_results.means.items(), key=lambda x: x[1])

        # Verify it's significantly better than others
        for variant, mean_val in primary_results.means.items():
            if variant != best_variant[0]:
                # Simplified check - in practice would check all pairwise comparisons
                if abs(best_variant[1] - mean_val) < 0.01:  # Very small difference
                    return None  # Too close to call

        return best_variant[0]

    def _calculate_overall_confidence(self, primary_results: MetricResult) -> float:
        """Calculate overall confidence in the experiment results"""

        if not primary_results.variant_results:
            return 0.0

        # Base confidence on statistical significance and sample sizes
        confidence = 0.5  # Neutral starting point

        if primary_results.significant:
            confidence += 0.3  # Significant result

        # Factor in sample sizes
        min_samples = min(len(values) for values in primary_results.variant_results.values())
        if min_samples >= 1000:
            confidence += 0.2
        elif min_samples >= 100:
            confidence += 0.1

        return min(confidence, 1.0)


class ExperimentManager:
    """Manages the lifecycle of A/B testing experiments"""

    def __init__(self, config_store=None, result_store=None):
        self.config_store = config_store or InMemoryConfigStore()
        self.result_store = result_store or InMemoryResultStore()
        self.active_experiments = {}
        self.traffic_allocator = TrafficAllocator()
        self.statistical_analyzer = StatisticalAnalyzer()

    def create_experiment(self, config: ExperimentConfig) -> str:
        """Create a new A/B testing experiment"""

        # Validate configuration
        self._validate_experiment_config(config)

        # Store configuration
        self.config_store.save_experiment_config(config.experiment_id, config)

        logger.info(f"Created experiment: {config.experiment_id} - {config.name}")

        return config.experiment_id

    def start_experiment(self, experiment_id: str):
        """Start an A/B testing experiment"""

        if experiment_id not in self.active_experiments:
            # Load from store if not in memory
            config = self.config_store.get_experiment_config(experiment_id)
            if not config:
                raise ValueError(f"Experiment {experiment_id} not found")

            self.active_experiments[experiment_id] = {
                'config': config,
                'results_data': {
                    'variant_results': defaultdict(lambda: defaultdict(list)),
                    'sample_counts': defaultdict(int)
                }
            }

        experiment = self.active_experiments[experiment_id]
        experiment['config'].status = "running"
        experiment['config'].started_at = time.time()

        # Update stored config
        self.config_store.save_experiment_config(experiment_id, experiment['config'])

        logger.info(f"Started experiment: {experiment_id}")

    def stop_experiment(self, experiment_id: str):
        """Stop an A/B testing experiment"""

        if experiment_id not in self.active_experiments:
            raise ValueError(f"Experiment {experiment_id} not found")

        experiment = self.active_experiments[experiment_id]
        experiment['config'].status = "stopped"
        experiment['config'].ended_at = time.time()

        # Generate final results
        results = self.statistical_analyzer.analyze_experiment(
            experiment['config'],
            experiment['results_data']
        )

        # Store results
        self.result_store.save_experiment_results(experiment_id, results)

        # Update stored config
        self.config_store.save_experiment_config(experiment_id, experiment['config'])

        logger.info(f"Stopped experiment: {experiment_id}")

    def record_query_result(self, experiment_id: str, variant_id: str,
                          metrics: Dict[str, float]):
        """Record the result of a query for an experiment"""

        if experiment_id not in self.active_experiments:
            return  # Experiment not active

        experiment = self.active_experiments[experiment_id]

        # Record metrics for each tracked metric
        config = experiment['config']

        # Primary metric
        if config.primary_metric in metrics:
            experiment['results_data']['variant_results'][config.primary_metric][variant_id].append(
                metrics[config.primary_metric]
            )

        # Secondary metrics
        for secondary_metric in config.secondary_metrics:
            if secondary_metric in metrics:
                experiment['results_data']['variant_results'][secondary_metric][variant_id].append(
                    metrics[secondary_metric]
                )

        # Update sample count
        experiment['results_data']['sample_counts'][variant_id] += 1

    def allocate_traffic(self, experiment_id: str,
                        session_id: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        """Allocate traffic for an experiment"""

        if experiment_id not in self.active_experiments:
            # Default to baseline if experiment not found
            return "baseline", {}

        experiment = self.active_experiments[experiment_id]
        config = experiment['config']

        if config.status != "running":
            return "baseline", {}

        # Allocate variant
        variant_id = self.traffic_allocator.allocate_traffic(config, session_id)

        # Get variant configuration
        variant_config = next(
            (v for v in config.variants if v.variant_id == variant_id),
            None
        )

        if not variant_config:
            return "baseline", {}

        return variant_id, variant_config.config_overrides

    def get_experiment_status(self, experiment_id: str) -> Dict[str, Any]:
        """Get the current status of an experiment"""

        if experiment_id not in self.active_experiments:
            config = self.config_store.get_experiment_config(experiment_id)
            if not config:
                raise ValueError(f"Experiment {experiment_id} not found")
        else:
            config = self.active_experiments[experiment_id]['config']

        status = {
            'experiment_id': config.experiment_id,
            'name': config.name,
            'status': config.status,
            'created_at': config.created_at,
            'started_at': config.started_at,
            'ended_at': config.ended_at
        }

        # Add results if experiment is completed
        if config.status == "stopped":
            results = self.result_store.get_experiment_results(experiment_id)
            if results:
                status['results'] = {
                    'total_samples': results.total_samples,
                    'winner': results.winner,
                    'confidence_level': results.confidence_level
                }

        return status

    def _validate_experiment_config(self, config: ExperimentConfig):
        """Validate experiment configuration"""

        if not config.experiment_id:
            raise ValueError("Experiment ID is required")

        if not config.variants:
            raise ValueError("At least one variant is required")

        if not config.traffic_allocation:
            raise ValueError("Traffic allocation is required")

        # Validate traffic allocation sums to reasonable value
        total_allocation = sum(config.traffic_allocation.values())
        if abs(total_allocation - 1.0) > 0.01:  # Allow small floating point errors
            raise ValueError(f"Traffic allocation must sum to 1.0, got {total_allocation}")

        # Validate variant references
        variant_ids = {v.variant_id for v in config.variants}
        allocation_ids = set(config.traffic_allocation.keys())

        if variant_ids != allocation_ids:
            raise ValueError("Variant IDs in config don't match traffic allocation")


class InMemoryConfigStore:
    """Simple in-memory configuration store"""

    def __init__(self):
        self.configs = {}

    def save_experiment_config(self, experiment_id: str, config: ExperimentConfig):
        """Save experiment configuration"""
        self.configs[experiment_id] = config

    def get_experiment_config(self, experiment_id: str) -> Optional[ExperimentConfig]:
        """Get experiment configuration"""
        return self.configs.get(experiment_id)


class InMemoryResultStore:
    """Simple in-memory result store"""

    def __init__(self):
        self.results = {}

    def save_experiment_results(self, experiment_id: str, results: ExperimentResults):
        """Save experiment results"""
        self.results[experiment_id] = results

    def get_experiment_results(self, experiment_id: str) -> Optional[ExperimentResults]:
        """Get experiment results"""
        return self.results.get(experiment_id)


# Convenience functions for common A/B testing operations

def create_simple_ab_test(experiment_name: str,
                         variant_a_config: Dict[str, Any],
                         variant_b_config: Dict[str, Any],
                         primary_metric: str = "ndcg@5") -> ExperimentConfig:
    """
    Create a simple A/B test with two variants.

    Args:
        experiment_name: Name of the experiment
        variant_a_config: Configuration for variant A
        variant_b_config: Configuration for variant B
        primary_metric: Primary metric to optimize

    Returns:
        Experiment configuration
    """

    experiment_id = f"ab_test_{int(time.time())}_{hashlib.md5(experiment_name.encode()).hexdigest()[:8]}"

    variants = [
        VariantConfig(
            variant_id="variant_a",
            name="Variant A",
            description=f"Variant A for {experiment_name}",
            config_overrides=variant_a_config
        ),
        VariantConfig(
            variant_id="variant_b",
            name="Variant B",
            description=f"Variant B for {experiment_name}",
            config_overrides=variant_b_config
        )
    ]

    return ExperimentConfig(
        experiment_id=experiment_id,
        name=experiment_name,
        description=f"A/B test comparing two variants",
        variants=variants,
        traffic_allocation={"variant_a": 0.5, "variant_b": 0.5},
        primary_metric=primary_metric,
        secondary_metrics=["hit@3", "mrr"],
        min_sample_size=1000,
        max_duration_days=7
    )


def quick_ab_test_analysis(variant_a_results: List[float],
                          variant_b_results: List[float],
                          alpha: float = 0.05) -> Dict[str, Any]:
    """
    Perform quick A/B test analysis on two result sets.

    Args:
        variant_a_results: Results for variant A
        variant_b_results: Results for variant B
        alpha: Significance level

    Returns:
        Analysis results
    """

    analyzer = StatisticalAnalyzer()

    # Create mock metric result
    mock_results = {
        'variant_a': variant_a_results,
        'variant_b': variant_b_results
    }

    metric_result = analyzer._analyze_metric("quick_test", mock_results, alpha)

    return {
        'variant_a_mean': metric_result.means.get('variant_a', 0),
        'variant_b_mean': metric_result.means.get('variant_b', 0),
        'improvement': (metric_result.means.get('variant_b', 0) -
                       metric_result.means.get('variant_a', 0)),
        'p_value': metric_result.p_value,
        'significant': metric_result.significant,
        'effect_size': metric_result.effect_size,
        'confidence_intervals': metric_result.confidence_intervals
    }


# Example usage and test functions

def example_moe_ab_test() -> ExperimentConfig:
    """Create an example A/B test for MoE evaluation"""

    return create_simple_ab_test(
        experiment_name="MoE vs Baseline",
        variant_a_config={
            "moe": {"enabled": False}
        },
        variant_b_config={
            "moe": {
                "enabled": True,
                "router": {"enabled": True},
                "gate": {"enabled": True},
                "reranker": {"stage1_enabled": True}
            }
        },
        primary_metric="ndcg@5"
    )


def run_example_ab_test():
    """Run a simple example A/B test"""

    # Create experiment
    experiment = example_moe_ab_test()
    manager = ExperimentManager()

    manager.create_experiment(experiment)
    manager.start_experiment(experiment.experiment_id)

    # Simulate some queries
    for i in range(100):
        # Allocate traffic
        variant_id, config = manager.allocate_traffic(
            experiment.experiment_id,
            session_id=f"session_{i}"
        )

        # Simulate query result
        if variant_id == "variant_a":
            # Baseline performance
            result = {"ndcg@5": random.gauss(0.75, 0.05)}
        else:
            # MoE performance (slightly better)
            result = {"ndcg@5": random.gauss(0.78, 0.05)}

        # Record result
        manager.record_query_result(experiment.experiment_id, variant_id, result)

    # Stop experiment and get results
    manager.stop_experiment(experiment.experiment_id)

    # Get final status
    status = manager.get_experiment_status(experiment.experiment_id)

    print(f"Experiment completed: {status}")

    return status


if __name__ == "__main__":
    # Run example when executed directly
    run_example_ab_test()