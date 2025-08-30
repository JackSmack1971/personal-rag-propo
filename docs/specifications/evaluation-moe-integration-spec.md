# Evaluation and MoE Integration Specification

**Document ID:** EVALUATION-MOE-INTEGRATION-001
**Version:** 1.0.0
**Date:** 2025-08-30
**Authors:** SPARC Specification Writer

## 1. Overview

This specification defines the integration between the comprehensive evaluation harness and the Mixture of Experts (MoE) system for the Personal RAG Chatbot. It establishes how evaluation metrics, A/B testing, and performance monitoring work seamlessly with MoE components including expert routing, selective gating, and two-stage reranking.

## 2. System Architecture Integration

### 2.1 Integration Points

```mermaid
graph TB
    A[User Query] --> B[RAG Pipeline]
    B --> C[MoE System]
    C --> D[Expert Router]
    C --> E[Selective Gate]
    C --> F[Two-Stage Reranker]
    D --> G[Evaluation Harness]
    E --> G
    F --> G
    B --> H[Answer Generation]
    H --> G

    G --> I[Retrieval Metrics]
    G --> J[Citation Metrics]
    G --> K[MoE-Specific Metrics]
    G --> L[A/B Testing Framework]

    I --> M[Hit@k, NDCG@k, MRR]
    J --> N[Span Accuracy, Completeness]
    K --> O[Routing Accuracy, Gate Efficiency]
    L --> P[Variant Comparison]
```

### 2.2 Data Flow Integration

#### Query Processing with Evaluation

```python
class MoEEvaluatedRAGPipeline:
    """RAG Pipeline with integrated MoE evaluation"""

    def __init__(self, config, evaluator=None, ab_tester=None):
        self.config = config
        self.evaluator = evaluator
        self.ab_tester = ab_tester

        # Initialize MoE components
        self.expert_router = ExpertRouter(config.moe)
        self.selective_gate = SelectiveGate(config.moe)
        self.reranker = TwoStageReranker(config.moe)

        # Initialize evaluation tracking
        self.query_tracker = QueryTracker()

    async def process_query(self, query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Process query with full MoE and evaluation integration"""

        start_time = time.time()
        query_id = str(uuid.uuid4())

        # 1. A/B Testing: Allocate variant if experiment active
        variant_id, variant_config = "baseline", {}
        if self.ab_tester:
            variant_id, variant_config = self.ab_tester.allocate_traffic(
                self.config.active_experiment, session_id
            )

        # 2. Apply variant configuration
        effective_config = self._merge_configs(self.config, variant_config)

        # 3. Query embedding
        embed_start = time.time()
        query_embedding = await self._generate_embedding(query, effective_config)
        embed_time = time.time() - embed_start

        # 4. MoE Processing
        moe_start = time.time()

        # Expert routing
        routing_result = await self._perform_expert_routing(
            query_embedding, effective_config
        )

        # Selective gating
        gate_result = await self._perform_selective_gating(
            routing_result, effective_config
        )

        # Vector retrieval (conditional)
        retrieval_result = await self._perform_retrieval(
            query_embedding, gate_result, effective_config
        )

        # Two-stage reranking
        reranking_result = await self._perform_reranking(
            query, retrieval_result, effective_config
        )

        moe_time = time.time() - moe_start

        # 5. Answer generation
        answer_start = time.time()
        answer_result = await self._generate_answer(
            query, reranking_result, effective_config
        )
        answer_time = time.time() - answer_start

        # 6. Evaluation data collection
        evaluation_data = self._collect_evaluation_data(
            query_id, query, routing_result, gate_result,
            retrieval_result, reranking_result, answer_result
        )

        # 7. Record metrics for A/B testing
        if self.ab_tester and variant_id != "baseline":
            await self.ab_tester.record_query_result(
                self.config.active_experiment,
                variant_id,
                evaluation_data['metrics']
            )

        # 8. Real-time evaluation (if enabled)
        if self.evaluator and self.config.evaluation.real_time_enabled:
            await self.evaluator.evaluate_query_real_time(evaluation_data)

        total_time = time.time() - start_time

        return {
            'query_id': query_id,
            'answer': answer_result['answer'],
            'citations': answer_result['citations'],
            'variant': variant_id,
            'performance': {
                'total_time': total_time,
                'embedding_time': embed_time,
                'moe_time': moe_time,
                'answer_time': answer_time
            },
            'moe_metadata': {
                'routing': routing_result,
                'gating': gate_result,
                'reranking': reranking_result
            }
        }
```

## 3. MoE-Specific Evaluation Metrics

### 3.1 Expert Routing Metrics

#### Routing Accuracy

**Definition**: Measures how accurately the expert router assigns queries to the most appropriate retrieval experts.

**Formula**:
```
Routing_Accuracy = (1/|Q|) * ∑_{q∈Q} I(route_q ∈ optimal_experts_q)
```

Where:
- `Q`: Set of evaluated queries
- `route_q`: Experts selected for query q
- `optimal_experts_q`: Ground truth optimal experts for query q
- `I()`: Indicator function

#### Routing Confidence Distribution

**Definition**: Analyzes the confidence distribution of routing decisions to identify routing quality patterns.

```python
def evaluate_routing_confidence(routing_results: List[Dict]) -> Dict[str, float]:
    """Evaluate routing confidence distribution"""

    confidences = [r['confidence'] for r in routing_results]

    return {
        'mean_confidence': statistics.mean(confidences),
        'confidence_std': statistics.stdev(confidences),
        'high_confidence_ratio': sum(1 for c in confidences if c > 0.8) / len(confidences),
        'low_confidence_ratio': sum(1 for c in confidences if c < 0.3) / len(confidences)
    }
```

#### Expert Utilization Balance

**Definition**: Measures how evenly queries are distributed across available experts.

```python
def evaluate_expert_utilization(expert_assignments: Dict[str, int]) -> Dict[str, float]:
    """Evaluate expert utilization balance"""

    total_assignments = sum(expert_assignments.values())

    if total_assignments == 0:
        return {'balance_score': 0.0}

    # Calculate utilization rates
    utilization_rates = {
        expert: count / total_assignments
        for expert, count in expert_assignments.items()
    }

    # Ideal utilization (uniform distribution)
    ideal_rate = 1.0 / len(expert_assignments)

    # Balance score (1.0 = perfect balance, 0.0 = complete imbalance)
    balance_score = 1.0 - (statistics.stdev(list(utilization_rates.values())) / ideal_rate)

    return {
        'balance_score': max(0.0, balance_score),  # Clamp to [0, 1]
        'utilization_rates': utilization_rates,
        'most_utilized': max(utilization_rates.items(), key=lambda x: x[1]),
        'least_utilized': min(utilization_rates.items(), key=lambda x: x[1])
    }
```

### 3.2 Selective Gate Metrics

#### Gate Efficiency

**Definition**: Measures the effectiveness of the selective gate in reducing unnecessary retrieval operations.

**Formula**:
```
Gate_Efficiency = (Retrievals_Avoided / Total_Queries) * 100%
```

Where:
- `Retrievals_Avoided`: Queries where retrieval was correctly skipped
- `Total_Queries`: Total number of processed queries

#### Gate Precision and Recall

**Definition**: Evaluates the accuracy of gate decisions in predicting retrieval necessity.

```python
def evaluate_gate_performance(gate_decisions: List[Dict]) -> Dict[str, float]:
    """
    Evaluate selective gate performance

    Args:
        gate_decisions: List of gate decision records with ground truth

    Returns:
        Dictionary of gate performance metrics
    """

    # Extract decisions and ground truth
    decisions = []
    ground_truth = []

    for decision in gate_decisions:
        decisions.append(decision['should_retrieve'])
        ground_truth.append(decision['actually_needed_retrieval'])

    # Calculate confusion matrix
    tp = sum(1 for d, gt in zip(decisions, ground_truth) if d and gt)    # True positive
    tn = sum(1 for d, gt in zip(decisions, ground_truth) if not d and not gt)  # True negative
    fp = sum(1 for d, gt in zip(decisions, ground_truth) if d and not gt)  # False positive
    fn = sum(1 for d, gt in zip(decisions, ground_truth) if not d and gt)  # False negative

    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    accuracy = (tp + tn) / len(decisions) if decisions else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'accuracy': accuracy,
        'retrieval_rate': (tp + fp) / len(decisions),  # How often retrieval is performed
        'efficiency_gain': tn / len(decisions)  # How often retrieval is correctly avoided
    }
```

#### Adaptive K-Selection Performance

**Definition**: Evaluates how well the adaptive k-selection balances retrieval quality and efficiency.

```python
def evaluate_adaptive_k_performance(k_selections: List[Dict]) -> Dict[str, Any]:
    """
    Evaluate adaptive k-selection performance

    Args:
        k_selections: List of k-selection decisions with outcomes

    Returns:
        Dictionary of k-selection performance metrics
    """

    k_values = [s['selected_k'] for s in k_selections]
    quality_scores = [s['resulting_quality'] for s in k_selections]
    efficiency_scores = [s['efficiency_gain'] for s in k_selections]

    # Analyze k distribution
    k_distribution = {}
    for k in k_values:
        k_distribution[k] = k_distribution.get(k, 0) + 1

    # Quality vs k correlation
    quality_by_k = {}
    for selection in k_selections:
        k = selection['selected_k']
        if k not in quality_by_k:
            quality_by_k[k] = []
        quality_by_k[k].append(selection['resulting_quality'])

    avg_quality_by_k = {
        k: statistics.mean(scores) for k, scores in quality_by_k.items()
    }

    return {
        'k_distribution': k_distribution,
        'avg_k': statistics.mean(k_values),
        'k_std': statistics.stdev(k_values),
        'quality_by_k': avg_quality_by_k,
        'quality_efficiency_tradeoff': statistics.correlation(quality_scores, efficiency_scores)
    }
```

### 3.3 Reranking Performance Metrics

#### Reranking Quality Improvement

**Definition**: Measures the improvement in retrieval quality after reranking.

**Formula**:
```
Quality_Improvement = ((Score_Post - Score_Pre) / Score_Pre) * 100%
```

Where:
- `Score_Post`: Quality score after reranking
- `Score_Pre`: Quality score before reranking

#### Reranking Efficiency Metrics

**Definition**: Evaluates the computational efficiency of the reranking process.

```python
def evaluate_reranking_efficiency(reranking_results: List[Dict]) -> Dict[str, float]:
    """Evaluate reranking efficiency"""

    processing_times = [r['processing_time'] for r in reranking_results]
    quality_improvements = [r['quality_improvement'] for r in reranking_results]

    return {
        'mean_processing_time': statistics.mean(processing_times),
        'processing_time_std': statistics.stdev(processing_times),
        'mean_quality_improvement': statistics.mean(quality_improvements),
        'quality_improvement_std': statistics.stdev(quality_improvements),
        'efficiency_ratio': (statistics.mean(quality_improvements) /
                           statistics.mean(processing_times))
    }
```

#### Stage-wise Reranking Analysis

**Definition**: Analyzes the contribution of each reranking stage.

```python
def analyze_reranking_stages(stage_results: List[Dict]) -> Dict[str, Any]:
    """Analyze contribution of each reranking stage"""

    stage_contributions = {}

    for result in stage_results:
        for stage, contribution in result['stage_contributions'].items():
            if stage not in stage_contributions:
                stage_contributions[stage] = []
            stage_contributions[stage].append(contribution)

    # Calculate average contribution per stage
    avg_contributions = {
        stage: statistics.mean(contributions)
        for stage, contributions in stage_contributions.items()
    }

    # Calculate incremental improvements
    stages = ['stage1', 'stage2']
    incremental_improvements = {}

    for i in range(1, len(stages)):
        prev_stage = stages[i-1]
        curr_stage = stages[i]

        if prev_stage in avg_contributions and curr_stage in avg_contributions:
            incremental_improvements[f"{prev_stage}_to_{curr_stage}"] = (
                avg_contributions[curr_stage] - avg_contributions[prev_stage]
            )

    return {
        'stage_contributions': avg_contributions,
        'incremental_improvements': incremental_improvements,
        'best_contributing_stage': max(avg_contributions.items(), key=lambda x: x[1])
    }
```

## 4. Integrated Evaluation Workflows

### 4.1 Real-Time MoE Evaluation

#### Query-Level Evaluation

```python
class RealTimeMoEEvaluator:
    """Real-time evaluation of MoE system performance"""

    def __init__(self, config):
        self.config = config
        self.metrics_buffer = defaultdict(list)
        self.moe_metrics_buffer = defaultdict(list)

    async def evaluate_query_real_time(self, evaluation_data: Dict[str, Any]):
        """Evaluate query in real-time"""

        query_id = evaluation_data['query_id']

        # Standard retrieval metrics
        retrieval_metrics = await self._calculate_retrieval_metrics(evaluation_data)

        # MoE-specific metrics
        moe_metrics = await self._calculate_moe_metrics(evaluation_data)

        # Performance metrics
        performance_metrics = await self._calculate_performance_metrics(evaluation_data)

        # Store in buffers
        for metric_name, value in retrieval_metrics.items():
            self.metrics_buffer[metric_name].append(value)

        for metric_name, value in moe_metrics.items():
            self.moe_metrics_buffer[metric_name].append(value)

        # Check for alerts
        await self._check_performance_alerts(
            query_id, retrieval_metrics, moe_metrics, performance_metrics
        )

        return {
            'retrieval_metrics': retrieval_metrics,
            'moe_metrics': moe_metrics,
            'performance_metrics': performance_metrics
        }

    async def _calculate_moe_metrics(self, evaluation_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate MoE-specific metrics"""

        moe_metadata = evaluation_data.get('moe_metadata', {})

        metrics = {}

        # Routing metrics
        if 'routing' in moe_metadata:
            routing = moe_metadata['routing']
            metrics['routing_confidence'] = routing.get('confidence', 0.0)
            metrics['experts_selected'] = len(routing.get('selected_experts', []))

        # Gate metrics
        if 'gating' in moe_metadata:
            gating = moe_metadata['gating']
            metrics['gate_decision'] = 1.0 if gating.get('should_retrieve', False) else 0.0
            metrics['selected_k'] = gating.get('selected_k', 0)

        # Reranking metrics
        if 'reranking' in moe_metadata:
            reranking = moe_metadata['reranking']
            metrics['reranking_improvement'] = reranking.get('quality_improvement', 0.0)
            metrics['reranking_time'] = reranking.get('processing_time', 0.0)

        return metrics

    async def _check_performance_alerts(self, query_id: str,
                                       retrieval_metrics: Dict[str, float],
                                       moe_metrics: Dict[str, float],
                                       performance_metrics: Dict[str, float]):
        """Check for performance alerts"""

        alerts = []

        # Low routing confidence alert
        routing_confidence = moe_metrics.get('routing_confidence', 1.0)
        if routing_confidence < self.config.alerts.routing_confidence_threshold:
            alerts.append({
                'type': 'moe_routing',
                'severity': 'medium',
                'message': f'Low routing confidence: {routing_confidence:.3f}',
                'query_id': query_id
            })

        # High reranking time alert
        reranking_time = moe_metrics.get('reranking_time', 0.0)
        if reranking_time > self.config.alerts.reranking_time_threshold:
            alerts.append({
                'type': 'moe_reranking',
                'severity': 'medium',
                'message': f'High reranking time: {reranking_time:.3f}s',
                'query_id': query_id
            })

        # Trigger alerts
        for alert in alerts:
            await self._trigger_alert(alert)
```

### 4.2 A/B Testing with MoE Variants

#### MoE Component A/B Tests

```python
def create_moe_ab_test_configs() -> List[ExperimentConfig]:
    """Create A/B test configurations for MoE component evaluation"""

    experiments = []

    # Experiment 1: Router vs No Router
    experiments.append(ExperimentConfig(
        experiment_id="moe_router_ab_test",
        name="Expert Router A/B Test",
        description="Compare retrieval with and without expert routing",
        variants=[
            VariantConfig(
                variant_id="no_router",
                name="No Expert Routing",
                config_overrides={"moe": {"router": {"enabled": False}}}
            ),
            VariantConfig(
                variant_id="with_router",
                name="With Expert Routing",
                config_overrides={"moe": {"router": {"enabled": True}}}
            )
        ],
        traffic_allocation={"no_router": 0.5, "with_router": 0.5},
        primary_metric="ndcg@5",
        secondary_metrics=["mrr", "routing_accuracy"],
        min_sample_size=1000
    ))

    # Experiment 2: Gate Efficiency Test
    experiments.append(ExperimentConfig(
        experiment_id="moe_gate_ab_test",
        name="Selective Gate A/B Test",
        description="Compare retrieval with and without selective gating",
        variants=[
            VariantConfig(
                variant_id="no_gate",
                name="No Selective Gating",
                config_overrides={"moe": {"gate": {"enabled": False}}}
            ),
            VariantConfig(
                variant_id="with_gate",
                name="With Selective Gating",
                config_overrides={"moe": {"gate": {"enabled": True}}}
            )
        ],
        traffic_allocation={"no_gate": 0.5, "with_gate": 0.5},
        primary_metric="ndcg@5",
        secondary_metrics=["gate_efficiency", "query_latency"],
        min_sample_size=1000
    ))

    # Experiment 3: Full MoE Ablation Study
    experiments.append(ExperimentConfig(
        experiment_id="moe_ablation_study",
        name="MoE Ablation Study",
        description="Test impact of individual MoE components",
        variants=[
            VariantConfig(
                variant_id="baseline",
                name="Baseline (No MoE)",
                config_overrides={"moe": {"enabled": False}}
            ),
            VariantConfig(
                variant_id="router_only",
                name="Router Only",
                config_overrides={
                    "moe": {
                        "enabled": True,
                        "router": {"enabled": True},
                        "gate": {"enabled": False},
                        "reranker": {"stage1_enabled": False}
                    }
                }
            ),
            VariantConfig(
                variant_id="router_gate",
                name="Router + Gate",
                config_overrides={
                    "moe": {
                        "enabled": True,
                        "router": {"enabled": True},
                        "gate": {"enabled": True},
                        "reranker": {"stage1_enabled": False}
                    }
                }
            ),
            VariantConfig(
                variant_id="full_moe",
                name="Full MoE System",
                config_overrides={
                    "moe": {
                        "enabled": True,
                        "router": {"enabled": True},
                        "gate": {"enabled": True},
                        "reranker": {"stage1_enabled": True}
                    }
                }
            )
        ],
        traffic_allocation={
            "baseline": 0.3,
            "router_only": 0.2,
            "router_gate": 0.2,
            "full_moe": 0.3
        },
        primary_metric="ndcg@5",
        secondary_metrics=[
            "mrr", "routing_accuracy", "gate_efficiency",
            "reranking_improvement", "query_latency"
        ],
        min_sample_size=2000
    ))

    return experiments
```

### 4.3 Performance Regression Detection

#### MoE-Specific Regression Monitoring

```python
class MoERegressionMonitor:
    """Monitor for performance regressions in MoE system"""

    def __init__(self, config):
        self.config = config
        self.baseline_metrics = self._load_baseline_metrics()
        self.recent_metrics = defaultdict(list)
        self.regression_thresholds = {
            'routing_accuracy': -0.05,  # 5% degradation
            'gate_efficiency': -0.10,   # 10% degradation
            'reranking_improvement': -0.15,  # 15% degradation
            'query_latency': 0.20       # 20% increase
        }

    def check_for_regressions(self, current_metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Check for performance regressions"""

        regressions = []

        for metric_name, current_value in current_metrics.items():
            if metric_name in self.baseline_metrics:
                baseline_value = self.baseline_metrics[metric_name]['mean']
                threshold = self.regression_thresholds.get(metric_name, -0.05)

                # Calculate relative change
                if baseline_value != 0:
                    relative_change = (current_value - baseline_value) / baseline_value
                else:
                    relative_change = 0.0

                # Check for regression
                if relative_change < threshold:
                    regressions.append({
                        'metric': metric_name,
                        'baseline_value': baseline_value,
                        'current_value': current_value,
                        'relative_change': relative_change,
                        'threshold': threshold,
                        'severity': self._calculate_severity(relative_change, threshold)
                    })

        # Update recent metrics buffer
        for metric_name, value in current_metrics.items():
            self.recent_metrics[metric_name].append(value)

            # Maintain buffer size
            if len(self.recent_metrics[metric_name]) > self.config.buffer_size:
                self.recent_metrics[metric_name] = self.recent_metrics[metric_name][-self.config.buffer_size:]

        return regressions

    def _calculate_severity(self, relative_change: float, threshold: float) -> str:
        """Calculate regression severity"""

        deviation = abs(relative_change - threshold) / abs(threshold)

        if deviation > 2.0:
            return "critical"
        elif deviation > 1.0:
            return "high"
        elif deviation > 0.5:
            return "medium"
        else:
            return "low"

    def _load_baseline_metrics(self) -> Dict[str, Dict[str, float]]:
        """Load baseline metrics from historical data"""

        # This would typically load from a database or file
        # For now, return default baselines
        return {
            'routing_accuracy': {'mean': 0.85, 'std': 0.05},
            'gate_efficiency': {'mean': 0.30, 'std': 0.08},
            'reranking_improvement': {'mean': 0.12, 'std': 0.03},
            'query_latency': {'mean': 1.2, 'std': 0.3}
        }
```

## 5. Configuration Integration

### 5.1 Unified Configuration Schema

```yaml
# Integrated evaluation and MoE configuration
evaluation:
  enabled: true
  real_time_enabled: true
  ab_testing_enabled: true

  # Standard evaluation metrics
  metrics:
    retrieval:
      enabled: ["hit@1", "hit@3", "hit@5", "ndcg@5", "mrr"]
    citation:
      enabled: ["span_accuracy", "completeness", "correctness"]
    answer_quality:
      enabled: ["factual_consistency", "hallucination_rate"]

  # MoE-specific evaluation
  moe_metrics:
    routing:
      enabled: true
      track_confidence: true
      track_expert_utilization: true
    gating:
      enabled: true
      track_efficiency: true
      track_precision_recall: true
    reranking:
      enabled: true
      track_improvement: true
      track_stage_contributions: true

  # A/B testing configuration
  ab_testing:
    active_experiment: "moe_ablation_study"
    traffic_allocation_strategy: "weighted"
    statistical_test: "t_test"
    alpha: 0.05

  # Performance monitoring
  performance:
    regression_detection:
      enabled: true
      baseline_window_days: 7
      alert_threshold_std: 2.0
    real_time_alerts:
      enabled: true
      routing_confidence_threshold: 0.3
      reranking_time_threshold: 2.0

# MoE system configuration
moe:
  enabled: true

  # Expert router configuration
  router:
    enabled: true
    experts: ["general", "technical", "personal", "code"]
    centroid_refresh_interval: 3600
    confidence_threshold: 0.6

  # Selective gate configuration
  gate:
    enabled: true
    retrieve_sim_threshold: 0.62
    low_sim_threshold: 0.45
    k_min: 4
    k_max: 15
    default_top_k: 8

  # Reranking configuration
  reranker:
    stage1_enabled: true
    stage2_enabled: false
    cross_encoder_model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
    uncertainty_threshold: 0.15
    max_rerank_candidates: 50

  # Evaluation integration
  evaluation:
    metrics_collection: true
    performance_monitoring: true
    detailed_logging: true
```

## 6. Monitoring and Alerting Integration

### 6.1 Integrated Dashboard

#### Real-Time MoE Performance Dashboard

```python
class MoEPerformanceDashboard:
    """Integrated dashboard for MoE system monitoring"""

    def __init__(self, evaluator, ab_tester, performance_monitor):
        self.evaluator = evaluator
        self.ab_tester = ab_tester
        self.performance_monitor = performance_monitor

    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive MoE performance report"""

        report = {
            'timestamp': time.time(),
            'system_status': self._get_system_status(),
            'performance_metrics': self.performance_monitor.get_current_performance(),
            'moe_metrics': self._get_moe_metrics_summary(),
            'ab_testing_status': self._get_ab_testing_status(),
            'alerts': self._get_active_alerts(),
            'recommendations': self._generate_recommendations()
        }

        return report

    def _get_moe_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of MoE-specific metrics"""

        # This would aggregate metrics from the evaluator
        return {
            'routing': {
                'average_confidence': 0.78,
                'expert_utilization_balance': 0.85
            },
            'gating': {
                'efficiency': 0.32,
                'precision': 0.88,
                'recall': 0.76
            },
            'reranking': {
                'average_improvement': 0.14,
                'stage1_contribution': 0.10,
                'stage2_contribution': 0.04
            }
        }

    def _get_ab_testing_status(self) -> Dict[str, Any]:
        """Get current A/B testing status"""

        if not self.ab_tester:
            return {'status': 'disabled'}

        active_experiments = []

        # This would query active experiments from ab_tester
        # For now, return mock data
        return {
            'active_experiments': active_experiments,
            'total_experiments': 0,
            'experiments_running': 0
        }

    def _get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get list of active alerts"""

        # This would aggregate alerts from all components
        return [
            {
                'id': 'alert_001',
                'type': 'performance',
                'severity': 'medium',
                'message': 'Query latency increased by 15%',
                'timestamp': time.time() - 300
            }
        ]

    def _generate_recommendations(self) -> List[str]:
        """Generate system optimization recommendations"""

        recommendations = []

        # Analyze current metrics and generate recommendations
        # This would be based on the actual metrics collected

        recommendations.extend([
            "Consider increasing expert router confidence threshold",
            "Optimize selective gate parameters for better efficiency",
            "Enable stage 2 reranking for high-uncertainty queries"
        ])

        return recommendations
```

### 6.2 Automated Optimization

#### MoE Parameter Tuning

```python
class MoEParameterOptimizer:
    """Automated optimization of MoE parameters based on evaluation feedback"""

    def __init__(self, config, evaluator):
        self.config = config
        self.evaluator = evaluator
        self.optimization_history = []

    async def optimize_parameters(self) -> Dict[str, Any]:
        """Optimize MoE parameters based on recent performance"""

        # Get recent evaluation results
        recent_results = await self.evaluator.get_recent_results(window_hours=24)

        # Analyze performance patterns
        analysis = self._analyze_performance_patterns(recent_results)

        # Generate parameter recommendations
        recommendations = self._generate_parameter_recommendations(analysis)

        # Apply safe parameter updates
        applied_changes = await self._apply_safe_parameter_updates(recommendations)

        # Record optimization attempt
        self.optimization_history.append({
            'timestamp': time.time(),
            'analysis': analysis,
            'recommendations': recommendations,
            'applied_changes': applied_changes
        })

        return {
            'recommendations': recommendations,
            'applied_changes': applied_changes,
            'expected_impact': self._estimate_impact(applied_changes)
        }

    def _analyze_performance_patterns(self, results: List[Dict]) -> Dict[str, Any]:
        """Analyze performance patterns from evaluation results"""

        # Extract MoE-specific metrics
        routing_confidences = [r['moe_metrics'].get('routing_confidence', 0) for r in results]
        gate_efficiencies = [r['moe_metrics'].get('gate_efficiency', 0) for r in results]
        reranking_improvements = [r['moe_metrics'].get('reranking_improvement', 0) for r in results]

        return {
            'avg_routing_confidence': statistics.mean(routing_confidences),
            'routing_confidence_std': statistics.stdev(routing_confidences),
            'avg_gate_efficiency': statistics.mean(gate_efficiencies),
            'gate_efficiency_std': statistics.stdev(gate_efficiencies),
            'avg_reranking_improvement': statistics.mean(reranking_improvements),
            'reranking_improvement_std': statistics.stdev(reranking_improvements)
        }

    def _generate_parameter_recommendations(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate parameter optimization recommendations"""

        recommendations = {}

        # Router confidence threshold optimization
        if analysis['routing_confidence_std'] > 0.2:
            if analysis['avg_routing_confidence'] < 0.7:
                recommendations['router_confidence_threshold'] = 'decrease'
            else:
                recommendations['router_confidence_threshold'] = 'increase'

        # Gate efficiency optimization
        if analysis['avg_gate_efficiency'] < 0.2:
            recommendations['gate_efficiency'] = 'improve_selectivity'
        elif analysis['avg_gate_efficiency'] > 0.4:
            recommendations['gate_efficiency'] = 'reduce_selectivity'

        # Reranking optimization
        if analysis['avg_reranking_improvement'] < 0.05:
            recommendations['reranking'] = 'enable_stage2'
        elif analysis['avg_reranking_improvement'] > 0.2:
            recommendations['reranking'] = 'optimize_stage1'

        return recommendations
```

## 7. Quality Assurance and Validation

### 7.1 Integration Testing

#### End-to-End MoE Evaluation Test

```python
def test_moe_evaluation_integration():
    """Test complete MoE evaluation integration"""

    # Setup test components
    config = load_test_config()
    evaluator = RealTimeMoEEvaluator(config)
    ab_tester = ExperimentManager()
    performance_monitor = PerformanceMonitor()

    # Create test query
    test_query = {
        'query_id': 'test_query_001',
        'query_text': 'What are the benefits of machine learning?',
        'relevant_documents': ['doc1', 'doc2'],
        'ground_truth_answer': 'Machine learning offers benefits such as...',
        'citations': []
    }

    # Simulate MoE processing
    moe_metadata = {
        'routing': {
            'selected_experts': ['technical', 'general'],
            'confidence': 0.85
        },
        'gating': {
            'should_retrieve': True,
            'selected_k': 8
        },
        'reranking': {
            'quality_improvement': 0.12,
            'processing_time': 0.8
        }
    }

    # Test evaluation integration
    evaluation_data = {
        'query_id': test_query['query_id'],
        'query': test_query,
        'retrieval_results': {'retrieved_docs': ['doc1', 'doc3', 'doc2']},
        'answer_result': {
            'answer': test_query['ground_truth_answer'],
            'citations': test_query['citations']
        },
        'moe_metadata': moe_metadata,
        'performance': {
            'total_time': 1.5,
            'embedding_time': 0.3,
            'moe_time': 0.8,
            'answer_time': 0.4
        }
    }

    # Run evaluation
    results = asyncio.run(evaluator.evaluate_query_real_time(evaluation_data))

    # Validate results structure
    assert 'retrieval_metrics' in results
    assert 'moe_metrics' in results
    assert 'performance_metrics' in results

    # Validate MoE metrics
    moe_metrics = results['moe_metrics']
    assert 'routing_confidence' in moe_metrics
    assert 'gate_decision' in moe_metrics
    assert 'reranking_improvement' in moe_metrics

    print("MoE evaluation integration test passed!")
```

### 7.2 Performance Validation

#### MoE System Performance Benchmarks

```python
def benchmark_moe_system_performance():
    """Benchmark MoE system performance with evaluation overhead"""

    test_queries = load_test_query_set(100)

    # Benchmark without evaluation
    start_time = time.time()
    for query in test_queries:
        result = process_query_baseline(query)
    baseline_time = time.time() - start_time

    # Benchmark with full MoE evaluation
    start_time = time.time()
    for query in test_queries:
        result = process_query_with_moe_evaluation(query)
    moe_eval_time = time.time() - start_time

    # Calculate evaluation overhead
    evaluation_overhead = (moe_eval_time - baseline_time) / baseline_time * 100

    print(f"Baseline processing time: {baseline_time:.2f}s")
    print(f"MoE evaluation time: {moe_eval_time:.2f}s")
    print(f"Evaluation overhead: {evaluation_overhead:.1f}%")

    # Assert acceptable overhead
    assert evaluation_overhead < 50.0, f"Evaluation overhead too high: {evaluation_overhead:.1f}%"
```

## 8. Future Enhancements

### 8.1 Advanced Integration Features

- **Online Learning**: Continuous MoE parameter optimization based on evaluation feedback
- **Federated Evaluation**: Distributed evaluation across multiple MoE system instances
- **Multi-Objective Optimization**: Balance quality, latency, and resource usage
- **Predictive Monitoring**: ML-based prediction of MoE performance issues

### 8.2 Research Integration

- **Novel Metrics**: Integration of cutting-edge evaluation metrics from research
- **Comparative Analysis**: Automated comparison with state-of-the-art systems
- **Benchmark Datasets**: Integration with standard IR and QA benchmark datasets
- **Publication-Ready Reports**: Generate research paper quality evaluation reports

---

This specification provides a comprehensive framework for integrating evaluation capabilities with the MoE system, enabling data-driven optimization and continuous improvement of the Personal RAG Chatbot's retrieval and answering capabilities.