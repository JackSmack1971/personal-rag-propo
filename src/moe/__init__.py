"""
Mixture of Experts (MoE) Architecture Package

This package implements a sophisticated MoE system for enhanced retrieval-augmented generation
with intelligent query routing, adaptive retrieval, and multi-stage reranking.

Components:
- router: Expert routing based on query characteristics
- gate: Selective retrieval with adaptive k-selection
- reranker: Two-stage reranking with cross-encoder and LLM
- config: Configuration management for MoE system
- integration: Orchestration and pipeline management

Author: SPARC Code Implementer
Date: 2025-08-30
"""

from .config import (
    MoEConfig,
    MoERouterConfig,
    MoEGateConfig,
    MoERerankerConfig,
    MoEIntegrationConfig,
    get_moe_config,
    update_moe_config,
    save_moe_config
)

from .router import (
    ExpertRouter,
    RoutingDecision,
    get_expert_router,
    save_router_state
)

from .gate import (
    SelectiveGate,
    GateDecision,
    RetrievalMatch,
    get_selective_gate
)

from .reranker import (
    TwoStageReranker,
    RerankerResult,
    get_two_stage_reranker
)

from .integration import (
    MoEPipeline,
    MoEPipelineResult,
    get_moe_pipeline,
    process_query_with_moe
)

__version__ = "1.0.0"
__all__ = [
    # Configuration
    "MoEConfig",
    "MoERouterConfig",
    "MoEGateConfig",
    "MoERerankerConfig",
    "MoEIntegrationConfig",
    "get_moe_config",
    "update_moe_config",
    "save_moe_config",

    # Router
    "ExpertRouter",
    "RoutingDecision",
    "get_expert_router",
    "save_router_state",

    # Gate
    "SelectiveGate",
    "GateDecision",
    "RetrievalMatch",
    "get_selective_gate",

    # Reranker
    "TwoStageReranker",
    "RerankerResult",
    "get_two_stage_reranker",

    # Integration
    "MoEPipeline",
    "MoEPipelineResult",
    "get_moe_pipeline",
    "process_query_with_moe",
]