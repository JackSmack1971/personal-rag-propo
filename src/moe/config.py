"""
MoE Configuration Module

This module provides configuration classes and utilities for the Mixture of Experts (MoE) system.
It extends the base application configuration with MoE-specific settings and validation.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
import yaml
import logging

logger = logging.getLogger(__name__)


@dataclass
class MoERouterConfig:
    """Configuration for Expert Router component"""

    enabled: bool = True
    experts: List[str] = field(default_factory=lambda: ["general", "technical", "personal", "code"])
    centroid_refresh_interval: int = 3600  # seconds
    top_k_experts: int = 2
    similarity_threshold: float = 0.3
    confidence_decay_factor: float = 0.95

    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.top_k_experts < 1:
            raise ValueError("top_k_experts must be >= 1")

        if not (0.0 <= self.similarity_threshold <= 1.0):
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")

        if not self.experts:
            raise ValueError("experts list cannot be empty")


@dataclass
class MoEGateConfig:
    """Configuration for Selective Gate component"""

    enabled: bool = True
    retrieve_sim_threshold: float = 0.62
    low_sim_threshold: float = 0.45
    k_min: int = 4
    k_max: int = 15
    default_top_k: int = 8
    high_score_cutoff: float = 0.8
    low_score_cutoff: float = 0.5
    confidence_weight: float = 0.7
    complexity_weight: float = 0.3
    adaptation_rate: float = 0.01

    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.k_min < 1 or self.k_max < self.k_min:
            raise ValueError("k_min must be >= 1 and k_max must be >= k_min")

        if not (0.0 <= self.retrieve_sim_threshold <= 1.0):
            raise ValueError("retrieve_sim_threshold must be between 0.0 and 1.0")

        if not (0.0 <= self.low_sim_threshold <= self.retrieve_sim_threshold):
            raise ValueError("low_sim_threshold must be <= retrieve_sim_threshold")


@dataclass
class MoERerankerConfig:
    """Configuration for Two-Stage Reranker component"""

    enabled: bool = True
    stage1_enabled: bool = True
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    max_rerank_candidates: int = 50
    batch_size: int = 16
    stage2_enabled: bool = False
    uncertainty_threshold: float = 0.15
    llm_temperature: float = 0.0
    max_llm_candidates: int = 10
    cache_enabled: bool = True
    cache_size: int = 1000
    early_termination_threshold: float = 0.8
    min_improvement_threshold: float = 0.05
    max_diversity_penalty: float = 0.1

    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        if self.max_rerank_candidates < 1:
            raise ValueError("max_rerank_candidates must be >= 1")

        if not (0.0 <= self.uncertainty_threshold <= 1.0):
            raise ValueError("uncertainty_threshold must be between 0.0 and 1.0")

        if self.cache_size < 0:
            raise ValueError("cache_size must be >= 0")


@dataclass
class MoEIntegrationConfig:
    """Configuration for MoE integration settings"""

    fallback_on_failure: bool = True
    error_threshold: float = 0.1
    performance_monitoring: bool = True
    health_check_interval: int = 60
    force_baseline_mode: bool = False

    def __post_init__(self):
        """Validate configuration after initialization"""
        if not (0.0 <= self.error_threshold <= 1.0):
            raise ValueError("error_threshold must be between 0.0 and 1.0")

        if self.health_check_interval < 1:
            raise ValueError("health_check_interval must be >= 1")


@dataclass
class MoEConfig:
    """Master configuration for Mixture of Experts system"""

    enabled: bool = False
    router: MoERouterConfig = field(default_factory=MoERouterConfig)
    gate: MoEGateConfig = field(default_factory=MoEGateConfig)
    reranker: MoERerankerConfig = field(default_factory=MoERerankerConfig)
    integration: MoEIntegrationConfig = field(default_factory=MoEIntegrationConfig)

    def __post_init__(self):
        """Validate master configuration"""
        # Ensure component configs are properly initialized
        if not isinstance(self.router, MoERouterConfig):
            self.router = MoERouterConfig(**self.router) if self.router else MoERouterConfig()

        if not isinstance(self.gate, MoEGateConfig):
            self.gate = MoEGateConfig(**self.gate) if self.gate else MoEGateConfig()

        if not isinstance(self.reranker, MoERerankerConfig):
            self.reranker = MoERerankerConfig(**self.reranker) if self.reranker else MoERerankerConfig()

        if not isinstance(self.integration, MoEIntegrationConfig):
            self.integration = MoEIntegrationConfig(**self.integration) if self.integration else MoEIntegrationConfig()

    def is_component_enabled(self, component: str) -> bool:
        """Check if a specific component is enabled"""
        if not self.enabled:
            return False

        component_map = {
            "router": self.router.enabled,
            "gate": self.gate.enabled,
            "reranker": self.reranker.enabled,
        }

        return component_map.get(component, False)

    def get_component_config(self, component: str) -> Optional[Any]:
        """Get configuration for a specific component"""
        component_configs = {
            "router": self.router,
            "gate": self.gate,
            "reranker": self.reranker,
            "integration": self.integration,
        }

        return component_configs.get(component)

    def update_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration from dictionary"""
        for key, value in config_dict.items():
            if hasattr(self, key):
                if key in ["router", "gate", "reranker", "integration"]:
                    # Handle nested config objects
                    current_config = getattr(self, key)
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            if hasattr(current_config, sub_key):
                                setattr(current_config, sub_key, sub_value)
                    else:
                        # Replace entire config object
                        config_class = type(current_config)
                        setattr(self, key, config_class(**value) if value else config_class())
                else:
                    # Handle simple attributes
                    setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "enabled": self.enabled,
            "router": {
                "enabled": self.router.enabled,
                "experts": self.router.experts,
                "centroid_refresh_interval": self.router.centroid_refresh_interval,
                "top_k_experts": self.router.top_k_experts,
                "similarity_threshold": self.router.similarity_threshold,
                "confidence_decay_factor": self.router.confidence_decay_factor,
            },
            "gate": {
                "enabled": self.gate.enabled,
                "retrieve_sim_threshold": self.gate.retrieve_sim_threshold,
                "low_sim_threshold": self.gate.low_sim_threshold,
                "k_min": self.gate.k_min,
                "k_max": self.gate.k_max,
                "default_top_k": self.gate.default_top_k,
                "high_score_cutoff": self.gate.high_score_cutoff,
                "low_score_cutoff": self.gate.low_score_cutoff,
                "confidence_weight": self.gate.confidence_weight,
                "complexity_weight": self.gate.complexity_weight,
                "adaptation_rate": self.gate.adaptation_rate,
            },
            "reranker": {
                "enabled": self.reranker.enabled,
                "stage1_enabled": self.reranker.stage1_enabled,
                "cross_encoder_model": self.reranker.cross_encoder_model,
                "max_rerank_candidates": self.reranker.max_rerank_candidates,
                "batch_size": self.reranker.batch_size,
                "stage2_enabled": self.reranker.stage2_enabled,
                "uncertainty_threshold": self.reranker.uncertainty_threshold,
                "llm_temperature": self.reranker.llm_temperature,
                "max_llm_candidates": self.reranker.max_llm_candidates,
                "cache_enabled": self.reranker.cache_enabled,
                "cache_size": self.reranker.cache_size,
                "early_termination_threshold": self.reranker.early_termination_threshold,
                "min_improvement_threshold": self.reranker.min_improvement_threshold,
                "max_diversity_penalty": self.reranker.max_diversity_penalty,
            },
            "integration": {
                "fallback_on_failure": self.integration.fallback_on_failure,
                "error_threshold": self.integration.error_threshold,
                "performance_monitoring": self.integration.performance_monitoring,
                "health_check_interval": self.integration.health_check_interval,
                "force_baseline_mode": self.integration.force_baseline_mode,
            },
        }


class MoEConfigManager:
    """Manager for MoE configuration with file I/O and validation"""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._find_config_path()
        self.config = MoEConfig()
        self._load_config()

    def _find_config_path(self) -> str:
        """Find configuration file path"""
        search_paths = [
            "config.yaml",
            "config.yml",
            "moe-config.yaml",
            "moe-config.yml",
            Path.home() / ".personal-rag" / "config.yaml",
        ]

        for path in search_paths:
            if Path(path).exists():
                return str(path)

        return "config.yaml"  # Default fallback

    def _load_config(self) -> None:
        """Load configuration from file"""
        config_path = Path(self.config_path)

        if not config_path.exists():
            logger.info(f"Configuration file not found at {config_path}, using defaults")
            return

        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f) or {}

            # Extract MoE configuration
            moe_data = config_data.get("moe", {})

            # Update configuration
            self.config.update_from_dict(moe_data)

            logger.info(f"Loaded MoE configuration from {config_path}")

        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {e}")
            logger.info("Using default MoE configuration")

    def save_config(self) -> bool:
        """Save current configuration to file"""
        try:
            config_path = Path(self.config_path)
            config_path.parent.mkdir(parents=True, exist_ok=True)

            # Load existing config or create new
            if config_path.exists():
                with open(config_path, 'r') as f:
                    existing_config = yaml.safe_load(f) or {}
            else:
                existing_config = {}

            # Update MoE section
            existing_config["moe"] = self.config.to_dict()

            # Save updated configuration
            with open(config_path, 'w') as f:
                yaml.dump(existing_config, f, default_flow_style=False, indent=2)

            logger.info(f"Saved MoE configuration to {config_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save configuration to {config_path}: {e}")
            return False

    def update_config(self, updates: Dict[str, Any]) -> bool:
        """Update configuration with validation"""
        try:
            # Create temporary config for validation
            temp_config = MoEConfig()
            temp_config.update_from_dict(self.config.to_dict())  # Copy current
            temp_config.update_from_dict(updates)

            # Validate by checking __post_init__ doesn't raise
            temp_config.__post_init__()

            # Update actual config
            self.config.update_from_dict(updates)

            # Save to file
            return self.save_config()

        except Exception as e:
            logger.error(f"Configuration update failed: {e}")
            return False

    def get_config(self) -> MoEConfig:
        """Get current configuration"""
        return self.config

    def reset_to_defaults(self) -> bool:
        """Reset configuration to defaults"""
        self.config = MoEConfig()
        return self.save_config()

    def validate_config(self) -> List[str]:
        """Validate current configuration and return any errors"""
        errors = []

        try:
            # This will trigger __post_init__ validation
            temp_config = MoEConfig()
            temp_config.update_from_dict(self.config.to_dict())
            temp_config.__post_init__()
        except ValueError as e:
            errors.append(str(e))
        except Exception as e:
            errors.append(f"Unexpected validation error: {e}")

        return errors


# Global configuration manager instance
_config_manager: Optional[MoEConfigManager] = None


def get_moe_config() -> MoEConfig:
    """Get the global MoE configuration"""
    global _config_manager

    if _config_manager is None:
        _config_manager = MoEConfigManager()

    return _config_manager.get_config()


def update_moe_config(updates: Dict[str, Any]) -> bool:
    """Update the global MoE configuration"""
    global _config_manager

    if _config_manager is None:
        _config_manager = MoEConfigManager()

    return _config_manager.update_config(updates)


def save_moe_config() -> bool:
    """Save the current MoE configuration to file"""
    global _config_manager

    if _config_manager is None:
        _config_manager = MoEConfigManager()

    return _config_manager.save_config()


def reload_moe_config() -> bool:
    """Reload MoE configuration from file"""
    global _config_manager

    _config_manager = MoEConfigManager()
    return True


# Environment variable integration
def load_from_env() -> MoEConfig:
    """Load MoE configuration from environment variables"""
    config = MoEConfig()

    # Master toggle
    config.enabled = os.getenv("MOE_ENABLED", "false").lower() == "true"

    # Router configuration
    config.router.enabled = os.getenv("MOE_ROUTER_ENABLED", "true").lower() == "true"
    config.router.top_k_experts = int(os.getenv("MOE_ROUTER_TOP_K", "2"))
    config.router.similarity_threshold = float(os.getenv("MOE_ROUTER_SIMILARITY_THRESHOLD", "0.3"))

    # Gate configuration
    config.gate.enabled = os.getenv("MOE_GATE_ENABLED", "true").lower() == "true"
    config.gate.retrieve_sim_threshold = float(os.getenv("MOE_GATE_RETRIEVE_THRESHOLD", "0.62"))
    config.gate.low_sim_threshold = float(os.getenv("MOE_GATE_LOW_SIM_THRESHOLD", "0.45"))
    config.gate.k_min = int(os.getenv("MOE_GATE_K_MIN", "4"))
    config.gate.k_max = int(os.getenv("MOE_GATE_K_MAX", "15"))
    config.gate.default_top_k = int(os.getenv("MOE_GATE_DEFAULT_TOP_K", "8"))

    # Reranker configuration
    config.reranker.enabled = os.getenv("MOE_RERANKER_ENABLED", "true").lower() == "true"
    config.reranker.stage1_enabled = os.getenv("MOE_RERANKER_STAGE1_ENABLED", "true").lower() == "true"
    config.reranker.stage2_enabled = os.getenv("MOE_RERANKER_STAGE2_ENABLED", "false").lower() == "true"
    config.reranker.uncertainty_threshold = float(os.getenv("MOE_RERANKER_UNCERTAINTY_THRESHOLD", "0.15"))
    config.reranker.batch_size = int(os.getenv("MOE_RERANKER_BATCH_SIZE", "16"))
    config.reranker.max_rerank_candidates = int(os.getenv("MOE_RERANKER_MAX_CANDIDATES", "50"))

    # Integration configuration
    config.integration.fallback_on_failure = os.getenv("MOE_FALLBACK_ON_FAILURE", "true").lower() == "true"
    config.integration.performance_monitoring = os.getenv("MOE_PERFORMANCE_MONITORING", "true").lower() == "true"
    config.integration.force_baseline_mode = os.getenv("MOE_FORCE_BASELINE", "false").lower() == "true"

    return config


# Utility functions for configuration management
def create_default_config_file(filepath: str = "config.yaml") -> bool:
    """Create a default configuration file"""
    config_manager = MoEConfigManager(filepath)
    config_manager.reset_to_defaults()
    return config_manager.save_config()


def print_config_summary(config: Optional[MoEConfig] = None) -> None:
    """Print a summary of the current MoE configuration"""
    if config is None:
        config = get_moe_config()

    print("=== MoE Configuration Summary ===")
    print(f"Enabled: {config.enabled}")

    if config.enabled:
        print(f"Router: {config.router.enabled} ({len(config.router.experts)} experts)")
        print(f"Gate: {config.gate.enabled} (threshold: {config.gate.retrieve_sim_threshold})")
        print(f"Reranker: {config.reranker.enabled} (stage2: {config.reranker.stage2_enabled})")

    print(f"Fallback on failure: {config.integration.fallback_on_failure}")
    print(f"Performance monitoring: {config.integration.performance_monitoring}")


if __name__ == "__main__":
    # CLI interface for configuration management
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "show":
            print_config_summary()
        elif command == "create-default":
            filepath = sys.argv[2] if len(sys.argv) > 2 else "config.yaml"
            if create_default_config_file(filepath):
                print(f"Created default configuration file: {filepath}")
            else:
                print("Failed to create configuration file")
        elif command == "validate":
            config = get_moe_config()
            errors = MoEConfigManager().validate_config()
            if errors:
                print("Configuration validation errors:")
                for error in errors:
                    print(f"  - {error}")
            else:
                print("Configuration is valid")
        else:
            print("Usage: python moe_config.py [show|create-default|validate]")
    else:
        print_config_summary()