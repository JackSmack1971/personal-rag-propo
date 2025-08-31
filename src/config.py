import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path
import yaml
import logging

logger = logging.getLogger(__name__)

@dataclass
class AppConfig:
    # Core API Configuration (required fields first)
    OPENROUTER_API_KEY: str
    OPENROUTER_MODEL: str
    OPENROUTER_REFERER: str
    OPENROUTER_TITLE: str

    # Pinecone Configuration (v7.x)
    PINECONE_API_KEY: str
    PINECONE_INDEX: str
    PINECONE_CLOUD: str
    PINECONE_REGION: str

    # Embedding Configuration
    EMBED_MODEL: str
    NAMESPACE: str
    TOP_K: int

    # ---- 2025 Stack Configuration ----
    # OpenRouter settings
    OPENROUTER_MAX_TOKENS: int = 1000

    # Pinecone settings
    PINECONE_GRPC_ENABLED: bool = True

    # Sentence-Transformers backend (torch/onnx/openvino)
    SENTENCE_TRANSFORMERS_BACKEND: str = "torch"

    # Gradio 5.x settings
    GRADIO_ANALYTICS_ENABLED: bool = False
    GRADIO_SSR_ENABLED: bool = True
    GRADIO_AUTH_ENABLED: bool = False
    GRADIO_THEME: str = "soft"
    GRADIO_SERVER_NAME: str = "0.0.0.0"
    GRADIO_SERVER_PORT: int = 7860

    # MoE Configuration (2025 features)
    moe_enabled: bool = False
    moe_config: Optional[Dict[str, Any]] = None

    # Security settings
    TRUST_REMOTE_CODE: bool = False
    MAX_FILE_SIZE_MB: int = 10
    ALLOWED_FILE_TYPES: list = field(default_factory=lambda: [".pdf", ".txt", ".md"])

    # Enhanced Security Configuration (OWASP LLM Top 10 2025)
    ENABLE_SECURITY_HEADERS: bool = True
    ENABLE_CORS_PROTECTION: bool = True
    CORS_ALLOWED_ORIGINS: list = field(default_factory=lambda: ["http://localhost:7860", "https://localhost:7860"])
    ENABLE_RATE_LIMITING: bool = True
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = 60
    RATE_LIMIT_BURST_LIMIT: int = 100
    ENABLE_INPUT_VALIDATION: bool = True
    ENABLE_CONTENT_FILTERING: bool = True
    LOG_SECURITY_EVENTS: bool = True

    # Authentication & Authorization
    ENABLE_AUTH: bool = False
    JWT_SECRET_KEY: Optional[str] = None
    JWT_ALGORITHM: str = "HS256"
    SESSION_TIMEOUT_MINUTES: int = 30
    MAX_LOGIN_ATTEMPTS: int = 5
    LOCKOUT_DURATION_MINUTES: int = 15

    # API Security
    API_KEY_ROTATION_DAYS: int = 30
    API_REQUEST_TIMEOUT: int = 60
    API_MAX_RETRIES: int = 3
    API_RATE_LIMIT_PER_SERVICE: Dict[str, int] = field(default_factory=lambda: {
        "openrouter": 60,
        "pinecone": 30
    })

    # Network Security
    ENABLE_HTTPS: bool = True
    SSL_CERT_FILE: Optional[str] = None
    SSL_KEY_FILE: Optional[str] = None
    ENABLE_HSTS: bool = True
    HSTS_MAX_AGE: int = 31536000  # 1 year
    TRUSTED_PROXIES: list = field(default_factory=list)

    # Incident Response
    ENABLE_INCIDENT_DETECTION: bool = True
    INCIDENT_ALERT_EMAILS: list = field(default_factory=list)
    INCIDENT_SLACK_WEBHOOK: Optional[str] = None
    INCIDENT_RESPONSE_TIMEOUT: int = 300  # 5 minutes

    # Audit Logging
    AUDIT_LOG_RETENTION_DAYS: int = 90
    AUDIT_LOG_ENCRYPTION: bool = False
    AUDIT_LOG_INTEGRITY_CHECK: bool = True

    # Compliance
    COMPLIANCE_MODE: str = "standard"  # standard, strict, relaxed
    GDPR_COMPLIANCE: bool = True
    DATA_RETENTION_DAYS: int = 2555  # 7 years for GDPR
    ANONYMIZE_LOGS: bool = False

    # Performance settings
    ENABLE_SPARSE_ENCODING: bool = False
    CACHE_EMBEDDINGS: bool = True
    MAX_CONTEXT_LENGTH: int = 8192
    BATCH_SIZE: int = 32

    # Monitoring and logging
    ENABLE_PERFORMANCE_MONITORING: bool = True
    LOG_LEVEL: str = "INFO"

    @classmethod
    def from_env(cls):
        """Load configuration from environment variables and YAML config file"""
        # Try to load from YAML config first
        config_path = os.getenv("CONFIG_PATH", "config.yaml")
        yaml_config = cls._load_yaml_config(config_path)

        # Load MoE configuration if available
        moe_enabled = cls._get_bool_env("MOE_ENABLED", yaml_config, False)
        if not moe_enabled and yaml_config.get("moe", {}).get("enabled", False):
            moe_enabled = True

        moe_config = None
        if moe_enabled:
            try:
                from .moe.config import get_moe_config
                moe_config_obj = get_moe_config()
                moe_config = moe_config_obj.to_dict()
            except Exception as e:
                logger.warning(f"Could not load MoE configuration: {e}")
                moe_enabled = False

        return cls(
            # Core API Configuration
            OPENROUTER_API_KEY=cls._get_env("OPENROUTER_API_KEY", yaml_config, ""),
            OPENROUTER_MODEL=cls._get_env("OPENROUTER_MODEL", yaml_config, "openrouter/auto"),
            OPENROUTER_REFERER=cls._get_env("OPENROUTER_REFERER", yaml_config, "http://localhost:7860"),
            OPENROUTER_TITLE=cls._get_env("OPENROUTER_TITLE", yaml_config, "Personal RAG (Enhanced 2025)"),

            # Pinecone Configuration
            PINECONE_API_KEY=cls._get_env("PINECONE_API_KEY", yaml_config, ""),
            PINECONE_INDEX=cls._get_env("PINECONE_INDEX", yaml_config, "personal-rag"),
            PINECONE_CLOUD=cls._get_env("PINECONE_CLOUD", yaml_config, "aws"),
            PINECONE_REGION=cls._get_env("PINECONE_REGION", yaml_config, "us-east-1"),

            # Embedding Configuration
            EMBED_MODEL=cls._get_env("EMBED_MODEL", yaml_config, "BAAI/bge-small-en-v1.5"),
            NAMESPACE=cls._get_env("NAMESPACE", yaml_config, "default"),
            TOP_K=int(cls._get_env("TOP_K", yaml_config, "6")),

            # 2025 Stack Configuration
            OPENROUTER_MAX_TOKENS=int(cls._get_env("OPENROUTER_MAX_TOKENS", yaml_config, "1000")),
            PINECONE_GRPC_ENABLED=cls._get_bool_env("PINECONE_GRPC_ENABLED", yaml_config, True),
            SENTENCE_TRANSFORMERS_BACKEND=cls._get_env("SENTENCE_TRANSFORMERS_BACKEND", yaml_config, "torch"),

            # Gradio 5.x settings
            GRADIO_ANALYTICS_ENABLED=cls._get_bool_env("GRADIO_ANALYTICS_ENABLED", yaml_config, False),
            GRADIO_SSR_ENABLED=cls._get_bool_env("GRADIO_SSR_ENABLED", yaml_config, True),
            GRADIO_AUTH_ENABLED=cls._get_bool_env("GRADIO_AUTH_ENABLED", yaml_config, False),
            GRADIO_THEME=cls._get_env("GRADIO_THEME", yaml_config, "soft"),
            GRADIO_SERVER_NAME=cls._get_env("GRADIO_SERVER_NAME", yaml_config, "0.0.0.0"),
            GRADIO_SERVER_PORT=int(cls._get_env("GRADIO_SERVER_PORT", yaml_config, "7860")),

            # MoE Configuration
            moe_enabled=moe_enabled,
            moe_config=moe_config,

            # Security settings
            TRUST_REMOTE_CODE=cls._get_bool_env("TRUST_REMOTE_CODE", yaml_config, False),
            MAX_FILE_SIZE_MB=int(cls._get_env("MAX_FILE_SIZE_MB", yaml_config, "10")),
            ALLOWED_FILE_TYPES=yaml_config.get("security", {}).get("allowed_file_types", [".pdf", ".txt", ".md"]),

            # Enhanced Security Configuration
            ENABLE_SECURITY_HEADERS=cls._get_bool_env("ENABLE_SECURITY_HEADERS", yaml_config, True),
            ENABLE_CORS_PROTECTION=cls._get_bool_env("ENABLE_CORS_PROTECTION", yaml_config, True),
            CORS_ALLOWED_ORIGINS=yaml_config.get("security", {}).get("cors_allowed_origins", ["http://localhost:7860", "https://localhost:7860"]),
            ENABLE_RATE_LIMITING=cls._get_bool_env("ENABLE_RATE_LIMITING", yaml_config, True),
            RATE_LIMIT_REQUESTS_PER_MINUTE=int(cls._get_env("RATE_LIMIT_REQUESTS_PER_MINUTE", yaml_config, "60")),
            RATE_LIMIT_BURST_LIMIT=int(cls._get_env("RATE_LIMIT_BURST_LIMIT", yaml_config, "100")),
            ENABLE_INPUT_VALIDATION=cls._get_bool_env("ENABLE_INPUT_VALIDATION", yaml_config, True),
            ENABLE_CONTENT_FILTERING=cls._get_bool_env("ENABLE_CONTENT_FILTERING", yaml_config, True),
            LOG_SECURITY_EVENTS=cls._get_bool_env("LOG_SECURITY_EVENTS", yaml_config, True),

            # Authentication & Authorization
            ENABLE_AUTH=cls._get_bool_env("ENABLE_AUTH", yaml_config, False),
            JWT_SECRET_KEY=os.getenv("JWT_SECRET_KEY"),
            JWT_ALGORITHM=cls._get_env("JWT_ALGORITHM", yaml_config, "HS256"),
            SESSION_TIMEOUT_MINUTES=int(cls._get_env("SESSION_TIMEOUT_MINUTES", yaml_config, "30")),
            MAX_LOGIN_ATTEMPTS=int(cls._get_env("MAX_LOGIN_ATTEMPTS", yaml_config, "5")),
            LOCKOUT_DURATION_MINUTES=int(cls._get_env("LOCKOUT_DURATION_MINUTES", yaml_config, "15")),

            # API Security
            API_KEY_ROTATION_DAYS=int(cls._get_env("API_KEY_ROTATION_DAYS", yaml_config, "30")),
            API_REQUEST_TIMEOUT=int(cls._get_env("API_REQUEST_TIMEOUT", yaml_config, "60")),
            API_MAX_RETRIES=int(cls._get_env("API_MAX_RETRIES", yaml_config, "3")),
            API_RATE_LIMIT_PER_SERVICE=yaml_config.get("api", {}).get("rate_limits", {
                "openrouter": 60,
                "pinecone": 30
            }),

            # Network Security
            ENABLE_HTTPS=cls._get_bool_env("ENABLE_HTTPS", yaml_config, True),
            SSL_CERT_FILE=os.getenv("SSL_CERT_FILE"),
            SSL_KEY_FILE=os.getenv("SSL_KEY_FILE"),
            ENABLE_HSTS=cls._get_bool_env("ENABLE_HSTS", yaml_config, True),
            HSTS_MAX_AGE=int(cls._get_env("HSTS_MAX_AGE", yaml_config, "31536000")),
            TRUSTED_PROXIES=yaml_config.get("network", {}).get("trusted_proxies", []),

            # Incident Response
            ENABLE_INCIDENT_DETECTION=cls._get_bool_env("ENABLE_INCIDENT_DETECTION", yaml_config, True),
            INCIDENT_ALERT_EMAILS=os.getenv("INCIDENT_ALERT_EMAILS", "").split(",") if os.getenv("INCIDENT_ALERT_EMAILS") else [],
            INCIDENT_SLACK_WEBHOOK=os.getenv("INCIDENT_SLACK_WEBHOOK"),
            INCIDENT_RESPONSE_TIMEOUT=int(cls._get_env("INCIDENT_RESPONSE_TIMEOUT", yaml_config, "300")),

            # Audit Logging
            AUDIT_LOG_RETENTION_DAYS=int(cls._get_env("AUDIT_LOG_RETENTION_DAYS", yaml_config, "90")),
            AUDIT_LOG_ENCRYPTION=cls._get_bool_env("AUDIT_LOG_ENCRYPTION", yaml_config, False),
            AUDIT_LOG_INTEGRITY_CHECK=cls._get_bool_env("AUDIT_LOG_INTEGRITY_CHECK", yaml_config, True),

            # Compliance
            COMPLIANCE_MODE=cls._get_env("COMPLIANCE_MODE", yaml_config, "standard"),
            GDPR_COMPLIANCE=cls._get_bool_env("GDPR_COMPLIANCE", yaml_config, True),
            DATA_RETENTION_DAYS=int(cls._get_env("DATA_RETENTION_DAYS", yaml_config, "2555")),
            ANONYMIZE_LOGS=cls._get_bool_env("ANONYMIZE_LOGS", yaml_config, False),

            # Performance settings
            ENABLE_SPARSE_ENCODING=cls._get_bool_env("ENABLE_SPARSE_ENCODING", yaml_config, False),
            CACHE_EMBEDDINGS=cls._get_bool_env("CACHE_EMBEDDINGS", yaml_config, True),
            MAX_CONTEXT_LENGTH=int(cls._get_env("MAX_CONTEXT_LENGTH", yaml_config, "8192")),
            BATCH_SIZE=int(cls._get_env("BATCH_SIZE", yaml_config, "32")),

            # Monitoring and logging
            ENABLE_PERFORMANCE_MONITORING=cls._get_bool_env("ENABLE_PERFORMANCE_MONITORING", yaml_config, True),
            LOG_LEVEL=cls._get_env("LOG_LEVEL", yaml_config, "INFO"),
        )

    @classmethod
    def _load_yaml_config(cls, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not Path(config_path).exists():
            return {}

        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning(f"Failed to load YAML config from {config_path}: {e}")
            return {}

    @classmethod
    def _get_env(cls, key: str, yaml_config: Dict[str, Any], default: str) -> str:
        """Get value from environment or YAML config"""
        return os.getenv(key, yaml_config.get(key.lower().replace("_", "_"), default))

    @classmethod
    def _get_bool_env(cls, key: str, yaml_config: Dict[str, Any], default: bool) -> bool:
        """Get boolean value from environment or YAML config"""
        env_value = os.getenv(key)
        if env_value is not None:
            return env_value.lower() in ("true", "1", "yes", "on")

        yaml_value = yaml_config.get(key.lower().replace("_", "_"))
        if yaml_value is not None:
            return bool(yaml_value)

        return default
