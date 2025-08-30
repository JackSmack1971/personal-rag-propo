"""
Secure Configuration Management for Personal RAG Chatbot
Implements secure configuration loading, validation, and management with encryption support.

Author: SPARC Security Architect
Date: 2025-08-30
"""

import os
import json
import yaml
import hashlib
import secrets
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, asdict
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

from .security import log_security_event

@dataclass
class SecureConfig:
    """Secure configuration container with encryption support"""

    # Core Application Settings
    app_name: str = "Personal RAG Chatbot"
    app_version: str = "2.0.0"
    environment: str = "development"
    debug: bool = False

    # Security Settings
    enable_security: bool = True
    enable_authentication: bool = True
    enable_authorization: bool = True
    enable_rate_limiting: bool = True
    enable_input_validation: bool = True
    enable_audit_logging: bool = True

    # Authentication Settings
    jwt_secret_key: Optional[str] = None
    jwt_algorithm: str = "HS256"
    session_timeout_minutes: int = 30
    password_min_length: int = 12
    enable_mfa: bool = False

    # API Security
    openrouter_api_key: Optional[str] = None
    openrouter_model: str = "openrouter/auto"
    openrouter_referer: Optional[str] = None
    openrouter_title: str = "Personal RAG Chatbot"
    openrouter_max_tokens: int = 1000

    pinecone_api_key: Optional[str] = None
    pinecone_index: str = "personal-rag"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    pinecone_namespace: str = "default"

    # File Security
    max_file_size_mb: int = 10
    allowed_file_types: List[str] = None
    enable_file_scanning: bool = True
    quarantine_suspicious_files: bool = True

    # Network Security
    enable_https: bool = True
    enable_hsts: bool = True
    enable_csp: bool = True
    cors_allowed_origins: List[str] = None
    trusted_proxies: List[str] = None

    # Monitoring and Alerting
    enable_monitoring: bool = True
    alert_email_recipients: List[str] = None
    log_level: str = "INFO"
    security_log_level: str = "INFO"

    # Encryption Settings
    encryption_enabled: bool = True
    encryption_key: Optional[str] = None

    def __post_init__(self):
        if self.allowed_file_types is None:
            self.allowed_file_types = ['.pdf', '.txt', '.md']

        if self.cors_allowed_origins is None:
            self.cors_allowed_origins = ['http://localhost:7860', 'https://localhost:7860']

        if self.alert_email_recipients is None:
            self.alert_email_recipients = []

        if self.trusted_proxies is None:
            self.trusted_proxies = []

        # Generate encryption key if not provided
        if self.encryption_enabled and not self.encryption_key:
            self.encryption_key = self._generate_encryption_key()

        # Generate JWT secret if not provided
        if not self.jwt_secret_key:
            self.jwt_secret_key = secrets.token_hex(32)

class ConfigEncryption:
    """Configuration encryption and decryption"""

    @staticmethod
    def _derive_key(password: str, salt: bytes) -> bytes:
        """Derive encryption key from password"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return base64.urlsafe_b64encode(kdf.derive(password.encode()))

    @staticmethod
    def encrypt_sensitive_data(data: str, password: str) -> str:
        """Encrypt sensitive data"""
        salt = secrets.token_bytes(16)
        key = ConfigEncryption._derive_key(password, salt)
        f = Fernet(key)

        encrypted_data = f.encrypt(data.encode())
        # Store salt + encrypted data
        combined = salt + encrypted_data
        return base64.urlsafe_b64encode(combined).decode()

    @staticmethod
    def decrypt_sensitive_data(encrypted_data: str, password: str) -> str:
        """Decrypt sensitive data"""
        try:
            combined = base64.urlsafe_b64decode(encrypted_data.encode())
            salt = combined[:16]
            encrypted = combined[16:]

            key = ConfigEncryption._derive_key(password, salt)
            f = Fernet(key)

            return f.decrypt(encrypted).decode()
        except Exception as e:
            log_security_event("CONFIG_DECRYPTION_FAILED", {"error": str(e)}, "ERROR")
            raise ValueError("Failed to decrypt configuration data")

class SecureConfigManager:
    """Secure configuration management with validation and encryption"""

    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self.config_file = self.config_dir / "app_config.enc"
        self.backup_dir = self.config_dir / "backups"
        self.backup_dir.mkdir(exist_ok=True)

        # Master password for encryption (should be provided securely)
        self._master_password = os.getenv("CONFIG_MASTER_PASSWORD", "default_insecure_password")

    def load_config(self) -> SecureConfig:
        """Load and decrypt configuration"""

        if not self.config_file.exists():
            log_security_event("CONFIG_FILE_NOT_FOUND", {
                "config_file": str(self.config_file)
            }, "WARNING")
            return self._create_default_config()

        try:
            with open(self.config_file, 'r') as f:
                encrypted_data = f.read()

            decrypted_data = ConfigEncryption.decrypt_sensitive_data(
                encrypted_data, self._master_password
            )

            config_dict = json.loads(decrypted_data)
            config = SecureConfig(**config_dict)

            log_security_event("CONFIG_LOADED", {
                "environment": config.environment,
                "version": config.app_version
            }, "INFO")

            return config

        except Exception as e:
            log_security_event("CONFIG_LOAD_FAILED", {
                "error": str(e),
                "config_file": str(self.config_file)
            }, "ERROR")
            # Fallback to default config
            return self._create_default_config()

    def save_config(self, config: SecureConfig) -> bool:
        """Encrypt and save configuration"""

        try:
            # Create backup before saving
            self._create_backup()

            # Convert to dict and encrypt
            config_dict = asdict(config)
            config_json = json.dumps(config_dict, indent=2)

            encrypted_data = ConfigEncryption.encrypt_sensitive_data(
                config_json, self._master_password
            )

            # Write encrypted config
            with open(self.config_file, 'w') as f:
                f.write(encrypted_data)

            log_security_event("CONFIG_SAVED", {
                "environment": config.environment,
                "version": config.app_version
            }, "INFO")

            return True

        except Exception as e:
            log_security_event("CONFIG_SAVE_FAILED", {
                "error": str(e)
            }, "ERROR")
            return False

    def _create_default_config(self) -> SecureConfig:
        """Create default configuration"""

        config = SecureConfig()

        # Load sensitive values from environment
        config.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        config.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        config.openrouter_referer = os.getenv("OPENROUTER_REFERER", "http://localhost:7860")
        config.jwt_secret_key = os.getenv("JWT_SECRET_KEY", secrets.token_hex(32))

        log_security_event("DEFAULT_CONFIG_CREATED", {
            "environment": config.environment
        }, "INFO")

        return config

    def _create_backup(self):
        """Create backup of current configuration"""

        if self.config_file.exists():
            timestamp = int(os.path.getmtime(self.config_file))
            backup_file = self.backup_dir / f"config_backup_{timestamp}.enc"

            try:
                import shutil
                shutil.copy2(self.config_file, backup_file)
                log_security_event("CONFIG_BACKUP_CREATED", {
                    "backup_file": str(backup_file)
                }, "INFO")
            except Exception as e:
                log_security_event("CONFIG_BACKUP_FAILED", {
                    "error": str(e)
                }, "WARNING")

    def validate_config(self, config: SecureConfig) -> List[str]:
        """Validate configuration for security and correctness"""

        issues = []

        # Check required fields
        if not config.openrouter_api_key:
            issues.append("OpenRouter API key is required")

        if not config.pinecone_api_key:
            issues.append("Pinecone API key is required")

        # Validate JWT secret
        if not config.jwt_secret_key or len(config.jwt_secret_key) < 32:
            issues.append("JWT secret key must be at least 32 characters")

        # Validate file size limits
        if config.max_file_size_mb > 100:
            issues.append("Maximum file size should not exceed 100MB")

        if config.max_file_size_mb < 1:
            issues.append("Maximum file size should be at least 1MB")

        # Validate session timeout
        if config.session_timeout_minutes > 480:  # 8 hours
            issues.append("Session timeout should not exceed 8 hours")

        if config.session_timeout_minutes < 5:
            issues.append("Session timeout should be at least 5 minutes")

        # Validate CORS origins
        for origin in config.cors_allowed_origins:
            if not origin.startswith(('http://', 'https://')):
                issues.append(f"Invalid CORS origin format: {origin}")

        # Log validation results
        if issues:
            log_security_event("CONFIG_VALIDATION_FAILED", {
                "issues_count": len(issues),
                "issues": issues
            }, "WARNING")
        else:
            log_security_event("CONFIG_VALIDATION_PASSED", {}, "INFO")

        return issues

    def update_config_value(self, key: str, value: Any) -> bool:
        """Securely update a single configuration value"""

        try:
            config = self.load_config()

            # Validate the key exists
            if not hasattr(config, key):
                log_security_event("CONFIG_UPDATE_INVALID_KEY", {
                    "key": key
                }, "WARNING")
                return False

            # Update the value
            setattr(config, key, value)

            # Validate the updated config
            issues = self.validate_config(config)
            if issues:
                log_security_event("CONFIG_UPDATE_VALIDATION_FAILED", {
                    "key": key,
                    "issues": issues
                }, "WARNING")
                return False

            # Save the updated config
            return self.save_config(config)

        except Exception as e:
            log_security_event("CONFIG_UPDATE_FAILED", {
                "key": key,
                "error": str(e)
            }, "ERROR")
            return False

    def get_config_hash(self) -> str:
        """Get hash of current configuration for integrity checking"""

        if not self.config_file.exists():
            return ""

        try:
            with open(self.config_file, 'rb') as f:
                content = f.read()
                return hashlib.sha256(content).hexdigest()
        except Exception as e:
            log_security_event("CONFIG_HASH_FAILED", {"error": str(e)}, "ERROR")
            return ""

class EnvironmentConfigLoader:
    """Load configuration from environment variables with validation"""

    @staticmethod
    def load_from_environment() -> SecureConfig:
        """Load configuration from environment variables"""

        config = SecureConfig()

        # Load core settings
        config.environment = os.getenv("ENVIRONMENT", "development")
        config.debug = os.getenv("DEBUG", "false").lower() == "true"

        # Load security settings
        config.enable_authentication = os.getenv("ENABLE_AUTHENTICATION", "true").lower() == "true"
        config.enable_authorization = os.getenv("ENABLE_AUTHORIZATION", "true").lower() == "true"
        config.enable_rate_limiting = os.getenv("ENABLE_RATE_LIMITING", "true").lower() == "true"

        # Load API keys
        config.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        config.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        config.openrouter_referer = os.getenv("OPENROUTER_REFERER")
        config.openrouter_title = os.getenv("OPENROUTER_TITLE", "Personal RAG Chatbot")

        # Load JWT settings
        config.jwt_secret_key = os.getenv("JWT_SECRET_KEY", secrets.token_hex(32))

        # Load file security settings
        config.max_file_size_mb = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
        allowed_types = os.getenv("ALLOWED_FILE_TYPES", ".pdf,.txt,.md")
        config.allowed_file_types = [ext.strip() for ext in allowed_types.split(",")]

        # Load network security settings
        config.enable_https = os.getenv("ENABLE_HTTPS", "true").lower() == "true"
        config.enable_hsts = os.getenv("ENABLE_HSTS", "true").lower() == "true"
        config.enable_csp = os.getenv("ENABLE_CSP", "true").lower() == "true"

        cors_origins = os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:7860,https://localhost:7860")
        config.cors_allowed_origins = [origin.strip() for origin in cors_origins.split(",")]

        # Load monitoring settings
        config.enable_monitoring = os.getenv("ENABLE_MONITORING", "true").lower() == "true"
        config.log_level = os.getenv("LOG_LEVEL", "INFO")
        config.security_log_level = os.getenv("SECURITY_LOG_LEVEL", "INFO")

        log_security_event("ENV_CONFIG_LOADED", {
            "environment": config.environment,
            "debug": config.debug
        }, "INFO")

        return config

# Global configuration manager instance
config_manager = SecureConfigManager()

def load_secure_config() -> SecureConfig:
    """Load secure configuration with fallback to environment"""
    try:
        config = config_manager.load_config()
        issues = config_manager.validate_config(config)

        if issues:
            log_security_event("CONFIG_ISSUES_FOUND", {
                "issues": issues
            }, "WARNING")

        return config
    except Exception as e:
        log_security_event("CONFIG_LOAD_FALLBACK", {
            "error": str(e)
        }, "WARNING")
        return EnvironmentConfigLoader.load_from_environment()

def save_secure_config(config: SecureConfig) -> bool:
    """Save secure configuration"""
    issues = config_manager.validate_config(config)
    if issues:
        log_security_event("CONFIG_SAVE_BLOCKED", {
            "issues": issues
        }, "ERROR")
        return False

    return config_manager.save_config(config)

def get_config_hash() -> str:
    """Get configuration integrity hash"""
    return config_manager.get_config_hash()