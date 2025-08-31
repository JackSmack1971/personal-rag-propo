"""
Security Module for Personal RAG Chatbot
Implements comprehensive security controls including input validation,
sanitization, rate limiting, and security monitoring.

Author: SPARC Security Architect
Date: 2025-08-30
"""

import os
import re
import hashlib
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
from urllib.parse import urlparse
import json
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class SecurityConfig:
    """Security configuration settings"""
    max_file_size_mb: int = 10
    allowed_file_types: Optional[List[str]] = None
    rate_limit_requests_per_minute: int = 60
    rate_limit_burst_limit: int = 100
    max_query_length: int = 2000
    max_response_length: int = 10000
    enable_security_headers: bool = True
    enable_cors_protection: bool = True
    cors_allowed_origins: Optional[List[str]] = None
    enable_rate_limiting: bool = True
    enable_input_validation: bool = True
    enable_content_filtering: bool = True
    log_security_events: bool = True

    def __post_init__(self):
        if self.allowed_file_types is None:
            self.allowed_file_types = ['.pdf', '.txt', '.md']
        if self.cors_allowed_origins is None:
            self.cors_allowed_origins = ['http://localhost:7860', 'https://localhost:7860']

class RateLimiter:
    """Advanced rate limiting with sliding window"""

    def __init__(self, requests_per_minute: int = 60, burst_limit: int = 100):
        self.requests_per_minute = requests_per_minute
        self.burst_limit = burst_limit
        self.requests: Dict[str, List[float]] = {}
        self.bans: Dict[str, float] = {}

    def is_rate_limited(self, identifier: str) -> bool:
        """Check if identifier is rate limited"""
        current_time = time.time()

        # Check if banned
        if identifier in self.bans:
            if current_time < self.bans[identifier]:
                return True
            else:
                del self.bans[identifier]

        # Clean old requests
        if identifier in self.requests:
            cutoff_time = current_time - 60  # 1 minute window
            self.requests[identifier] = [
                req_time for req_time in self.requests[identifier]
                if req_time > cutoff_time
            ]

        # Check rate limit
        if identifier not in self.requests:
            self.requests[identifier] = []

        if len(self.requests[identifier]) >= self.requests_per_minute:
            # Implement progressive ban
            ban_duration = min(len(self.requests[identifier]) - self.requests_per_minute + 1, 300)  # Max 5 min ban
            self.bans[identifier] = current_time + ban_duration
            logger.warning(f"Rate limit exceeded for {identifier}, banned for {ban_duration}s")
            return True

        # Check burst limit
        recent_requests = [req for req in self.requests[identifier] if current_time - req < 1]
        if len(recent_requests) >= self.burst_limit:
            return True

        return False

    def record_request(self, identifier: str) -> None:
        """Record a request for rate limiting"""
        current_time = time.time()
        if identifier not in self.requests:
            self.requests[identifier] = []
        self.requests[identifier].append(current_time)

class InputValidator:
    """Comprehensive input validation and sanitization"""

    def __init__(self, config: SecurityConfig):
        self.config = config
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for validation"""
        # Dangerous patterns to detect (OWASP LLM Top 10 2025)
        self.dangerous_patterns = [
            re.compile(r'<script[^>]*>.*?</script>', re.IGNORECASE | re.DOTALL),
            re.compile(r'javascript:', re.IGNORECASE),
            re.compile(r'data:', re.IGNORECASE),
            re.compile(r'vbscript:', re.IGNORECASE),
            re.compile(r'on\w+\s*=', re.IGNORECASE),
            re.compile(r'eval\s*\(', re.IGNORECASE),
            re.compile(r'exec\s*\(', re.IGNORECASE),
            re.compile(r'__import__\s*\(', re.IGNORECASE),
            re.compile(r'open\s*\(', re.IGNORECASE),
            re.compile(r'file://', re.IGNORECASE),
            re.compile(r'\\x[0-9a-fA-F]{2}', re.IGNORECASE),  # Hex encoding
            re.compile(r'\\u[0-9a-fA-F]{4}', re.IGNORECASE),  # Unicode encoding
            # Additional LLM-specific patterns
            re.compile(r'ignore\s+(all\s+)?previous\s+instructions', re.IGNORECASE),
            re.compile(r'system\s+prompt[:\s]', re.IGNORECASE),
            re.compile(r'you\s+are\s+now\s+', re.IGNORECASE),
            re.compile(r'forget\s+your\s+', re.IGNORECASE),
            re.compile(r'override\s+(your\s+)?', re.IGNORECASE),
            re.compile(r'bypass\s+(your\s+)?', re.IGNORECASE),
            re.compile(r'do\s+not\s+follow', re.IGNORECASE),
            re.compile(r'jailbreak', re.IGNORECASE),
            re.compile(r'dan\s+mode', re.IGNORECASE),
        ]

        # SQL injection patterns (enhanced)
        self.sql_patterns = [
            re.compile(r';\s*(select|insert|update|delete|drop|create|alter)', re.IGNORECASE),
            re.compile(r'union\s+select', re.IGNORECASE),
            re.compile(r'--|#|/\*|\*/', re.IGNORECASE),
            re.compile(r'1=1\s*--', re.IGNORECASE),
            re.compile(r'1=1\s*#', re.IGNORECASE),
            re.compile(r'or\s+1=1', re.IGNORECASE),
            re.compile(r'and\s+1=1', re.IGNORECASE),
        ]

        # Path traversal patterns (enhanced)
        self.path_patterns = [
            re.compile(r'\.\./|\.\.\\'),
            re.compile(r'~'),
            re.compile(r'\\'),
            re.compile(r'%2e%2e%2f', re.IGNORECASE),  # URL encoded ../
            re.compile(r'%2e%2e/', re.IGNORECASE),     # URL encoded ../
            re.compile(r'\.\.%2f', re.IGNORECASE),     # Mixed encoding
        ]

        # File-based attack patterns
        self.file_attack_patterns = [
            re.compile(r'\.exe|\.bat|\.cmd|\.scr|\.pif|\.com', re.IGNORECASE),
            re.compile(r'\.vbs|\.js|\.jar|\.hta', re.IGNORECASE),
            re.compile(r'\.zip|\.rar|\.7z', re.IGNORECASE),  # Archive files that could contain malware
        ]

    def validate_file_upload(self, file_path: str, file_content: bytes) -> Tuple[bool, str]:
        """Validate file upload with comprehensive checks"""

        # Check file extension
        file_ext = Path(file_path).suffix.lower()
        allowed_types = self.config.allowed_file_types or ['.pdf', '.txt', '.md']
        if file_ext not in allowed_types:
            return False, f"File type {file_ext} not allowed. Only {', '.join(allowed_types)} files are permitted."

        # Check file size
        file_size_mb = len(file_content) / (1024 * 1024)
        if file_size_mb > self.config.max_file_size_mb:
            return False, f"File size {file_size_mb:.1f}MB exceeds limit of {self.config.max_file_size_mb}MB"

        # Check for file-based attacks
        if any(pattern.search(file_path) for pattern in self.file_attack_patterns):
            return False, "File name contains potentially malicious patterns"

        # Check for malicious content
        content_str = file_content.decode('utf-8', errors='ignore')

        # Check for dangerous patterns
        for pattern in self.dangerous_patterns:
            if pattern.search(content_str):
                return False, "File contains potentially malicious content (dangerous patterns detected)"

        # Check for SQL injection patterns
        for pattern in self.sql_patterns:
            if pattern.search(content_str):
                return False, "File contains potentially malicious SQL injection patterns"

        # Check for path traversal
        for pattern in self.path_patterns:
            if pattern.search(file_path) or pattern.search(content_str):
                return False, "File contains path traversal attempts"

        # Check for file attack patterns in content
        for pattern in self.file_attack_patterns:
            if pattern.search(content_str):
                return False, "File contains references to potentially malicious file types"

        # Entropy analysis for obfuscated content
        if self._check_high_entropy(content_str):
            return False, "File contains potentially obfuscated or encrypted content"

        # Additional content validation based on file type
        if file_ext == '.pdf':
            return self._validate_pdf_content(file_content)
        elif file_ext in ['.txt', '.md']:
            return self._validate_text_content(content_str)

        return True, "File validation passed"

    def _validate_pdf_content(self, content: bytes) -> Tuple[bool, str]:
        """Validate PDF file content"""
        # Check PDF header
        if not content.startswith(b'%PDF-'):
            return False, "Invalid PDF file format"

        # Check for embedded scripts or dangerous content
        content_str = content.decode('latin-1', errors='ignore')
        if any(pattern.search(content_str) for pattern in self.dangerous_patterns):
            return False, "PDF contains potentially malicious content"

        return True, "PDF validation passed"

    def _validate_text_content(self, content: str) -> Tuple[bool, str]:
        """Validate text file content"""
        # Check content length
        if len(content) > 10 * 1024 * 1024:  # 10MB text limit
            return False, "Text content too large"

        # Check for excessive special characters (potential obfuscation)
        special_chars = sum(1 for c in content if not c.isalnum() and not c.isspace())
        if special_chars / len(content) > 0.5:
            return False, "Content contains excessive special characters"

        return True, "Text validation passed"

    def validate_query(self, query: str) -> Tuple[bool, str]:
        """Validate user query"""

        # Check query length
        if len(query) > self.config.max_query_length:
            return False, f"Query too long ({len(query)} chars, max {self.config.max_query_length})"

        # Check for dangerous patterns
        for pattern in self.dangerous_patterns:
            if pattern.search(query):
                return False, "Query contains potentially malicious content"

        # Check for prompt injection attempts
        injection_indicators = [
            "ignore previous instructions",
            "system prompt",
            "you are now",
            "forget your",
            "override",
            "bypass",
        ]

        query_lower = query.lower()
        for indicator in injection_indicators:
            if indicator in query_lower:
                return False, "Query contains potential prompt injection attempt"

        return True, "Query validation passed"

    def sanitize_input(self, input_str: str) -> str:
        """Sanitize input by removing dangerous content"""
        if not input_str:
            return input_str

        # Remove script tags
        input_str = re.sub(r'<script[^>]*>.*?</script>', '', input_str, flags=re.IGNORECASE | re.DOTALL)

        # Remove javascript: URLs
        input_str = re.sub(r'javascript:', '', input_str, flags=re.IGNORECASE)

        # Remove event handlers
        input_str = re.sub(r'on\w+\s*=', '', input_str, flags=re.IGNORECASE)

        # Remove dangerous function calls
        dangerous_functions = ['eval', 'exec', '__import__', 'open', 'file']
        for func in dangerous_functions:
            input_str = re.sub(rf'\b{func}\s*\(', f'_{func}_disabled(', input_str, flags=re.IGNORECASE)

        return input_str

    def _check_high_entropy(self, content: str, threshold: float = 0.7) -> bool:
        """Check if content has high entropy (potential obfuscation)"""
        if len(content) < 100:  # Skip short content
            return False

        # Calculate character frequency
        char_freq = {}
        for char in content:
            char_freq[char] = char_freq.get(char, 0) + 1

        # Calculate entropy
        entropy = 0
        content_len = len(content)
        for count in char_freq.values():
            probability = count / content_len
            if probability > 0:
                entropy -= probability * (probability.bit_length() - 1)  # Approximation

        # Normalize entropy (0-1 scale)
        max_entropy = len(char_freq)  # Maximum possible entropy
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

        return normalized_entropy > threshold

class SecurityAuditor:
    """Security event auditing and logging"""

    def __init__(self, config: SecurityConfig):
        self.config = config
        self.audit_log_file = Path("logs/security_audit.log")
        self.audit_log_file.parent.mkdir(exist_ok=True)

    def log_security_event(self, event_type: str, details: Dict[str, Any],
                          severity: str = "INFO", user_id: Optional[str] = None) -> None:
        """Log security event with comprehensive details"""

        if not self.config.log_security_events:
            return

        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "severity": severity,
            "user_id": user_id or "anonymous",
            "details": details,
            "system_context": self._get_system_context(),
            "integrity_hash": None
        }

        # Create integrity hash
        entry_str = json.dumps(audit_entry, sort_keys=True, default=str)
        audit_entry["integrity_hash"] = hashlib.sha256(entry_str.encode()).hexdigest()

        # Write to log file
        try:
            with open(self.audit_log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(audit_entry, default=str) + '\n')
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")

        # Log to application logger
        log_level = getattr(logging, severity.upper(), logging.INFO)
        logger.log(log_level, f"Security Event: {event_type} - {details}")

    def _get_system_context(self) -> Dict[str, Any]:
        """Get system context for audit logging"""
        try:
            import platform
            hostname = platform.node()
        except:
            hostname = "unknown"

        return {
            "hostname": hostname,
            "process_id": os.getpid(),
            "user": os.getenv("USER") or os.getenv("USERNAME") or "unknown",
            "working_directory": str(Path.cwd())
        }

class SecurityMonitor:
    """Real-time security monitoring and alerting"""

    def __init__(self, config: SecurityConfig):
        self.config = config
        self.alert_thresholds = {
            "failed_validations": 10,
            "rate_limit_hits": 50,
            "suspicious_patterns": 5,
            "file_upload_rejections": 20
        }
        self.metrics = {
            "failed_validations": 0,
            "rate_limit_hits": 0,
            "suspicious_patterns": 0,
            "file_upload_rejections": 0,
            "total_requests": 0,
            "blocked_requests": 0
        }
        self.last_reset = time.time()

    def record_metric(self, metric_name: str, value: int = 1) -> None:
        """Record security metric"""
        if metric_name in self.metrics:
            self.metrics[metric_name] += value

        # Check for alert conditions
        if metric_name in self.alert_thresholds:
            if self.metrics[metric_name] >= self.alert_thresholds[metric_name]:
                self._trigger_alert(metric_name, self.metrics[metric_name])

    def get_metrics(self) -> Dict[str, Any]:
        """Get current security metrics"""
        current_time = time.time()

        # Reset metrics if it's been more than an hour
        if current_time - self.last_reset > 3600:
            for key in self.metrics:
                self.metrics[key] = 0
            self.last_reset = current_time

        return {
            **self.metrics,
            "uptime_seconds": current_time - self.last_reset,
            "alert_thresholds": self.alert_thresholds
        }

    def _trigger_alert(self, metric_name: str, current_value: int) -> None:
        """Trigger security alert"""
        alert_message = f"Security Alert: {metric_name} threshold exceeded ({current_value})"
        logger.warning(alert_message)

        # In production, this would integrate with alerting systems
        # For now, just log the alert
        self._log_alert(metric_name, current_value)

    def _log_alert(self, metric_name: str, value: int) -> None:
        """Log security alert"""
        alert_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "alert_type": "threshold_exceeded",
            "metric": metric_name,
            "value": value,
            "threshold": self.alert_thresholds.get(metric_name, 0)
        }

        alert_log_file = Path("logs/security_alerts.log")
        alert_log_file.parent.mkdir(exist_ok=True)

        try:
            with open(alert_log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(alert_entry, default=str) + '\n')
        except Exception as e:
            logger.error(f"Failed to write alert log: {e}")

class APISecurityManager:
    """API security management with key rotation and validation"""

    def __init__(self, config: SecurityConfig):
        self.config = config
        self.key_store_file = Path("config/api_keys.enc")
        self.key_store_file.parent.mkdir(exist_ok=True)

    def validate_api_key(self, service: str, api_key: str) -> bool:
        """Validate API key format and security"""
        if not api_key or len(api_key.strip()) == 0:
            return False

        # Service-specific validation
        if service.lower() == "openrouter":
            # OpenRouter keys start with "sk-or-v1-"
            if not api_key.startswith("sk-or-v1-"):
                return False
            if len(api_key) < 50:  # Minimum expected length
                return False

        elif service.lower() == "pinecone":
            # Pinecone keys are typically 64+ characters
            if len(api_key) < 64:
                return False

        # Check for obviously fake/test keys
        if api_key.lower().startswith("your-") or api_key.lower().startswith("test"):
            return False

        return True

    def rotate_api_key(self, service: str, new_key: str) -> bool:
        """Rotate API key with validation"""
        if not self.validate_api_key(service, new_key):
            return False

        # In production, this would encrypt and store the key securely
        # For now, just validate and log
        logger.info(f"API key rotated for service: {service}")
        return True

    def get_secure_headers(self, service: str, api_key: str) -> Dict[str, str]:
        """Get secure headers for API requests"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": f"Personal-RAG/2.0.0-{service}",
            "X-Request-ID": hashlib.md5(f"{time.time()}-{api_key[:10]}".encode()).hexdigest(),
            "X-Timestamp": str(int(time.time()))
        }

        # Service-specific headers
        if service.lower() == "openrouter":
            headers.update({
                "HTTP-Referer": os.getenv("OPENROUTER_REFERER", ""),
                "X-Title": os.getenv("OPENROUTER_TITLE", "Personal RAG")
            })

        return headers

# Global security instances
_security_config = SecurityConfig()
_rate_limiter = RateLimiter(
    requests_per_minute=_security_config.rate_limit_requests_per_minute,
    burst_limit=_security_config.rate_limit_burst_limit
)
_input_validator = InputValidator(_security_config)
_security_auditor = SecurityAuditor(_security_config)
_security_monitor = SecurityMonitor(_security_config)
_api_security = APISecurityManager(_security_config)

# Convenience functions for easy integration
def validate_file_upload(file_path: str, file_content: bytes) -> Tuple[bool, str]:
    """Validate file upload"""
    _security_monitor.record_metric("total_requests")
    result = _input_validator.validate_file_upload(file_path, file_content)
    if not result[0]:
        _security_monitor.record_metric("file_upload_rejections")
        _security_auditor.log_security_event("FILE_UPLOAD_REJECTED", {
            "file_path": file_path,
            "reason": result[1]
        }, "WARNING")
    return result

def validate_query(query: str) -> Tuple[bool, str]:
    """Validate user query"""
    result = _input_validator.validate_query(query)
    if not result[0]:
        _security_monitor.record_metric("failed_validations")
        _security_auditor.log_security_event("QUERY_VALIDATION_FAILED", {
            "query_length": len(query),
            "reason": result[1]
        }, "WARNING")
    return result

def check_rate_limit(identifier: str) -> bool:
    """Check if request should be rate limited"""
    if _rate_limiter.is_rate_limited(identifier):
        _security_monitor.record_metric("rate_limit_hits")
        _security_auditor.log_security_event("RATE_LIMIT_EXCEEDED", {
            "identifier": identifier
        }, "WARNING")
        return True
    _rate_limiter.record_request(identifier)
    return False

def sanitize_input(input_str: str) -> str:
    """Sanitize input string"""
    return _input_validator.sanitize_input(input_str)

def log_security_event(event_type: str, details: Dict[str, Any], severity: str = "INFO"):
    """Log security event"""
    _security_auditor.log_security_event(event_type, details, severity)

def get_security_metrics() -> Dict[str, Any]:
    """Get current security metrics"""
    return _security_monitor.get_metrics()

def get_secure_api_headers(service: str, api_key: str) -> Dict[str, str]:
    """Get secure headers for API requests"""
    return _api_security.get_secure_headers(service, api_key)