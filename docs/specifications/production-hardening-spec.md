# Production Hardening Specification

## Document Information
- **Document ID:** PROD-HARDEN-SPEC-001
- **Version:** 1.0.0
- **Created:** 2025-08-30
- **Last Updated:** 2025-08-30
- **Status:** Draft

## Executive Summary

This specification defines production hardening requirements for the Personal RAG Chatbot system, covering security hardening, operational resilience, configuration management, and production readiness. The hardening measures ensure the system can operate reliably and securely in production environments.

## 1. Security Hardening

### 1.1 Application Security

#### Input Validation Hardening
```python
class HardenedInputValidator:
    """Production-grade input validation"""

    def __init__(self):
        self._file_scanner = FileSecurityScanner()
        self._content_analyzer = ContentSecurityAnalyzer()
        self._rate_limiter = AdaptiveRateLimiter()

    def validate_file_upload(self, file_path: str, content: bytes) -> ValidationResult:
        """Comprehensive file validation for production"""

        # Basic validation
        if not self._validate_file_type(file_path):
            return ValidationResult.REJECTED_FILE_TYPE

        if not self._validate_file_size(content):
            return ValidationResult.REJECTED_FILE_SIZE

        # Security scanning
        if not self._file_scanner.scan_for_malware(content):
            return ValidationResult.SECURITY_THREAT_DETECTED

        # Content analysis
        if not self._content_analyzer.analyze_content(content):
            return ValidationResult.MALICIOUS_CONTENT

        # Rate limiting
        if not self._rate_limiter.check_rate_limit():
            return ValidationResult.RATE_LIMIT_EXCEEDED

        return ValidationResult.APPROVED

    def _validate_file_type(self, file_path: str) -> bool:
        """Strict file type validation"""
        allowed_extensions = {'.pdf', '.txt', '.md'}
        allowed_mime_types = {
            'application/pdf',
            'text/plain',
            'text/markdown'
        }

        # Check extension
        if not any(file_path.lower().endswith(ext) for ext in allowed_extensions):
            return False

        # Check MIME type
        mime_type = magic.from_buffer(content, mime=True)
        return mime_type in allowed_mime_types

    def _validate_file_size(self, content: bytes) -> bool:
        """File size validation with production limits"""
        max_size = int(os.getenv('MAX_FILE_SIZE_MB', '10')) * 1024 * 1024
        return len(content) <= max_size
```

#### API Security Hardening
```python
class HardenedAPIManager:
    """Production-hardened API communication"""

    def __init__(self):
        self._session_manager = SecureSessionManager()
        self._key_manager = APIKeyManager()
        self._circuit_breaker = CircuitBreaker()

    def execute_secure_request(self, endpoint: str, payload: dict) -> dict:
        """Execute API request with comprehensive security"""

        # Circuit breaker check
        if self._circuit_breaker.is_open():
            raise CircuitBreakerOpenException()

        # Key rotation check
        api_key = self._key_manager.get_current_key()

        # Secure session
        session = self._session_manager.get_secure_session()

        try:
            response = self._make_request(session, endpoint, payload, api_key)

            # Update circuit breaker
            self._circuit_breaker.record_success()

            return response

        except Exception as e:
            # Update circuit breaker
            self._circuit_breaker.record_failure()

            # Log security event
            self._log_security_event('API_REQUEST_FAILED', {
                'endpoint': endpoint,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            })

            raise

    def _make_request(self, session, endpoint: str, payload: dict, api_key: str) -> dict:
        """Make secure HTTP request"""
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
            'User-Agent': f'Personal-RAG/2.0.0-{self._get_instance_id()}',
            'X-Request-ID': str(uuid.uuid4()),
            'X-Timestamp': str(int(time.time()))
        }

        # Request signing for additional security
        signature = self._sign_request(payload, headers)
        headers['X-Signature'] = signature

        response = session.post(
            endpoint,
            json=payload,
            headers=headers,
            timeout=self._get_timeout(),
            verify=True,
            cert=self._get_client_cert()
        )

        response.raise_for_status()
        return response.json()
```

### 1.2 System Security

#### Process Isolation
```python
class ProcessHardener:
    """System process hardening"""

    def harden_process(self) -> None:
        """Apply production hardening to current process"""

        # Drop privileges if running as root
        self._drop_privileges()

        # Set resource limits
        self._set_resource_limits()

        # Configure security policies
        self._configure_seccomp()
        self._configure_capabilities()

        # Initialize security monitoring
        self._initialize_security_monitoring()

    def _drop_privileges(self) -> None:
        """Drop root privileges to unprivileged user"""
        if os.getuid() == 0:
            # Switch to application user
            app_uid = pwd.getpwnam('raguser').pw_uid
            app_gid = grp.getgrnam('raguser').gr_gid

            os.setgid(app_gid)
            os.setuid(app_uid)

    def _set_resource_limits(self) -> None:
        """Set resource limits for security"""
        limits = [
            (resource.RLIMIT_AS, (4 * 1024 * 1024 * 1024, 6 * 1024 * 1024 * 1024)),  # 4GB-6GB RAM
            (resource.RLIMIT_CPU, (300, 600)),  # 5-10 minutes CPU time
            (resource.RLIMIT_NOFILE, (1024, 2048)),  # File descriptors
            (resource.RLIMIT_NPROC, (128, 256))  # Processes/threads
        ]

        for limit_type, (soft, hard) in limits:
            resource.setrlimit(limit_type, (soft, hard))
```

#### File System Security
```python
class FileSystemHardener:
    """File system security hardening"""

    def harden_file_system(self) -> None:
        """Apply file system hardening"""

        # Set secure permissions
        self._set_secure_permissions()

        # Create secure temporary directories
        self._create_secure_temp_dirs()

        # Configure file access controls
        self._configure_file_access()

        # Initialize file integrity monitoring
        self._initialize_file_monitoring()

    def _set_secure_permissions(self) -> None:
        """Set secure file permissions"""
        secure_paths = [
            ('./config', 0o750),
            ('./logs', 0o750),
            ('./cache', 0o750),
            ('./data', 0o750)
        ]

        for path, permissions in secure_paths:
            if os.path.exists(path):
                os.chmod(path, permissions)
                # Set ownership to application user
                shutil.chown(path, user='raguser', group='raguser')

    def _create_secure_temp_dirs(self) -> None:
        """Create secure temporary directories"""
        import tempfile

        # Create application-specific temp directory
        app_temp = tempfile.mkdtemp(prefix='rag_', suffix='_tmp')
        os.chmod(app_temp, 0o700)

        # Set as default temp directory for application
        tempfile.tempdir = app_temp
```

## 2. Operational Hardening

### 2.1 Error Handling and Resilience

#### Graceful Degradation
```python
class ResilienceManager:
    """Production resilience management"""

    def __init__(self):
        self._health_checker = HealthChecker()
        self._fallback_manager = FallbackManager()
        self._recovery_manager = RecoveryManager()

    def execute_with_resilience(self, operation: callable) -> Any:
        """Execute operation with comprehensive error handling"""

        try:
            # Pre-execution health check
            if not self._health_checker.is_system_healthy():
                return self._fallback_manager.execute_fallback(operation)

            # Execute operation
            result = operation()

            # Post-execution validation
            if not self._validate_result(result):
                raise InvalidResultException()

            return result

        except Exception as e:
            # Log error with context
            self._log_error_with_context(e, operation)

            # Attempt recovery
            if self._recovery_manager.can_recover(e):
                return self._recovery_manager.recover(operation, e)

            # Fallback execution
            return self._fallback_manager.execute_fallback(operation)

    def _validate_result(self, result) -> bool:
        """Validate operation result"""
        if result is None:
            return False

        # Type validation
        if not isinstance(result, (dict, list, str)):
            return False

        # Content validation
        if isinstance(result, dict):
            return self._validate_dict_result(result)

        return True
```

#### Circuit Breaker Pattern
```python
class ProductionCircuitBreaker:
    """Production-grade circuit breaker"""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = CircuitState.CLOSED

    def execute(self, operation: callable) -> Any:
        """Execute operation through circuit breaker"""

        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
            else:
                raise CircuitBreakerOpenException()

        try:
            result = operation()

            # Success - reset failure count
            self._record_success()
            return result

        except Exception as e:
            # Failure - increment counter
            self._record_failure()

            # Check if should open circuit
            if self.failure_count >= self.failure_threshold:
                self._open_circuit()

            raise e

    def _record_success(self) -> None:
        """Record successful operation"""
        self.failure_count = 0
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED

    def _record_failure(self) -> None:
        """Record failed operation"""
        self.failure_count += 1
        self.last_failure_time = time.time()

    def _open_circuit(self) -> None:
        """Open the circuit breaker"""
        self.state = CircuitState.OPEN
        logger.warning("Circuit breaker opened due to repeated failures")

    def _should_attempt_reset(self) -> bool:
        """Check if should attempt to reset circuit"""
        return (time.time() - self.last_failure_time) > self.recovery_timeout
```

### 2.2 Resource Management

#### Memory Hardening
```python
class MemoryHardener:
    """Production memory management"""

    def __init__(self):
        self._memory_monitor = MemoryMonitor()
        self._gc_manager = GarbageCollectionManager()
        self._cache_manager = CacheManager()

    def harden_memory_usage(self) -> None:
        """Apply memory hardening measures"""

        # Set memory limits
        self._set_memory_limits()

        # Configure garbage collection
        self._configure_gc()

        # Initialize memory monitoring
        self._initialize_memory_monitoring()

        # Set up memory pressure handling
        self._configure_memory_pressure_handling()

    def _set_memory_limits(self) -> None:
        """Set production memory limits"""
        import resource

        # Set address space limit (4GB)
        resource.setrlimit(resource.RLIMIT_AS, (4 * 1024 * 1024 * 1024, -1))

        # Configure Python memory limits
        import gc
        gc.set_threshold(700, 10, 10)  # More aggressive GC

    def _configure_memory_pressure_handling(self) -> None:
        """Handle memory pressure situations"""
        def memory_pressure_handler():
            # Clear caches
            self._cache_manager.clear_all_caches()

            # Force garbage collection
            gc.collect()

            # Unload unused models
            self._unload_unused_models()

            logger.warning("Memory pressure handling activated")

        # Register memory pressure signal handler
        signal.signal(signal.SIGUSR1, lambda sig, frame: memory_pressure_handler())
```

### 2.3 Configuration Management

#### Secure Configuration
```python
class SecureConfigurationManager:
    """Production configuration management"""

    def __init__(self):
        self._config_validator = ConfigurationValidator()
        self._secret_manager = SecretManager()
        self._config_encryptor = ConfigurationEncryptor()

    def load_secure_configuration(self, config_path: str) -> dict:
        """Load configuration with security measures"""

        # Validate configuration file integrity
        if not self._validate_config_integrity(config_path):
            raise ConfigurationIntegrityException()

        # Load configuration
        config = self._load_config_file(config_path)

        # Validate configuration schema
        if not self._config_validator.validate_schema(config):
            raise ConfigurationValidationException()

        # Decrypt sensitive values
        config = self._decrypt_sensitive_values(config)

        # Apply security hardening
        config = self._apply_security_hardening(config)

        return config

    def _validate_config_integrity(self, config_path: str) -> bool:
        """Validate configuration file integrity"""
        if not os.path.exists(config_path):
            return False

        # Check file permissions
        stat = os.stat(config_path)
        if stat.st_mode & 0o077:  # Check if world/group readable
            logger.warning("Configuration file has overly permissive permissions")
            return False

        # Verify file ownership
        if stat.st_uid != os.getuid():
            logger.warning("Configuration file not owned by current user")
            return False

        return True

    def _decrypt_sensitive_values(self, config: dict) -> dict:
        """Decrypt sensitive configuration values"""
        encrypted_keys = ['openrouter.api_key', 'pinecone.api_key']

        for key_path in encrypted_keys:
            if self._has_key(config, key_path):
                encrypted_value = self._get_key(config, key_path)
                decrypted_value = self._secret_manager.decrypt(encrypted_value)
                self._set_key(config, key_path, decrypted_value)

        return config
```

## 3. Monitoring and Observability Hardening

### 3.1 Production Monitoring
```python
class ProductionMonitoringSystem:
    """Production-grade monitoring system"""

    def __init__(self):
        self._metrics_collector = MetricsCollector()
        self._log_aggregator = LogAggregator()
        self._alert_manager = AlertManager()
        self._health_checker = HealthChecker()

    def initialize_monitoring(self) -> None:
        """Initialize comprehensive monitoring"""

        # System metrics
        self._initialize_system_metrics()

        # Application metrics
        self._initialize_application_metrics()

        # Security metrics
        self._initialize_security_metrics()

        # Performance metrics
        self._initialize_performance_metrics()

        # Configure alerting
        self._configure_alerting()

    def _initialize_system_metrics(self) -> None:
        """Initialize system-level metrics"""
        metrics = [
            'cpu_usage_percent',
            'memory_usage_bytes',
            'disk_usage_percent',
            'network_connections',
            'open_file_descriptors'
        ]

        for metric in metrics:
            self._metrics_collector.register_gauge(metric, self._collect_system_metric)

    def _initialize_security_metrics(self) -> None:
        """Initialize security monitoring metrics"""
        security_metrics = [
            'failed_authentication_attempts',
            'file_upload_rejections',
            'api_rate_limit_hits',
            'model_loading_failures',
            'suspicious_request_patterns'
        ]

        for metric in security_metrics:
            self._metrics_collector.register_counter(metric)

    def collect_security_metrics(self) -> dict:
        """Collect comprehensive security metrics"""
        return {
            'authentication_failures': self._metrics_collector.get_counter('failed_authentication_attempts'),
            'file_rejections': self._metrics_collector.get_counter('file_upload_rejections'),
            'rate_limit_hits': self._metrics_collector.get_counter('api_rate_limit_hits'),
            'active_threats': self._detect_active_threats(),
            'security_incidents': self._get_recent_security_incidents()
        }
```

### 3.2 Log Security Hardening
```python
class SecureLoggingSystem:
    """Security-hardened logging system"""

    def __init__(self):
        self._log_encryptor = LogEncryptor()
        self._integrity_verifier = LogIntegrityVerifier()
        self._anomaly_detector = LogAnomalyDetector()

    def secure_log(self, level: str, message: str, context: dict = None) -> None:
        """Secure logging with integrity protection"""

        # Create log entry
        log_entry = self._create_secure_log_entry(level, message, context)

        # Detect anomalies
        if self._anomaly_detector.detect_anomaly(log_entry):
            self._handle_log_anomaly(log_entry)

        # Encrypt sensitive data
        log_entry = self._log_encryptor.encrypt_sensitive_data(log_entry)

        # Add integrity hash
        log_entry = self._integrity_verifier.add_integrity_hash(log_entry)

        # Write to secure log
        self._write_secure_log(log_entry)

    def _create_secure_log_entry(self, level: str, message: str, context: dict) -> dict:
        """Create secure log entry"""
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'level': level,
            'message': message,
            'context': context or {},
            'process_id': os.getpid(),
            'thread_id': threading.get_ident(),
            'user_id': self._get_current_user(),
            'session_id': self._get_session_id(),
            'ip_address': self._get_client_ip()
        }

    def _write_secure_log(self, log_entry: dict) -> None:
        """Write log entry with security measures"""
        # Ensure log directory security
        self._ensure_log_directory_security()

        # Rotate logs if necessary
        self._rotate_logs_if_needed()

        # Write with atomic operation
        self._atomic_log_write(log_entry)
```

## 4. Deployment Hardening

### 4.1 Container Security
```dockerfile
# Production Dockerfile with security hardening
FROM python:3.11-slim

# Security: Create non-root user
RUN groupadd -r raguser && useradd -r -g raguser raguser

# Security: Install security updates
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        && \
    rm -rf /var/lib/apt/lists/*

# Security: Set secure working directory
WORKDIR /app

# Security: Copy application with minimal permissions
COPY --chown=raguser:raguser . .

# Security: Install dependencies with security scanning
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir safety && \
    safety check

# Security: Remove unnecessary packages
RUN apt-get purge -y curl && \
    apt-get autoremove -y && \
    apt-get clean

# Security: Set read-only root filesystem
VOLUME ["/tmp", "/var/tmp"]

# Security: Drop all capabilities
USER raguser

# Security: Use exec form
CMD ["python", "app.py"]
```

### 4.2 Environment Hardening
```bash
#!/bin/bash
# Production environment hardening script

# Disable core dumps for security
echo '* hard core 0' >> /etc/security/limits.conf
echo '* soft core 0' >> /etc/security/limits.conf

# Configure sysctl security settings
cat >> /etc/sysctl.conf << EOF
# Security hardening
net.ipv4.tcp_syncookies = 1
net.ipv4.conf.all.rp_filter = 1
net.ipv4.conf.default.rp_filter = 1
net.ipv4.conf.all.accept_redirects = 0
net.ipv4.conf.default.accept_redirects = 0
kernel.randomize_va_space = 2
EOF

# Apply sysctl settings
sysctl -p

# Configure firewall (if applicable)
# ufw enable
# ufw default deny incoming
# ufw default allow outgoing
# ufw allow 7860/tcp  # Gradio port

# Set secure permissions on application directory
chown -R raguser:raguser /opt/personal-rag
chmod -R 750 /opt/personal-rag
```

## 5. Incident Response Hardening

### 5.1 Automated Response System
```python
class AutomatedIncidentResponder:
    """Automated incident response for production"""

    def __init__(self):
        self._threat_detector = ThreatDetector()
        self._response_orchestrator = ResponseOrchestrator()
        self._forensic_collector = ForensicCollector()

    def handle_incident(self, incident_type: str, context: dict) -> None:
        """Handle security incident with automated response"""

        # Assess incident severity
        severity = self._assess_severity(incident_type, context)

        # Collect forensic data
        forensic_data = self._forensic_collector.collect_evidence(context)

        # Execute automated response
        response_plan = self._create_response_plan(incident_type, severity)
        self._response_orchestrator.execute_plan(response_plan)

        # Escalate if necessary
        if self._requires_human_escalation(severity):
            self._escalate_to_human(response_plan, forensic_data)

    def _assess_severity(self, incident_type: str, context: dict) -> str:
        """Assess incident severity"""
        severity_rules = {
            'api_key_exposure': lambda ctx: 'critical' if ctx.get('key_type') == 'production' else 'high',
            'malicious_file': lambda ctx: 'high' if ctx.get('infection_confirmed') else 'medium',
            'rate_limit_attack': lambda ctx: 'medium',
            'model_poisoning': lambda ctx: 'critical'
        }

        rule = severity_rules.get(incident_type, lambda ctx: 'low')
        return rule(context)

    def _create_response_plan(self, incident_type: str, severity: str) -> dict:
        """Create automated response plan"""
        base_plans = {
            'critical': {
                'isolate_system': True,
                'rotate_credentials': True,
                'notify_security_team': True,
                'enable_enhanced_monitoring': True
            },
            'high': {
                'rotate_credentials': True,
                'increase_monitoring': True,
                'block_suspicious_ips': True
            },
            'medium': {
                'increase_monitoring': True,
                'log_enhanced_details': True
            }
        }

        return base_plans.get(severity, {})
```

## 6. Compliance and Audit Hardening

### 6.1 Audit Logging
```python
class ProductionAuditor:
    """Production audit logging system"""

    def __init__(self):
        self._audit_trail = AuditTrail()
        self._compliance_checker = ComplianceChecker()
        self._retention_manager = RetentionManager()

    def log_audit_event(self, event_type: str, details: dict) -> None:
        """Log auditable event with compliance requirements"""

        # Create audit entry
        audit_entry = {
            'event_id': str(uuid.uuid4()),
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': event_type,
            'actor': self._get_actor_identity(),
            'resource': details.get('resource'),
            'action': details.get('action'),
            'outcome': details.get('outcome'),
            'details': details,
            'compliance_flags': self._get_compliance_flags(event_type)
        }

        # Add integrity protection
        audit_entry['integrity_hash'] = self._calculate_integrity_hash(audit_entry)

        # Store in tamper-evident log
        self._audit_trail.append_entry(audit_entry)

        # Check retention requirements
        if self._requires_long_retention(event_type):
            self._retention_manager.mark_for_long_retention(audit_entry['event_id'])

    def _get_compliance_flags(self, event_type: str) -> list:
        """Get compliance requirements for event type"""
        compliance_matrix = {
            'authentication': ['GDPR', 'SOX'],
            'data_access': ['GDPR', 'HIPAA'],
            'configuration_change': ['SOX', 'PCI-DSS'],
            'security_incident': ['NIST', 'ISO27001']
        }

        return compliance_matrix.get(event_type, [])

    def generate_compliance_report(self, period: str) -> dict:
        """Generate compliance report for specified period"""
        entries = self._audit_trail.get_entries_for_period(period)

        return {
            'period': period,
            'total_events': len(entries),
            'compliance_summary': self._analyze_compliance(entries),
            'audit_findings': self._identify_audit_findings(entries),
            'recommendations': self._generate_recommendations(entries)
        }
```

## 7. Testing and Validation

### 7.1 Hardening Validation
```python
class HardeningValidator:
    """Validate production hardening measures"""

    def validate_hardening(self) -> dict:
        """Comprehensive hardening validation"""

        validation_results = {}

        # Security validations
        validation_results['security'] = self._validate_security_hardening()

        # Operational validations
        validation_results['operational'] = self._validate_operational_hardening()

        # Performance validations
        validation_results['performance'] = self._validate_performance_impact()

        # Compliance validations
        validation_results['compliance'] = self._validate_compliance_readiness()

        return validation_results

    def _validate_security_hardening(self) -> dict:
        """Validate security hardening measures"""
        return {
            'file_permissions': self._check_file_permissions(),
            'process_isolation': self._check_process_isolation(),
            'memory_limits': self._check_memory_limits(),
            'network_security': self._check_network_security(),
            'input_validation': self._test_input_validation()
        }

    def _check_file_permissions(self) -> bool:
        """Check file permission hardening"""
        critical_paths = [
            ('./config', 0o750),
            ('./logs', 0o750),
            ('./cache', 0o750)
        ]

        for path, expected_perms in critical_paths:
            if os.path.exists(path):
                actual_perms = os.stat(path).st_mode & 0o777
                if actual_perms != expected_perms:
                    return False
        return True
```

## 8. Conclusion

This production hardening specification provides comprehensive measures to ensure the Personal RAG Chatbot system is production-ready. The hardening covers security, operational resilience, monitoring, and compliance aspects.

**Key Hardening Principles**:
- **Defense in Depth**: Multiple layers of security controls
- **Fail-Safe Defaults**: Secure configuration by default
- **Continuous Monitoring**: Real-time security and performance monitoring
- **Automated Response**: Rapid incident detection and response
- **Compliance Ready**: Built-in audit and compliance capabilities

**Hardening Coverage**:
- **Security**: Input validation, API security, process isolation, file system security
- **Operational**: Error handling, resource management, configuration security
- **Monitoring**: Comprehensive metrics, secure logging, alerting
- **Compliance**: Audit logging, retention management, compliance reporting

The hardening measures ensure the system can operate securely and reliably in production environments while maintaining the performance and functionality requirements.