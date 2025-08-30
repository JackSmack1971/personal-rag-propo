# Security Configuration Templates

## Document Information
- **Document ID:** SEC-CONFIG-TEMPLATES-001
- **Version:** 1.0.0
- **Created:** 2025-08-30
- **Last Updated:** 2025-08-30
- **Status:** Draft

## Executive Summary

This document provides comprehensive security configuration templates for the Personal RAG Chatbot system, covering development, staging, and production environments. The templates include secure defaults, hardening configurations, and environment-specific customizations to ensure consistent security posture across all deployments.

## 1. Environment Configuration Templates

### 1.1 Development Environment Template

#### Security Configuration (config.dev.yaml)
```yaml
# Development Environment Security Configuration
# WARNING: This configuration is for development only - NOT for production use

application:
  name: "Personal RAG (Development)"
  version: "2.0.0-dev"
  debug: true  # Development only
  environment: "development"

# OpenRouter Configuration (Development)
openrouter:
  api_key: "${OPENROUTER_API_KEY}"  # Must be set in environment
  model: "openrouter/auto"
  referer: "http://localhost:7860"
  title: "Personal RAG Development"
  max_tokens: 1000
  temperature: 0.0
  # Development: More permissive rate limits
  requests_per_minute: 30
  requests_per_hour: 500

# Pinecone Configuration (Development)
pinecone:
  api_key: "${PINECONE_API_KEY}"  # Must be set in environment
  index: "personal-rag-dev"
  cloud: "aws"
  region: "us-east-1"
  namespace: "dev"
  grpc_enabled: false  # HTTP for development debugging

# Enhanced Embedding Configuration (Development)
embeddings:
  dense_model: "BAAI/bge-small-en-v1.5"
  backend: "torch"  # CPU for development
  cache_embeddings: true
  normalize_embeddings: true
  sparse_encoding_enabled: false
  # Development: Reduced precision for faster iteration
  model_precision: "fp32"

# Gradio UI Configuration (Development)
ui:
  analytics_enabled: false
  auth_enabled: false  # No auth for development
  theme: "soft"
  show_progress: "full"  # Detailed progress for debugging
  streaming_enabled: false
  mobile_optimized: true
  server_name: "0.0.0.0"  # Bind to all interfaces for development
  server_port: 7860

# MoE Configuration (Development)
moe:
  enabled: false  # Disabled by default in development
  router:
    enabled: false
    experts: ["general", "technical", "personal"]
    centroid_refresh_interval: 3600
  gate:
    enabled: false
    retrieve_sim_threshold: 0.62
    low_sim_threshold: 0.45
  reranker:
    enabled: false
    cross_encoder_model: "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Development-Specific Security Settings
security:
  # File upload restrictions (relaxed for development)
  max_file_size_mb: 50  # Larger for testing
  allowed_file_types: [".pdf", ".txt", ".md", ".docx"]
  trust_remote_code: false  # Always false

  # Rate limiting (relaxed for development)
  rate_limit_enabled: true
  rate_limit_requests_per_minute: 60
  rate_limit_burst_limit: 100

  # Logging (verbose for development)
  log_level: "DEBUG"
  security_log_level: "INFO"
  log_to_file: true
  log_to_console: true

  # Development security features
  enable_security_headers: false  # Disable for easier debugging
  enable_cors: true  # Enable for development tools
  cors_origins: ["http://localhost:3000", "http://localhost:7860"]

# Performance Configuration (Development)
performance:
  max_context_length: 8192
  batch_size: 8  # Smaller batches for development
  cache_size_mb: 256
  enable_monitoring: true
  log_performance_warnings: true

# Monitoring Configuration (Development)
monitoring:
  enabled: true
  metrics_interval_seconds: 30  # More frequent for development
  alert_on_errors: false  # Don't spam developers with alerts
  log_metrics: true
  export_metrics: false  # No external export in development
```

#### Environment Variables (.env.development)
```bash
# Development Environment Variables
# Copy this file to .env for development use

# Core API Keys (Required for development)
OPENROUTER_API_KEY=your_openrouter_key_here
PINECONE_API_KEY=your_pinecone_key_here

# Configuration Path
CONFIG_PATH=config.dev.yaml

# Development Environment Settings
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG

# Security Settings (Development)
TRUST_REMOTE_CODE=false
MAX_FILE_SIZE_MB=50
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS_PER_MINUTE=60

# Performance Settings (Development)
SENTENCE_TRANSFORMERS_BACKEND=torch
PINECONE_GRPC_ENABLED=false
GRADIO_ANALYTICS_ENABLED=false
GRADIO_AUTH_ENABLED=false

# MoE Settings (Development)
MOE_ENABLED=false
MOE_ROUTER_ENABLED=false
MOE_GATE_ENABLED=false
MOE_RERANKER_ENABLED=false

# Monitoring Settings (Development)
ENABLE_PERFORMANCE_MONITORING=true
METRICS_EXPORT_ENABLED=false
LOG_METRICS=true

# Development Tools
ENABLE_CORS=true
CORS_ORIGINS=http://localhost:3000,http://localhost:7860
ENABLE_SECURITY_HEADERS=false
```

### 1.2 Staging Environment Template

#### Security Configuration (config.staging.yaml)
```yaml
# Staging Environment Security Configuration
# This configuration bridges development and production

application:
  name: "Personal RAG (Staging)"
  version: "2.0.0-staging"
  debug: false  # No debug in staging
  environment: "staging"

# OpenRouter Configuration (Staging)
openrouter:
  api_key: "${OPENROUTER_API_KEY}"
  model: "openrouter/auto"
  referer: "https://staging.personal-rag.example.com"
  title: "Personal RAG Staging"
  max_tokens: 1000
  temperature: 0.0
  # Staging: Moderate rate limits
  requests_per_minute: 20
  requests_per_hour: 1000

# Pinecone Configuration (Staging)
pinecone:
  api_key: "${PINECONE_API_KEY}"
  index: "personal-rag-staging"
  cloud: "aws"
  region: "us-east-1"
  namespace: "staging"
  grpc_enabled: true

# Enhanced Embedding Configuration (Staging)
embeddings:
  dense_model: "BAAI/bge-small-en-v1.5"
  backend: "torch"  # Test both backends in staging
  cache_embeddings: true
  normalize_embeddings: true
  sparse_encoding_enabled: false
  model_precision: "fp16"  # Test reduced precision

# Gradio UI Configuration (Staging)
ui:
  analytics_enabled: false
  auth_enabled: true  # Enable basic auth for staging
  theme: "soft"
  show_progress: "minimal"
  streaming_enabled: false
  mobile_optimized: true
  server_name: "0.0.0.0"
  server_port: 7860

# Security Configuration (Staging)
security:
  # File upload restrictions (production-like)
  max_file_size_mb: 10
  allowed_file_types: [".pdf", ".txt", ".md"]
  trust_remote_code: false

  # Rate limiting (production-like)
  rate_limit_enabled: true
  rate_limit_requests_per_minute: 30
  rate_limit_burst_limit: 50

  # Logging (detailed for testing)
  log_level: "INFO"
  security_log_level: "INFO"
  log_to_file: true
  log_to_console: false

  # Security headers (enabled for testing)
  enable_security_headers: true
  enable_cors: false
  cors_origins: []

# MoE Configuration (Staging)
moe:
  enabled: true  # Test MoE in staging
  router:
    enabled: true
    experts: ["general", "technical", "personal", "code"]
    centroid_refresh_interval: 3600
  gate:
    enabled: true
    retrieve_sim_threshold: 0.62
    low_sim_threshold: 0.45
  reranker:
    enabled: true
    cross_encoder_model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
    uncertainty_threshold: 0.15

# Performance Configuration (Staging)
performance:
  max_context_length: 8192
  batch_size: 16
  cache_size_mb: 512
  enable_monitoring: true
  log_performance_warnings: true

# Monitoring Configuration (Staging)
monitoring:
  enabled: true
  metrics_interval_seconds: 60
  alert_on_errors: true
  log_metrics: true
  export_metrics: true  # Test metrics export
  alert_endpoints:
    - "slack://staging-alerts"
    - "email://staging-team@example.com"
```

#### Environment Variables (.env.staging)
```bash
# Staging Environment Variables

# Core API Keys
OPENROUTER_API_KEY=your_staging_openrouter_key_here
PINECONE_API_KEY=your_staging_pinecone_key_here

# Configuration Path
CONFIG_PATH=config.staging.yaml

# Environment Settings
ENVIRONMENT=staging
DEBUG=false
LOG_LEVEL=INFO

# Security Settings
TRUST_REMOTE_CODE=false
MAX_FILE_SIZE_MB=10
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS_PER_MINUTE=30

# Performance Settings
SENTENCE_TRANSFORMERS_BACKEND=torch
PINECONE_GRPC_ENABLED=true
GRADIO_ANALYTICS_ENABLED=false
GRADIO_AUTH_ENABLED=true

# MoE Settings (Test in staging)
MOE_ENABLED=true
MOE_ROUTER_ENABLED=true
MOE_GATE_ENABLED=true
MOE_RERANKER_ENABLED=true

# Monitoring Settings
ENABLE_PERFORMANCE_MONITORING=true
METRICS_EXPORT_ENABLED=true
LOG_METRICS=true
ALERT_ENDPOINTS=slack://staging-alerts,email://staging-team@example.com
```

### 1.3 Production Environment Template

#### Security Configuration (config.production.yaml)
```yaml
# Production Environment Security Configuration
# CRITICAL: This configuration is for production use only

application:
  name: "Personal RAG (Production)"
  version: "2.0.0"
  debug: false  # Never enable debug in production
  environment: "production"

# OpenRouter Configuration (Production)
openrouter:
  api_key: "${OPENROUTER_API_KEY}"
  model: "openrouter/auto"
  referer: "https://personal-rag.example.com"
  title: "Personal RAG Production"
  max_tokens: 1000
  temperature: 0.0
  # Production: Conservative rate limits
  requests_per_minute: 15
  requests_per_hour: 500

# Pinecone Configuration (Production)
pinecone:
  api_key: "${PINECONE_API_KEY}"
  index: "personal-rag-production"
  cloud: "aws"
  region: "us-east-1"
  namespace: "production"
  grpc_enabled: true

# Enhanced Embedding Configuration (Production)
embeddings:
  dense_model: "BAAI/bge-small-en-v1.5"
  backend: "openvino"  # Use OpenVINO for production performance
  cache_embeddings: true
  normalize_embeddings: true
  sparse_encoding_enabled: false
  model_precision: "fp16"  # Reduced precision for performance

# Gradio UI Configuration (Production)
ui:
  analytics_enabled: false
  auth_enabled: true  # Require authentication
  theme: "soft"
  show_progress: "minimal"
  streaming_enabled: false
  mobile_optimized: true
  server_name: "127.0.0.1"  # Bind to localhost, use reverse proxy
  server_port: 7860

# Security Configuration (Production)
security:
  # File upload restrictions (strict)
  max_file_size_mb: 5  # Conservative limit
  allowed_file_types: [".pdf", ".txt", ".md"]
  trust_remote_code: false

  # Rate limiting (strict)
  rate_limit_enabled: true
  rate_limit_requests_per_minute: 20
  rate_limit_burst_limit: 30

  # Logging (security-focused)
  log_level: "WARNING"
  security_log_level: "INFO"
  log_to_file: true
  log_to_console: false

  # Security headers (maximum protection)
  enable_security_headers: true
  enable_cors: false
  cors_origins: []

  # Additional production security
  enable_https_redirect: true
  enable_hsts: true
  enable_csp: true
  session_timeout_minutes: 30

# MoE Configuration (Production)
moe:
  enabled: true  # Enable in production after testing
  router:
    enabled: true
    experts: ["general", "technical", "personal", "code"]
    centroid_refresh_interval: 3600
  gate:
    enabled: true
    retrieve_sim_threshold: 0.62
    low_sim_threshold: 0.45
  reranker:
    enabled: true
    stage2_enabled: false  # Disable expensive LLM reranking
    cross_encoder_model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
    uncertainty_threshold: 0.15

# Performance Configuration (Production)
performance:
  max_context_length: 8192
  batch_size: 32
  cache_size_mb: 1024
  enable_monitoring: true
  log_performance_warnings: false  # Don't spam logs

# Monitoring Configuration (Production)
monitoring:
  enabled: true
  metrics_interval_seconds: 60
  alert_on_errors: true
  log_metrics: false  # Don't log metrics to avoid spam
  export_metrics: true
  alert_endpoints:
    - "slack://production-alerts"
    - "pagerduty://production-incidents"
    - "email://production-team@example.com"
  alert_escalation:
    critical:
      immediate_channels: ["pagerduty", "sms"]
      escalation_time: 300  # 5 minutes
    high:
      immediate_channels: ["slack", "email"]
      escalation_time: 1800  # 30 minutes
```

#### Environment Variables (.env.production)
```bash
# Production Environment Variables
# CRITICAL: Secure these values properly

# Core API Keys
OPENROUTER_API_KEY=your_production_openrouter_key_here
PINECONE_API_KEY=your_production_pinecone_key_here

# Configuration Path
CONFIG_PATH=config.production.yaml

# Environment Settings
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=WARNING

# Security Settings (Maximum security)
TRUST_REMOTE_CODE=false
MAX_FILE_SIZE_MB=5
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS_PER_MINUTE=20

# Performance Settings (Optimized)
SENTENCE_TRANSFORMERS_BACKEND=openvino
PINECONE_GRPC_ENABLED=true
GRADIO_ANALYTICS_ENABLED=false
GRADIO_AUTH_ENABLED=true

# MoE Settings (Production)
MOE_ENABLED=true
MOE_ROUTER_ENABLED=true
MOE_GATE_ENABLED=true
MOE_RERANKER_ENABLED=true
MOE_STAGE2_ENABLED=false

# Monitoring Settings (Production)
ENABLE_PERFORMANCE_MONITORING=true
METRICS_EXPORT_ENABLED=true
LOG_METRICS=false
ALERT_ENDPOINTS=slack://production-alerts,pagerduty://production-incidents,email://production-team@example.com

# Production Security
ENABLE_HTTPS_REDIRECT=true
ENABLE_HSTS=true
ENABLE_CSP=true
SESSION_TIMEOUT_MINUTES=30
```

## 2. Security Hardening Templates

### 2.1 System Hardening Template

#### Docker Security Configuration (Dockerfile.production)
```dockerfile
# Production Dockerfile with Security Hardening
FROM python:3.11-slim

# Security: Create non-root user at the beginning
RUN groupadd -r raguser && useradd -r -g raguser raguser

# Security: Install security updates and minimal packages
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        gnupg \
        lsb-release \
        && \
    rm -rf /var/lib/apt/lists/*

# Security: Add security repositories
RUN curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# Security: Create application directory with correct permissions
RUN mkdir -p /app && chown -R raguser:raguser /app

# Security: Set working directory
WORKDIR /app

# Security: Copy application with minimal permissions
COPY --chown=raguser:raguser . .

# Security: Install dependencies with security scanning
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir safety && \
    safety check --exit-code 1 && \
    pip uninstall -y safety

# Security: Remove unnecessary build dependencies
RUN apt-get purge -y curl gnupg lsb-release && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/cache/apt/*

# Security: Create non-writable directories
RUN mkdir -p /app/logs /app/cache /app/data && \
    chown -R raguser:raguser /app/logs /app/cache /app/data && \
    chmod 755 /app/logs /app/cache /app/data

# Security: Set read-only root filesystem where possible
VOLUME ["/tmp", "/var/tmp", "/app/logs", "/app/cache"]

# Security: Switch to non-root user
USER raguser

# Security: Set secure environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Security: Use exec form for CMD
CMD ["python", "-m", "app"]
```

#### System Security Configuration (security-limits.conf)
```bash
# Production Security Limits Configuration
# Place in /etc/security/limits.d/rag.conf

# User limits for raguser
raguser soft nofile 1024
raguser hard nofile 2048
raguser soft nproc 128
raguser hard nproc 256

# Memory limits
raguser soft as 4294967296
raguser hard as 6442450944

# CPU limits (5 minutes soft, 10 minutes hard)
raguser soft cpu 300
raguser hard cpu 600

# Disable core dumps for security
raguser soft core 0
raguser hard core 0

# Stack size limits
raguser soft stack 8192
raguser hard stack 16384
```

### 2.2 Network Security Template

#### Firewall Configuration (firewall.sh)
```bash
#!/bin/bash
# Production Firewall Configuration
# Execute with root privileges

# Flush existing rules
iptables -F
iptables -X
iptables -t nat -F
iptables -t nat -X
iptables -t mangle -F
iptables -t mangle -X

# Default policies
iptables -P INPUT DROP
iptables -P FORWARD DROP
iptables -P OUTPUT ACCEPT

# Allow loopback interface
iptables -A INPUT -i lo -j ACCEPT

# Allow established connections
iptables -A INPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT

# Allow SSH (restrict to specific IPs in production)
# iptables -A INPUT -p tcp -s YOUR_IP_RANGE --dport 22 -j ACCEPT

# Allow HTTP/HTTPS for Gradio (through reverse proxy)
iptables -A INPUT -p tcp --dport 80 -j ACCEPT
iptables -A INPUT -p tcp --dport 443 -j ACCEPT

# Allow Gradio port (only from localhost/reverse proxy)
iptables -A INPUT -p tcp -s 127.0.0.1 --dport 7860 -j ACCEPT

# Rate limiting for SSH
iptables -A INPUT -p tcp --dport 22 -m conntrack --ctstate NEW -m recent --set
iptables -A INPUT -p tcp --dport 22 -m conntrack --ctstate NEW -m recent --update --seconds 60 --hitcount 4 -j DROP

# Drop invalid packets
iptables -A INPUT -m conntrack --ctstate INVALID -j DROP

# Protection against common attacks
iptables -A INPUT -p tcp --tcp-flags ALL NONE -j DROP
iptables -A INPUT -p tcp --tcp-flags SYN,FIN SYN,FIN -j DROP
iptables -A INPUT -p tcp --tcp-flags SYN,RST SYN,RST -j DROP
iptables -A INPUT -p tcp --tcp-flags FIN,RST FIN,RST -j DROP
iptables -A INPUT -p tcp --tcp-flags ACK,FIN FIN -j DROP
iptables -A INPUT -p tcp --tcp-flags ACK,PSH PSH -j DROP
iptables -A INPUT -p tcp --tcp-flags ACK,URG URG -j DROP

# Save rules
iptables-save > /etc/iptables/rules.v4

echo "Firewall configuration completed"
```

#### Nginx Reverse Proxy Configuration (nginx.conf)
```nginx
# Production Nginx Reverse Proxy Configuration
# Place in /etc/nginx/sites-available/personal-rag

upstream personal_rag_app {
    server 127.0.0.1:7860;
    keepalive 32;
}

server {
    listen 80;
    server_name personal-rag.example.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name personal-rag.example.com;

    # SSL Configuration
    ssl_certificate /etc/ssl/certs/personal-rag.crt;
    ssl_certificate_key /etc/ssl/private/personal-rag.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;

    # Security Headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline';" always;

    # HSTS
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

    # Client Max Body Size
    client_max_body_size 5M;

    # Rate Limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=20r/m;
    limit_req_zone $binary_remote_addr zone=upload:10m rate=10r/m;

    location / {
        # Rate limiting
        limit_req zone=api burst=5 nodelay;

        # Proxy settings
        proxy_pass http://personal_rag_app;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeout settings
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;

        # Buffer settings
        proxy_buffering on;
        proxy_buffer_size 128k;
        proxy_buffers 4 256k;
        proxy_busy_buffers_size 256k;
    }

    location /upload {
        # Stricter rate limiting for uploads
        limit_req zone=upload burst=2 nodelay;

        # File upload size limit
        client_max_body_size 5M;

        proxy_pass http://personal_rag_app;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Health check endpoint
    location /health {
        access_log off;
        return 200 "healthy\n";
        add_header Content-Type text/plain;
    }

    # Deny access to hidden files
    location ~ /\. {
        deny all;
        access_log off;
        log_not_found off;
    }
}
```

## 3. Monitoring and Alerting Templates

### 3.1 Prometheus Configuration Template

#### Prometheus Configuration (prometheus.yml)
```yaml
# Production Prometheus Configuration
global:
  scrape_interval: 60s
  evaluation_interval: 60s
  external_labels:
    environment: production
    service: personal-rag

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'personal-rag'
    static_configs:
      - targets: ['localhost:8000']
    scrape_interval: 30s
    scrape_timeout: 10s
    metrics_path: '/metrics'
    params:
      format: ['prometheus']
    relabel_configs:
      - source_labels: [__address__]
        target_label: instance
        replacement: 'personal-rag-prod'

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['localhost:9100']
    scrape_interval: 30s

  - job_name: 'cadvisor'
    static_configs:
      - targets: ['localhost:8080']
    scrape_interval: 30s
    metrics_path: '/metrics'
```

#### Alert Rules (alert_rules.yml)
```yaml
# Production Alert Rules

groups:
  - name: personal_rag_alerts
    rules:
      # Application Health Alerts
      - alert: PersonalRAGDown
        expr: up{job="personal-rag"} == 0
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Personal RAG is down"
          description: "Personal RAG has been down for more than 5 minutes."

      # Performance Alerts
      - alert: HighResponseTime
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 5
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High response time detected"
          description: "95th percentile response time is above 5 seconds for 10 minutes."

      # Error Rate Alerts
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) > 0.05
        for: 5m
        labels:
          severity: error
        annotations:
          summary: "High error rate detected"
          description: "Error rate is above 5% for 5 minutes."

      # Resource Usage Alerts
      - alert: HighCPUUsage
        expr: 100 - (avg by(instance) (irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100) > 90
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High CPU usage detected"
          description: "CPU usage is above 90% for 10 minutes."

      - alert: HighMemoryUsage
        expr: (1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes) * 100 > 85
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage detected"
          description: "Memory usage is above 85% for 10 minutes."

      # Security Alerts
      - alert: SecurityEventDetected
        expr: increase(security_events_total[5m]) > 0
        labels:
          severity: critical
        annotations:
          summary: "Security event detected"
          description: "A security event has been detected in the application logs."

      # MoE Performance Alerts
      - alert: MoERoutingFailure
        expr: increase(moe_routing_failures_total[5m]) > 5
        for: 5m
        labels:
          severity: error
        annotations:
          summary: "MoE routing failures detected"
          description: "MoE routing has failed more than 5 times in 5 minutes."
```

### 3.2 Grafana Dashboard Template

#### Dashboard Configuration (dashboard.json)
```json
{
  "dashboard": {
    "title": "Personal RAG Production Dashboard",
    "tags": ["personal-rag", "production"],
    "timezone": "UTC",
    "panels": [
      {
        "title": "Application Health",
        "type": "stat",
        "targets": [
          {
            "expr": "up{job=\"personal-rag\"}",
            "legendFormat": "Application Status"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "mappings": [
              {
                "options": {
                  "0": {
                    "text": "DOWN",
                    "color": "red"
                  },
                  "1": {
                    "text": "UP",
                    "color": "green"
                  }
                },
                "type": "value"
              }
            ]
          }
        }
      },
      {
        "title": "Response Time (95th percentile)",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ],
        "yAxes": [
          {
            "unit": "seconds",
            "min": 0
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total{status=~\"5..\"}[5m]) / rate(http_requests_total[5m]) * 100",
            "legendFormat": "Error Rate %"
          }
        ],
        "yAxes": [
          {
            "unit": "percent",
            "min": 0,
            "max": 100
          }
        ]
      },
      {
        "title": "Resource Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "100 - (avg by(instance) (irate(node_cpu_seconds_total{mode=\"idle\"}[5m])) * 100)",
            "legendFormat": "CPU Usage %"
          },
          {
            "expr": "(1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes) * 100",
            "legendFormat": "Memory Usage %"
          }
        ],
        "yAxes": [
          {
            "unit": "percent",
            "min": 0,
            "max": 100
          }
        ]
      },
      {
        "title": "Security Events",
        "type": "table",
        "targets": [
          {
            "expr": "increase(security_events_total[1h])",
            "legendFormat": "Security Events"
          }
        ]
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "30s"
  }
}
```

## 4. Backup and Recovery Templates

### 4.1 Backup Configuration Template

#### Backup Script (backup.sh)
```bash
#!/bin/bash
# Production Backup Script
# Execute daily via cron

# Configuration
BACKUP_DIR="/opt/personal-rag/backups"
RETENTION_DAYS=30
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Create backup directory if it doesn't exist
mkdir -p "$BACKUP_DIR"

# Function to log backup operations
log_backup() {
    echo "$(date): $1" >> "$BACKUP_DIR/backup.log"
}

# Start backup
log_backup "Starting backup process: $TIMESTAMP"

# Backup application configuration (excluding secrets)
log_backup "Backing up configuration files"
tar -czf "$BACKUP_DIR/config_$TIMESTAMP.tar.gz" \
    -C /opt/personal-rag \
    --exclude='*.key' \
    --exclude='*.pem' \
    --exclude='secrets' \
    config/

# Backup vector database data (if using local instance)
if [ -d "/opt/personal-rag/vector_db" ]; then
    log_backup "Backing up vector database"
    tar -czf "$BACKUP_DIR/vector_db_$TIMESTAMP.tar.gz" \
        -C /opt/personal-rag \
        vector_db/
fi

# Backup application logs
log_backup "Backing up application logs"
tar -czf "$BACKUP_DIR/logs_$TIMESTAMP.tar.gz" \
    -C /opt/personal-rag \
    logs/

# Backup MoE centroids and models (if applicable)
if [ -d "/opt/personal-rag/models" ]; then
    log_backup "Backing up ML models and centroids"
    tar -czf "$BACKUP_DIR/models_$TIMESTAMP.tar.gz" \
        -C /opt/personal-rag \
        models/
fi

# Encrypt sensitive backups
if [ -f "/opt/personal-rag/.backup_key" ]; then
    log_backup "Encrypting sensitive backups"
    openssl enc -aes-256-cbc \
        -salt \
        -in "$BACKUP_DIR/config_$TIMESTAMP.tar.gz" \
        -out "$BACKUP_DIR/config_$TIMESTAMP.tar.gz.enc" \
        -kfile /opt/personal-rag/.backup_key

    # Remove unencrypted version
    rm "$BACKUP_DIR/config_$TIMESTAMP.tar.gz"
fi

# Calculate backup sizes
CONFIG_SIZE=$(du -sh "$BACKUP_DIR/config_$TIMESTAMP.tar.gz"* | cut -f1)
LOGS_SIZE=$(du -sh "$BACKUP_DIR/logs_$TIMESTAMP.tar.gz" | cut -f1)

log_backup "Backup sizes - Config: $CONFIG_SIZE, Logs: $LOGS_SIZE"

# Clean up old backups
log_backup "Cleaning up backups older than $RETENTION_DAYS days"
find "$BACKUP_DIR" -name "*.tar.gz*" -type f -mtime +$RETENTION_DAYS -delete

# Verify backup integrity
log_backup "Verifying backup integrity"
if [ -f "$BACKUP_DIR/config_$TIMESTAMP.tar.gz" ]; then
    if tar -tzf "$BACKUP_DIR/config_$TIMESTAMP.tar.gz" > /dev/null 2>&1; then
        log_backup "Config backup integrity check: PASSED"
    else
        log_backup "Config backup integrity check: FAILED"
    fi
fi

# Send backup notification
if [ -f "/opt/personal-rag/.slack_webhook" ]; then
    curl -X POST -H 'Content-type: application/json' \
        --data "{\"text\":\"Personal RAG backup completed: $TIMESTAMP\"}" \
        $(cat /opt/personal-rag/.slack_webhook)
fi

log_backup "Backup process completed: $TIMESTAMP"

# Exit with success
exit 0
```

#### Backup Verification Script (verify_backup.sh)
```bash
#!/bin/bash
# Backup Verification Script
# Execute after backup completion

BACKUP_DIR="/opt/personal-rag/backups"
LATEST_BACKUP=$(ls -t "$BACKUP_DIR"/*.tar.gz* | head -1)

echo "Verifying backup: $LATEST_BACKUP"

# Check if backup file exists
if [ ! -f "$LATEST_BACKUP" ]; then
    echo "ERROR: Backup file not found"
    exit 1
fi

# Check backup file size
BACKUP_SIZE=$(stat -f%z "$LATEST_BACKUP" 2>/dev/null || stat -c%s "$LATEST_BACKUP")
if [ "$BACKUP_SIZE" -lt 1000 ]; then
    echo "ERROR: Backup file suspiciously small ($BACKUP_SIZE bytes)"
    exit 1
fi

# Verify backup integrity
if [[ "$LATEST_BACKUP" == *.enc ]]; then
    echo "Backup is encrypted, skipping integrity check"
else
    if ! tar -tzf "$LATEST_BACKUP" > /dev/null 2>&1; then
        echo "ERROR: Backup integrity check failed"
        exit 1
    fi
fi

# Check backup age
BACKUP_AGE=$(($(date +%s) - $(stat -f%B "$LATEST_BACKUP" 2>/dev/null || stat -c%Y "$LATEST_BACKUP")))
if [ "$BACKUP_AGE" -gt 86400 ]; then
    echo "WARNING: Backup is older than 24 hours"
fi

echo "Backup verification completed successfully"
exit 0
```

## 5. Compliance Configuration Templates

### 5.1 GDPR Compliance Template

#### Data Processing Configuration (gdpr-config.yaml)
```yaml
# GDPR Compliance Configuration

data_protection:
  # Data minimization settings
  retention_periods:
    user_queries: 2555  # 7 years for audit
    system_logs: 2555   # 7 years for audit
    performance_metrics: 365  # 1 year
    security_events: 2555  # 7 years

  # Data subject rights
  data_subject_rights:
    access: true
    rectification: true
    erasure: true
    restrict_processing: true
    data_portability: true
    object: true

  # Privacy by design
  privacy_settings:
    data_minimization: true
    purpose_limitation: true
    storage_limitation: true
    accuracy: true
    integrity: true
    confidentiality: true
    accountability: true

  # Consent management
  consent_management:
    required_consents:
      - data_processing
      - analytics
      - marketing
    consent_retention: 2555  # 7 years

  # Data breach notification
  breach_notification:
    detection_enabled: true
    notification_deadline: 72  # hours
    supervisory_authority_contacts:
      - name: "Data Protection Authority"
        email: "dpa@example.com"
        phone: "+1-555-0123"
    affected_users_notification: true

  # International data transfers
  data_transfers:
    adequacy_decisions:
      - country: "United States"
        adequacy_status: "partial"  # Requires SCCs
        safeguards: "Standard Contractual Clauses"
    transfer_mechanisms:
      - "Standard Contractual Clauses"
      - "Binding Corporate Rules"
      - "Adequacy Decision"
```

### 5.2 Audit Logging Template

#### Audit Configuration (audit-config.yaml)
```yaml
# Production Audit Configuration

audit:
  # Audit scope
  audit_scope:
    log_authentication: true
    log_authorization: true
    log_data_access: true
    log_configuration_changes: true
    log_security_events: true
    log_admin_actions: true

  # Audit log settings
  log_settings:
    format: "json"
    encryption: true
    integrity_protection: true
    compression: true
    retention_period: 2555  # 7 years
    storage_location: "/var/log/personal-rag/audit"

  # Audit events
  audit_events:
    user_login:
      enabled: true
      severity: "info"
      fields: ["timestamp", "user_id", "ip_address", "user_agent", "success"]
    data_access:
      enabled: true
      severity: "info"
      fields: ["timestamp", "user_id", "resource", "action", "result"]
    configuration_change:
      enabled: true
      severity: "warning"
      fields: ["timestamp", "user_id", "component", "old_value", "new_value"]
    security_event:
      enabled: true
      severity: "error"
      fields: ["timestamp", "event_type", "severity", "details", "source"]

  # Audit monitoring
  monitoring:
    real_time_alerts: true
    anomaly_detection: true
    compliance_reporting: true
    audit_log_integrity: true

  # Compliance reporting
  compliance:
    gdpr_compliance: true
    sox_compliance: true
    hipaa_compliance: false  # Enable if handling healthcare data
    pci_compliance: false    # Enable if handling payment data
```

## 6. Deployment Validation Checklist

### 6.1 Pre-Deployment Checklist

#### Security Validation
- [ ] All secrets are properly configured and not in source code
- [ ] File permissions are correctly set (no world-writable files)
- [ ] SSL/TLS certificates are valid and properly configured
- [ ] Firewall rules are in place and tested
- [ ] Security headers are enabled and configured
- [ ] Rate limiting is configured and tested
- [ ] Input validation is implemented for all user inputs
- [ ] Authentication and authorization are properly configured

#### Performance Validation
- [ ] Application starts within 2 seconds
- [ ] Memory usage is within configured limits
- [ ] CPU usage is reasonable under normal load
- [ ] Response times meet performance targets
- [ ] Caching is working correctly
- [ ] Database connections are properly configured

#### Operational Validation
- [ ] Logging is configured and working
- [ ] Monitoring is enabled and collecting metrics
- [ ] Backup procedures are in place and tested
- [ ] Recovery procedures are documented and tested
- [ ] Alerting is configured and tested

### 6.2 Post-Deployment Checklist

#### Security Verification
- [ ] Security scanning tools show no critical vulnerabilities
- [ ] Access controls are working as expected
- [ ] Security logs are being generated and monitored
- [ ] Incident response procedures are in place

#### Performance Verification
- [ ] Application performance meets targets under load
- [ ] Resource usage is within acceptable limits
- [ ] Monitoring dashboards are working correctly
- [ ] Alert thresholds are properly configured

#### Operational Verification
- [ ] Backup procedures are working correctly
- [ ] Log rotation is configured and working
- [ ] Monitoring alerts are being received
- [ ] Documentation is up to date

## 7. Conclusion

These security configuration templates provide comprehensive, production-ready configurations for the Personal RAG Chatbot system across development, staging, and production environments. The templates ensure:

**Security Principles**:
- **Defense in Depth**: Multiple layers of security controls
- **Secure by Default**: Conservative security settings by default
- **Environment-Specific**: Appropriate security levels for each environment
- **Compliance-Ready**: Built-in support for regulatory requirements

**Configuration Coverage**:
- **Application Security**: Input validation, authentication, authorization
- **System Security**: File permissions, process isolation, resource limits
- **Network Security**: Firewall rules, SSL/TLS, secure headers
- **Monitoring Security**: Secure logging, alerting, audit trails
- **Operational Security**: Backup procedures, access controls, incident response

**Template Benefits**:
- **Consistency**: Standardized configurations across environments
- **Maintainability**: Clear structure and documentation
- **Scalability**: Easily adaptable to different deployment scenarios
- **Security**: Comprehensive security controls and hardening measures
- **Compliance**: Built-in support for regulatory requirements

The templates serve as a foundation for secure, compliant, and maintainable deployments of the Personal RAG Chatbot system.