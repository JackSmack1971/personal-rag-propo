# Production Deployment Runbook

## Personal RAG Chatbot - Production Operations Guide

### Version: 2.0.0
### Last Updated: August 30, 2025
### Document Owner: DevOps Team

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Environment Setup](#environment-setup)
4. [Deployment Procedures](#deployment-procedures)
5. [Monitoring and Alerting](#monitoring-and-alerting)
6. [Security Procedures](#security-procedures)
7. [Backup and Recovery](#backup-and-recovery)
8. [Troubleshooting](#troubleshooting)
9. [Emergency Procedures](#emergency-procedures)
10. [Maintenance Tasks](#maintenance-tasks)

---

## Overview

This runbook provides comprehensive operational procedures for deploying, monitoring, and maintaining the Personal RAG Chatbot system in production environments.

### System Architecture

- **Frontend**: Gradio 5.x web interface
- **Backend**: Python 3.11 with Flask health endpoints
- **Vector Database**: Pinecone with gRPC
- **LLM Provider**: OpenRouter API
- **Monitoring**: Prometheus + Grafana
- **Caching**: Redis (optional)
- **Container Runtime**: Docker with security hardening

### Key Components

- **rag-chatbot-prod**: Main application container
- **monitoring**: Prometheus metrics collection
- **grafana**: Visualization and dashboards
- **redis**: Caching and session management (optional)

---

## Prerequisites

### System Requirements

- **OS**: Linux (Ubuntu 20.04+ recommended)
- **CPU**: 4+ cores
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 50GB available space
- **Network**: Stable internet connection

### Software Dependencies

- Docker Engine 24.0+
- Docker Compose 2.0+
- Git
- curl, wget, jq (for monitoring scripts)

### API Keys and Credentials

Required environment variables:
```bash
OPENROUTER_API_KEY=<your-key>
PINECONE_API_KEY=<your-key>
GRADIO_AUTH_USER=<admin-user>
GRADIO_AUTH_PASS=<secure-password>
```

### Network Configuration

- **Application Port**: 7860
- **Health Check Port**: 8000
- **Prometheus**: 9090
- **Grafana**: 3000
- **Redis**: 6379 (optional)

---

## Environment Setup

### 1. Clone Repository

```bash
git clone https://github.com/your-org/personal-rag-propo.git
cd personal-rag-propo
```

### 2. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit with your credentials
nano .env
```

### 3. Create Required Directories

```bash
mkdir -p logs data config/backups monitoring/grafana/data monitoring/prometheus
```

### 4. Set Proper Permissions

```bash
# Ensure logs directory is writable
chmod 755 logs
chmod 755 data
chmod 755 config/backups
```

---

## Deployment Procedures

### Standard Deployment

#### 1. Pre-deployment Checks

```bash
# Validate configuration
./scripts/validate_deployment.sh

# Check system resources
./scripts/check_system_resources.sh

# Backup current deployment (if exists)
./scripts/backup_current_deployment.sh
```

#### 2. Deploy Application

```bash
# Production deployment
docker-compose up -d rag-chatbot-prod

# Full stack deployment (with monitoring)
docker-compose --profile monitoring up -d

# Full stack with Redis
docker-compose --profile full up -d
```

#### 3. Post-deployment Validation

```bash
# Wait for services to be healthy
sleep 60

# Check application health
curl -f http://localhost:8000/health

# Check application accessibility
curl -f http://localhost:7860

# Validate monitoring
curl -f http://localhost:9090/-/healthy
curl -f http://localhost:3000/api/health
```

### Blue-Green Deployment

#### 1. Deploy New Version

```bash
# Deploy new version alongside existing
docker-compose up -d rag-chatbot-prod-new

# Wait for new version to be ready
sleep 120

# Switch traffic (if using load balancer)
# Update load balancer configuration
```

#### 2. Validation and Cutover

```bash
# Test new version
curl -f http://localhost:7861/health

# Run integration tests against new version
./scripts/run_integration_tests.sh http://localhost:7861

# Switch traffic to new version
# docker-compose up -d rag-chatbot-prod
# docker-compose stop rag-chatbot-prod-old
```

### Rollback Procedures

#### Emergency Rollback

```bash
# Stop current deployment
docker-compose stop rag-chatbot-prod

# Start previous version
docker-compose up -d rag-chatbot-prod-rollback

# Validate rollback
curl -f http://localhost:8000/health
```

#### Gradual Rollback

```bash
# Reduce traffic to new version gradually
# Monitor error rates and performance
# Complete rollback if issues persist
```

---

## Monitoring and Alerting

### Health Checks

#### Application Health

```bash
# Basic health check
curl -s http://localhost:8000/health | jq .

# Detailed health check
curl -s http://localhost:8000/health?detailed=true | jq .
```

#### System Health

```bash
# Check container status
docker-compose ps

# Check resource usage
docker stats

# Check logs
docker-compose logs --tail=50 rag-chatbot-prod
```

### Monitoring Dashboards

#### Grafana Access

- **URL**: http://localhost:3000
- **Default Credentials**: admin/admin
- **Main Dashboard**: Personal RAG Chatbot - Production Monitoring

#### Key Metrics to Monitor

1. **Application Health**
   - Service uptime
   - Response times (95th percentile)
   - Error rates

2. **System Resources**
   - CPU usage (< 80%)
   - Memory usage (< 85%)
   - Disk usage (< 90%)

3. **Business Metrics**
   - Request rate
   - User sessions
   - Document ingestion success rate

### Alert Response Procedures

#### Critical Alerts

**Alert: RAGChatbotDown**
```
1. Check application logs: docker-compose logs rag-chatbot-prod
2. Check system resources: docker stats
3. Attempt restart: docker-compose restart rag-chatbot-prod
4. If restart fails, initiate rollback procedure
5. Notify on-call engineer
```

**Alert: HighMemoryUsage**
```
1. Check memory usage: docker stats
2. Identify memory leaks in application logs
3. Restart service if memory > 90%
4. Scale resources if persistent issue
```

#### Warning Alerts

**Alert: HighResponseTime**
```
1. Check application performance logs
2. Analyze slow queries in logs
3. Optimize database queries if needed
4. Scale resources if load is high
```

---

## Security Procedures

### Access Control

#### Authentication Setup

```bash
# Enable authentication
export GRADIO_AUTH_ENABLED=true
export GRADIO_AUTH_USER=admin
export GRADIO_AUTH_PASS=<secure-password>

# Restart with authentication
docker-compose up -d rag-chatbot-prod
```

#### User Management

```bash
# Add new admin user (requires code change)
# Update environment variables
# Restart application
```

### Security Monitoring

#### Log Analysis

```bash
# View security logs
tail -f logs/security_audit.log

# Search for suspicious activity
grep "WARNING\|ERROR\|SECURITY" logs/security_audit.log

# Analyze rate limiting
grep "RATE_LIMIT" logs/security_audit.log | wc -l
```

#### Security Scans

```bash
# Run security scan
docker run --rm -v $(pwd):/app clair-scanner --ip localhost personal-rag-propo

# Check for vulnerabilities
docker scan personal-rag-propo
```

### Incident Response

#### Security Incident Procedure

1. **Detection**
   - Monitor security alerts
   - Review security logs
   - Check for unusual patterns

2. **Assessment**
   - Determine incident scope
   - Assess potential impact
   - Identify affected systems

3. **Containment**
   - Isolate affected systems
   - Block malicious traffic
   - Preserve evidence

4. **Recovery**
   - Restore from clean backup
   - Patch vulnerabilities
   - Monitor for recurrence

5. **Reporting**
   - Document incident details
   - Report to relevant stakeholders
   - Update security procedures

---

## Backup and Recovery

### Data Backup

#### Automated Backup

```bash
# Daily backup script
#!/bin/bash
BACKUP_DIR="/opt/backups/rag-chatbot"
DATE=$(date +%Y%m%d_%H%M%S)

# Backup application data
docker run --rm -v rag-chatbot_data:/data -v $BACKUP_DIR:/backup alpine tar czf /backup/data_$DATE.tar.gz -C /data .

# Backup configuration
cp .env $BACKUP_DIR/config_$DATE.env

# Backup logs
cp -r logs $BACKUP_DIR/logs_$DATE

# Clean old backups (keep last 7 days)
find $BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete
find $BACKUP_DIR -name "*.env" -mtime +7 -delete
find $BACKUP_DIR -name "logs_*" -mtime +7 -delete
```

#### Database Backup

```bash
# Backup Pinecone data (via API)
curl -X POST https://api.pinecone.io/indexes/personal-rag/export \
  -H "Api-Key: $PINECONE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"destination": "s3://your-backup-bucket"}'
```

### Disaster Recovery

#### Complete System Recovery

1. **Prepare Recovery Environment**
   ```bash
   # Set up new environment
   git clone <repository>
   cd personal-rag-propo
   cp backup/config.env .env
   ```

2. **Restore Data**
   ```bash
   # Restore from backup
   docker run --rm -v rag-chatbot_data:/data -v /opt/backups:/backup alpine tar xzf /backup/latest_data.tar.gz -C /data
   ```

3. **Deploy Application**
   ```bash
   docker-compose up -d
   ```

4. **Validate Recovery**
   ```bash
   curl -f http://localhost:8000/health
   # Run validation tests
   ```

#### Recovery Time Objectives (RTO)

- **RTO**: 4 hours for complete recovery
- **RPO**: 1 hour data loss tolerance
- **Recovery Point**: Last successful backup

---

## Troubleshooting

### Common Issues

#### Application Won't Start

```bash
# Check logs
docker-compose logs rag-chatbot-prod

# Check environment variables
docker-compose exec rag-chatbot-prod env

# Validate configuration
docker-compose exec rag-chatbot-prod python -c "from src.config import AppConfig; print('Config OK')"
```

#### High Memory Usage

```bash
# Check memory usage
docker stats

# Check application logs for memory leaks
docker-compose logs --tail=100 rag-chatbot-prod | grep -i memory

# Restart service
docker-compose restart rag-chatbot-prod
```

#### Slow Response Times

```bash
# Check system resources
docker stats

# Check application performance
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:8000/health

# Analyze logs for bottlenecks
docker-compose logs rag-chatbot-prod | grep -E "duration|time"
```

#### Database Connection Issues

```bash
# Check Pinecone connectivity
curl -H "Api-Key: $PINECONE_API_KEY" https://api.pinecone.io/indexes

# Check network connectivity
docker-compose exec rag-chatbot-prod ping -c 3 api.pinecone.io

# Validate API key
docker-compose exec rag-chatbot-prod python -c "import pinecone; pinecone.init(api_key='$PINECONE_API_KEY')"
```

### Log Analysis

#### Application Logs

```bash
# View recent logs
docker-compose logs --tail=50 rag-chatbot-prod

# Follow logs in real-time
docker-compose logs -f rag-chatbot-prod

# Search for specific errors
docker-compose logs rag-chatbot-prod | grep ERROR

# Export logs for analysis
docker-compose logs rag-chatbot-prod > app_logs_$(date +%Y%m%d).log
```

#### System Logs

```bash
# System logs
journalctl -u docker -f

# Container logs
docker logs rag-chatbot-prod

# Docker daemon logs
docker system events --since '1h'
```

---

## Emergency Procedures

### Service Outage Response

#### Immediate Actions

1. **Assess Situation**
   ```bash
   # Check service status
   docker-compose ps

   # Check health endpoint
   curl http://localhost:8000/health

   # Check system resources
   docker stats
   ```

2. **Attempt Quick Recovery**
   ```bash
   # Restart service
   docker-compose restart rag-chatbot-prod

   # Check logs for errors
   docker-compose logs --tail=20 rag-chatbot-prod
   ```

3. **Escalate if Needed**
   - Notify on-call engineer
   - Check monitoring alerts
   - Prepare rollback if necessary

#### Communication Plan

- **Internal**: Slack/Teams notification to devops channel
- **External**: Status page update if affecting users
- **Stakeholders**: Email notification for major incidents

### Data Loss Recovery

#### Immediate Response

1. **Stop All Operations**
   ```bash
   docker-compose stop
   ```

2. **Assess Data Loss**
   ```bash
   # Check available backups
   ls -la /opt/backups/

   # Determine last good backup
   ls -lt /opt/backups/*.tar.gz | head -5
   ```

3. **Execute Recovery**
   ```bash
   # Follow backup recovery procedure
   # Validate data integrity
   # Test application functionality
   ```

---

## Maintenance Tasks

### Regular Maintenance

#### Daily Tasks

```bash
# Check system health
curl -f http://localhost:8000/health

# Monitor resource usage
docker stats --no-stream

# Review error logs
docker-compose logs --since "24h" | grep -i error
```

#### Weekly Tasks

```bash
# Update dependencies
docker-compose pull

# Rotate logs
./scripts/rotate_logs.sh

# Check disk usage
df -h

# Validate backups
./scripts/validate_backups.sh
```

#### Monthly Tasks

```bash
# Security updates
docker-compose build --no-cache rag-chatbot-prod

# Performance optimization
./scripts/performance_audit.sh

# Capacity planning
./scripts/capacity_planning.sh
```

### Scheduled Updates

#### Application Updates

```bash
# Update application code
git pull origin main

# Build new image
docker-compose build rag-chatbot-prod

# Deploy with zero downtime
docker-compose up -d rag-chatbot-prod

# Validate deployment
curl -f http://localhost:8000/health
```

#### Security Patches

```bash
# Update base images
docker-compose pull

# Rebuild with latest security patches
docker-compose build --no-cache

# Deploy updated services
docker-compose up -d

# Validate security posture
./scripts/security_audit.sh
```

---

## Contact Information

### On-Call Schedule

- **Primary**: DevOps Team Lead
- **Secondary**: Senior Developer
- **Escalation**: Engineering Manager

### Support Contacts

- **Technical Support**: devops@company.com
- **Security Issues**: security@company.com
- **Business Impact**: business@company.com

### External Resources

- **Monitoring**: http://localhost:3000
- **Logs**: /var/log/rag-chatbot/
- **Documentation**: https://github.com/your-org/personal-rag-propo/docs/
- **Issues**: https://github.com/your-org/personal-rag-propo/issues

---

## Change Log

| Date | Version | Changes |
|------|---------|---------|
| 2025-08-30 | 2.0.0 | Initial production runbook for 2025 stack |
| 2025-08-30 | 2.0.0 | Added MoE monitoring and security procedures |

---

**Document Owner**: DevOps Team
**Review Date**: Quarterly
**Approval**: Engineering Manager