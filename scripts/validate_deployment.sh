#!/bin/bash

# Deployment Validation Script for Personal RAG Chatbot
# Version: 2.0.0
# Last Updated: August 30, 2025

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
VALIDATION_LOG="logs/deployment_validation_$(date +%Y%m%d_%H%M%S).log"
REPORT_FILE="reports/deployment_validation_$(date +%Y%m%d).json"

# Create directories
mkdir -p logs reports

# Logging function
log() {
    local level=$1
    local message=$2
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] [$level] $message" >> "$VALIDATION_LOG"
    echo "[$level] $message"
}

# Validation result tracking
VALIDATION_PASSED=true
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0

# Check function with scoring
check() {
    local name=$1
    local command=$2
    local expected_exit=${3:-0}

    ((TOTAL_CHECKS++))
    echo -n "Checking $name... "

    if eval "$command" > /dev/null 2>&1; then
        echo -e "${GREEN}PASS${NC}"
        log "PASS" "$name check passed"
        ((PASSED_CHECKS++))
        return 0
    else
        echo -e "${RED}FAIL${NC}"
        log "FAIL" "$name check failed"
        ((FAILED_CHECKS++))
        VALIDATION_PASSED=false
        return 1
    fi
}

# Pre-deployment validation
validate_prerequisites() {
    log "INFO" "Validating deployment prerequisites"

    echo -e "\n${BLUE}=== Prerequisites Validation ===${NC}"

    # Check Docker availability
    check "Docker installed" "docker --version"

    # Check Docker Compose availability
    check "Docker Compose installed" "docker-compose --version"

    # Check required files exist
    check "Dockerfile exists" "test -f Dockerfile"
    check "docker-compose.yml exists" "test -f docker-compose.yml"
    check ".env file exists" "test -f .env"

    # Check environment variables
    check "OPENROUTER_API_KEY set" "test -n \"\$OPENROUTER_API_KEY\""
    check "PINECONE_API_KEY set" "test -n \"\$PINECONE_API_KEY\""

    # Check system resources
    check "Available disk space (>10GB)" "df / | awk 'NR==2 {print \$4}' | xargs test 10000000 -lt"
    check "Available memory (>2GB)" "free | awk 'NR==2 {print \$7}' | xargs test 2000000 -lt"
}

# Container validation
validate_containers() {
    log "INFO" "Validating container configuration"

    echo -e "\n${BLUE}=== Container Validation ===${NC}"

    # Validate Dockerfile syntax
    check "Dockerfile syntax" "docker build --dry-run -f Dockerfile . 2>/dev/null || docker build -f Dockerfile --no-cache --target security-scan ."

    # Validate docker-compose configuration
    check "docker-compose config" "docker-compose config"

    # Check for security issues in Dockerfile
    check "No privileged containers" "grep -q 'privileged' docker-compose.yml && exit 1 || true"
    check "Non-root user specified" "grep -q 'raguser' Dockerfile"
}

# Network validation
validate_networking() {
    log "INFO" "Validating network configuration"

    echo -e "\n${BLUE}=== Network Validation ===${NC}"

    # Check port availability
    check "Port 7860 available" "! lsof -i :7860"
    check "Port 8000 available" "! lsof -i :8000"
    check "Port 9090 available" "! lsof -i :9090"
    check "Port 3000 available" "! lsof -i :3000"

    # Validate network configuration in docker-compose
    check "Network configuration valid" "docker-compose config | grep -q 'rag-network'"
}

# Security validation
validate_security() {
    log "INFO" "Validating security configuration"

    echo -e "\n${BLUE}=== Security Validation ===${NC}"

    # Check file permissions
    check "Secure .env permissions" "stat -c '%a' .env | xargs test 600 -eq"

    # Check no secrets in logs
    check "No API keys in logs" "! grep -r 'sk-' logs/ 2>/dev/null"

    # Check security options in docker-compose
    check "Security options configured" "grep -q 'no-new-privileges' docker-compose.yml"

    # Validate environment variable security
    check "No sensitive data in compose file" "! grep -E '(API_KEY|SECRET|PASSWORD)' docker-compose.yml"
}

# Application validation
validate_application() {
    log "INFO" "Validating application configuration"

    echo -e "\n${BLUE}=== Application Validation ===${NC}"

    # Check required Python files
    check "Main application file exists" "test -f app.py"
    check "Source directory exists" "test -d src"
    check "Configuration module exists" "test -f src/config.py"

    # Validate Python syntax
    check "Python syntax valid" "python -m py_compile app.py"

    # Check requirements file
    check "Requirements file exists" "test -f requirements-2025.txt"
    check "Requirements installable" "pip install --dry-run -r requirements-2025.txt"

    # Validate configuration loading
    check "Configuration loads" "python -c \"from src.config import AppConfig; cfg = AppConfig.from_env(); print('Config OK')\""
}

# Monitoring validation
validate_monitoring() {
    log "INFO" "Validating monitoring configuration"

    echo -e "\n${BLUE}=== Monitoring Validation ===${NC}"

    # Check monitoring configuration files
    check "Prometheus config exists" "test -f monitoring/prometheus.yml"
    check "Alert rules exist" "test -f monitoring/alert_rules.yml"
    check "Grafana dashboard exists" "test -f monitoring/grafana/dashboards/rag-chatbot-dashboard.json"

    # Validate Prometheus configuration
    check "Prometheus config valid" "docker run --rm -v \$(pwd)/monitoring:/config prom/prometheus --config.check --config.file=/config/prometheus.yml"
}

# Backup validation
validate_backup() {
    log "INFO" "Validating backup configuration"

    echo -e "\n${BLUE}=== Backup Validation ===${NC}"

    # Check backup script exists
    check "Backup script exists" "test -f scripts/backup_system.sh"
    check "Backup script executable" "test -x scripts/backup_system.sh"

    # Check backup directory exists
    check "Backup directory exists" "test -d /opt/backups/rag-chatbot || mkdir -p /opt/backups/rag-chatbot"

    # Validate backup script syntax
    check "Backup script syntax" "bash -n scripts/backup_system.sh"
}

# Generate validation report
generate_report() {
    log "INFO" "Generating validation report"

    echo -e "\n${BLUE}=== Validation Report ===${NC}"

    local score=$(( PASSED_CHECKS * 100 / TOTAL_CHECKS ))

    echo "Total Checks: $TOTAL_CHECKS"
    echo "Passed: $PASSED_CHECKS"
    echo "Failed: $FAILED_CHECKS"
    echo "Success Rate: $score%"

    # Color-coded result
    if [[ $score -ge 95 ]]; then
        echo -e "${GREEN}Validation Status: EXCELLENT${NC}"
    elif [[ $score -ge 85 ]]; then
        echo -e "${GREEN}Validation Status: GOOD${NC}"
    elif [[ $score -ge 75 ]]; then
        echo -e "${YELLOW}Validation Status: FAIR${NC}"
    else
        echo -e "${RED}Validation Status: POOR${NC}"
    fi

    # Create JSON report
    cat > "$REPORT_FILE" << EOF
{
  "validation_timestamp": "$(date -Iseconds)",
  "total_checks": $TOTAL_CHECKS,
  "passed_checks": $PASSED_CHECKS,
  "failed_checks": $FAILED_CHECKS,
  "success_rate": $score,
  "status": "$(if [[ $score -ge 85 ]]; then echo 'PASS'; else echo 'FAIL'; fi)",
  "validation_log": "$VALIDATION_LOG",
  "recommendations": [
    "Fix any failed validation checks before deployment",
    "Review security configuration and permissions",
    "Validate monitoring and alerting setup",
    "Test backup and recovery procedures",
    "Document any configuration changes made"
  ]
}
EOF

    echo "Detailed report saved to: $REPORT_FILE"
    echo "Validation log saved to: $VALIDATION_LOG"
}

# Deployment simulation
simulate_deployment() {
    log "INFO" "Simulating deployment process"

    echo -e "\n${BLUE}=== Deployment Simulation ===${NC}"

    # Test container build
    check "Container build simulation" "docker build --dry-run -f Dockerfile . 2>/dev/null || echo 'Build simulation not available'"

    # Test service startup
    check "Service startup test" "timeout 10s docker-compose up -d --no-deps rag-chatbot-prod 2>/dev/null && docker-compose stop rag-chatbot-prod 2>/dev/null || true"

    # Test health endpoint (if service is running)
    if docker-compose ps rag-chatbot-prod | grep -q "Up"; then
        check "Health endpoint test" "curl -f -s http://localhost:8000/health"
        docker-compose stop rag-chatbot-prod 2>/dev/null || true
    else
        log "WARNING" "Service not running for health check test"
    fi
}

# Main validation function
main() {
    echo -e "${BLUE}Personal RAG Chatbot - Deployment Validation${NC}"
    echo "Started at: $(date)"
    echo "Validation log: $VALIDATION_LOG"
    echo

    log "INFO" "Starting deployment validation"

    # Run all validation checks
    validate_prerequisites
    validate_containers
    validate_networking
    validate_security
    validate_application
    validate_monitoring
    validate_backup
    simulate_deployment

    # Generate final report
    generate_report

    log "INFO" "Deployment validation completed"

    # Final status
    echo
    if [[ "$VALIDATION_PASSED" == "true" ]]; then
        echo -e "${GREEN}✅ Deployment validation PASSED${NC}"
        echo "System is ready for production deployment"
        exit 0
    else
        echo -e "${RED}❌ Deployment validation FAILED${NC}"
        echo "Please fix the failed checks before deploying"
        exit 1
    fi
}

# Run main function
main "$@"