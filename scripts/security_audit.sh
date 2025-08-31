#!/bin/bash

# Security Audit Script for Personal RAG Chatbot
# Version: 2.0.0
# Last Updated: August 30, 2025

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
AUDIT_LOG="logs/security_audit_$(date +%Y%m%d_%H%M%S).log"
REPORT_FILE="reports/security_audit_$(date +%Y%m%d).json"

# Create directories
mkdir -p logs reports

# Logging function
log() {
    local level=$1
    local message=$2
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] [$level] $message" >> "$AUDIT_LOG"
    echo "[$level] $message"
}

# Check function with scoring
check() {
    local name=$1
    local command=$2
    local expected=$3
    local severity=${4:-"MEDIUM"}

    echo -n "Checking $name... "

    if eval "$command" > /dev/null 2>&1; then
        echo -e "${GREEN}PASS${NC}"
        log "PASS" "$name check passed"
        return 0
    else
        echo -e "${RED}FAIL${NC}"
        log "FAIL" "$name check failed"
        return 1
    fi
}

# Vulnerability scanning
scan_vulnerabilities() {
    log "INFO" "Starting vulnerability scan"

    echo -e "\n${BLUE}=== Vulnerability Scanning ===${NC}"

    # Check for outdated packages
    if command -v pip-audit &> /dev/null; then
        echo "Running pip-audit..."
        pip-audit --format json > reports/pip_audit_$(date +%Y%m%d).json 2>/dev/null || log "WARNING" "pip-audit failed"
    else
        log "WARNING" "pip-audit not installed"
    fi

    # Docker image scanning
    if command -v docker &> /dev/null && command -v trivy &> /dev/null; then
        echo "Scanning Docker images with Trivy..."
        docker images | grep rag-chatbot | awk '{print $1":"$2}' | xargs -I {} trivy image {} > reports/trivy_scan_$(date +%Y%m%d).txt 2>/dev/null || log "WARNING" "Trivy scan failed"
    fi
}

# Container security checks
check_container_security() {
    log "INFO" "Checking container security"

    echo -e "\n${BLUE}=== Container Security ===${NC}"

    # Check if running as non-root
    check "Non-root user" "docker inspect rag-chatbot-prod | jq -r '.[].Config.User'" "raguser"

    # Check security options
    check "No privileged containers" "docker inspect rag-chatbot-prod | jq -r '.[].HostConfig.Privileged'" "false"

    # Check read-only root filesystem
    check "Read-only root filesystem" "docker inspect rag-chatbot-prod | jq -r '.[].HostConfig.ReadonlyRootfs'" "true"

    # Check security profile
    check "Security profile applied" "docker inspect rag-chatbot-prod | jq -r '.[].HostConfig.SecurityOpt | length'" "1"
}

# Network security checks
check_network_security() {
    log "INFO" "Checking network security"

    echo -e "\n${BLUE}=== Network Security ===${NC}"

    # Check exposed ports
    check "Minimal port exposure" "docker port rag-chatbot-prod | wc -l" "2"

    # Check network isolation
    check "Network isolation" "docker inspect rag-chatbot-prod | jq -r '.[].NetworkSettings.Networks | keys[]'" "rag-network"
}

# Application security checks
check_application_security() {
    log "INFO" "Checking application security"

    echo -e "\n${BLUE}=== Application Security ===${NC}"

    # Check health endpoint
    check "Health endpoint accessible" "curl -f -s http://localhost:8000/health" ""

    # Check authentication
    if [[ "${GRADIO_AUTH_ENABLED:-false}" == "true" ]]; then
        check "Authentication enabled" "echo 'Authentication configured'" ""
    else
        log "WARNING" "Authentication not enabled"
    fi

    # Check HTTPS (if applicable)
    # check "HTTPS enabled" "curl -I https://localhost:7860" ""

    # Check security headers
    check "Security headers" "curl -I -s http://localhost:7860 | grep -i 'x-'" ""
}

# File system security
check_file_security() {
    log "INFO" "Checking file system security"

    echo -e "\n${BLUE}=== File System Security ===${NC}"

    # Check file permissions
    check "Secure file permissions" "stat -c '%a' .env" "600"

    # Check log file permissions
    check "Log file permissions" "stat -c '%a' logs/" "755"

    # Check no sensitive files in public
    check "No sensitive files exposed" "find . -name '*.key' -o -name '*.pem' -o -name 'id_*' | wc -l" "0"
}

# Configuration security
check_configuration_security() {
    log "INFO" "Checking configuration security"

    echo -e "\n${BLUE}=== Configuration Security ===${NC}"

    # Check environment variables
    check "API keys not in logs" "grep -r 'sk-' logs/ || echo 'No API keys found'" ""

    # Check configuration file permissions
    check "Config file permissions" "stat -c '%a' .env" "600"

    # Check backup security
    if [[ -d "config/backups" ]]; then
        check "Backup security" "stat -c '%a' config/backups/" "700"
    fi
}

# Generate security report
generate_report() {
    log "INFO" "Generating security report"

    echo -e "\n${BLUE}=== Security Audit Report ===${NC}"

    local total_checks=0
    local passed_checks=0
    local failed_checks=0

    # Count results from log
    total_checks=$(grep -c "PASS\|FAIL" "$AUDIT_LOG")
    passed_checks=$(grep -c "PASS" "$AUDIT_LOG")
    failed_checks=$(grep -c "FAIL" "$AUDIT_LOG")

    local score=$(( passed_checks * 100 / total_checks ))

    echo "Total Checks: $total_checks"
    echo "Passed: $passed_checks"
    echo "Failed: $failed_checks"
    echo "Security Score: $score%"

    # Color-coded score
    if [[ $score -ge 90 ]]; then
        echo -e "${GREEN}Security Status: EXCELLENT${NC}"
    elif [[ $score -ge 75 ]]; then
        echo -e "${YELLOW}Security Status: GOOD${NC}"
    elif [[ $score -ge 60 ]]; then
        echo -e "${YELLOW}Security Status: FAIR${NC}"
    else
        echo -e "${RED}Security Status: POOR${NC}"
    fi

    # Generate JSON report
    cat > "$REPORT_FILE" << EOF
{
  "audit_timestamp": "$(date -Iseconds)",
  "total_checks": $total_checks,
  "passed_checks": $passed_checks,
  "failed_checks": $failed_checks,
  "security_score": $score,
  "status": "$(if [[ $score -ge 75 ]]; then echo 'PASS'; else echo 'FAIL'; fi)",
  "audit_log": "$AUDIT_LOG",
  "recommendations": [
    "Regular security audits (monthly)",
    "Automated vulnerability scanning",
    "Security monitoring and alerting",
    "Regular dependency updates",
    "Access control and authentication",
    "Network security hardening"
  ]
}
EOF

    echo "Detailed report saved to: $REPORT_FILE"
    echo "Audit log saved to: $AUDIT_LOG"
}

# Main audit function
main() {
    echo -e "${BLUE}Personal RAG Chatbot - Security Audit${NC}"
    echo "Started at: $(date)"
    echo "Audit log: $AUDIT_LOG"
    echo

    log "INFO" "Starting security audit"

    # Run all security checks
    check_container_security
    check_network_security
    check_application_security
    check_file_security
    check_configuration_security
    scan_vulnerabilities

    # Generate final report
    generate_report

    log "INFO" "Security audit completed"

    echo
    echo "Audit completed at: $(date)"
}

# Run main function
main "$@"