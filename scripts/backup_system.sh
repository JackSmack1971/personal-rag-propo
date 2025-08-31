#!/bin/bash

# System Backup Script for Personal RAG Chatbot
# Version: 2.0.0
# Last Updated: August 30, 2025

set -e

# Configuration
BACKUP_ROOT="/opt/backups/rag-chatbot"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="rag_backup_$TIMESTAMP"
BACKUP_DIR="$BACKUP_ROOT/$BACKUP_NAME"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging
LOG_FILE="$BACKUP_ROOT/backup_$TIMESTAMP.log"

log() {
    local level=$1
    local message=$2
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] [$level] $message" >> "$LOG_FILE"
    echo "[$level] $message"
}

# Create backup directories
setup_backup_dirs() {
    log "INFO" "Setting up backup directories"

    mkdir -p "$BACKUP_DIR"
    mkdir -p "$BACKUP_DIR/data"
    mkdir -p "$BACKUP_DIR/config"
    mkdir -p "$BACKUP_DIR/logs"
    mkdir -p "$BACKUP_DIR/database"

    # Set secure permissions
    chmod 700 "$BACKUP_DIR"
    chmod 600 "$BACKUP_DIR"/*.*
}

# Backup application data
backup_application_data() {
    log "INFO" "Backing up application data"

    echo -e "${BLUE}Backing up application data...${NC}"

    # Stop containers for consistent backup
    docker-compose stop rag-chatbot-prod 2>/dev/null || log "WARNING" "Could not stop application container"

    # Backup data volume
    if docker volume ls | grep -q rag-chatbot_data; then
        log "INFO" "Backing up data volume"
        docker run --rm -v rag-chatbot_data:/source -v "$BACKUP_DIR/data":/backup alpine tar czf "/backup/data.tar.gz" -C /source . || log "ERROR" "Failed to backup data volume"
    else
        log "WARNING" "Data volume not found"
    fi

    # Restart containers
    docker-compose start rag-chatbot-prod 2>/dev/null || log "WARNING" "Could not restart application container"
}

# Backup configuration
backup_configuration() {
    log "INFO" "Backing up configuration"

    echo -e "${BLUE}Backing up configuration...${NC}"

    # Backup environment file (excluding sensitive data)
    if [[ -f ".env" ]]; then
        # Create sanitized backup (remove sensitive values)
        grep -v "API_KEY\|SECRET\|PASSWORD" .env > "$BACKUP_DIR/config/environment.env" || log "ERROR" "Failed to backup environment config"
        log "INFO" "Environment configuration backed up (sanitized)"
    fi

    # Backup docker-compose configuration
    cp docker-compose.yml "$BACKUP_DIR/config/" 2>/dev/null || log "WARNING" "Docker compose config not found"

    # Backup application configuration
    cp -r config/ "$BACKUP_DIR/config/app/" 2>/dev/null || log "WARNING" "Application config not found"
}

# Backup logs
backup_logs() {
    log "INFO" "Backing up logs"

    echo -e "${BLUE}Backing up logs...${NC}"

    # Backup application logs
    if [[ -d "logs" ]]; then
        cp -r logs "$BACKUP_DIR/logs/app" || log "ERROR" "Failed to backup application logs"
    fi

    # Backup container logs
    docker-compose logs > "$BACKUP_DIR/logs/containers_$TIMESTAMP.log" 2>/dev/null || log "WARNING" "Failed to backup container logs"

    # Backup system logs
    journalctl --since "1 day ago" > "$BACKUP_DIR/logs/system_$TIMESTAMP.log" 2>/dev/null || log "WARNING" "Failed to backup system logs"
}

# Backup database/vector store
backup_database() {
    log "INFO" "Backing up database/vector store"

    echo -e "${BLUE}Backing up database...${NC}"

    # For Pinecone, we can export data via API
    if [[ -n "${PINECONE_API_KEY:-}" ]]; then
        log "INFO" "Attempting Pinecone data export"

        # Note: This is a placeholder for actual Pinecone export
        # In production, you would use Pinecone's export API
        curl -X POST "https://api.pinecone.io/indexes/personal-rag/export" \
            -H "Api-Key: $PINECONE_API_KEY" \
            -H "Content-Type: application/json" \
            -d "{\"destination\": \"s3://your-backup-bucket/$BACKUP_NAME\"}" \
            > "$BACKUP_DIR/database/pinecone_export_$TIMESTAMP.json" 2>/dev/null || log "WARNING" "Pinecone export failed"
    else
        log "WARNING" "Pinecone API key not configured"
    fi

    # Backup Redis data if available
    if docker-compose ps redis | grep -q "Up"; then
        log "INFO" "Backing up Redis data"
        docker exec rag-redis redis-cli SAVE || log "WARNING" "Redis SAVE failed"
        docker cp rag-redis:/data/dump.rdb "$BACKUP_DIR/database/redis_$TIMESTAMP.rdb" 2>/dev/null || log "WARNING" "Redis backup failed"
    fi
}

# Create backup manifest
create_manifest() {
    log "INFO" "Creating backup manifest"

    local manifest="$BACKUP_DIR/manifest.json"

    cat > "$manifest" << EOF
{
  "backup_name": "$BACKUP_NAME",
  "timestamp": "$TIMESTAMP",
  "created_at": "$(date -Iseconds)",
  "version": "2.0.0",
  "components": {
    "application_data": $([[ -f "$BACKUP_DIR/data/data.tar.gz" ]] && echo "true" || echo "false"),
    "configuration": $([[ -d "$BACKUP_DIR/config" ]] && echo "true" || echo "false"),
    "logs": $([[ -d "$BACKUP_DIR/logs" ]] && echo "true" || echo "false"),
    "database": $([[ -d "$BACKUP_DIR/database" ]] && echo "true" || echo "false")
  },
  "size": "$(du -sh "$BACKUP_DIR" | cut -f1)",
  "files": $(find "$BACKUP_DIR" -type f | wc -l),
  "integrity_check": "pending"
}
EOF

    log "INFO" "Backup manifest created: $manifest"
}

# Verify backup integrity
verify_backup() {
    log "INFO" "Verifying backup integrity"

    echo -e "${BLUE}Verifying backup integrity...${NC}"

    local integrity_ok=true

    # Check if all expected files exist
    [[ -f "$BACKUP_DIR/manifest.json" ]] || integrity_ok=false
    [[ -d "$BACKUP_DIR/config" ]] || integrity_ok=false
    [[ -d "$BACKUP_DIR/logs" ]] || integrity_ok=false

    # Verify archive integrity
    if [[ -f "$BACKUP_DIR/data/data.tar.gz" ]]; then
        tar -tzf "$BACKUP_DIR/data/data.tar.gz" > /dev/null 2>&1 || integrity_ok=false
    fi

    if [[ "$integrity_ok" == "true" ]]; then
        log "INFO" "Backup integrity check passed"
        echo -e "${GREEN}Backup integrity: OK${NC}"
    else
        log "ERROR" "Backup integrity check failed"
        echo -e "${RED}Backup integrity: FAILED${NC}"
        return 1
    fi
}

# Cleanup old backups
cleanup_old_backups() {
    log "INFO" "Cleaning up old backups"

    echo -e "${BLUE}Cleaning up old backups...${NC}"

    # Keep only last 7 daily backups
    local daily_backups=$(find "$BACKUP_ROOT" -name "rag_backup_*" -type d -mtime +7 | wc -l)
    if [[ $daily_backups -gt 0 ]]; then
        find "$BACKUP_ROOT" -name "rag_backup_*" -type d -mtime +7 -exec rm -rf {} \; 2>/dev/null || log "WARNING" "Failed to cleanup old backups"
        log "INFO" "Cleaned up $daily_backups old backups"
    fi

    # Keep only last 4 weekly backups (older than 7 days but within 30 days)
    local weekly_backups=$(find "$BACKUP_ROOT" -name "rag_backup_*" -type d -mtime +30 | wc -l)
    if [[ $weekly_backups -gt 0 ]]; then
        find "$BACKUP_ROOT" -name "rag_backup_*" -type d -mtime +30 -exec rm -rf {} \; 2>/dev/null || log "WARNING" "Failed to cleanup old weekly backups"
        log "INFO" "Cleaned up $weekly_backups old weekly backups"
    fi
}

# Compress backup
compress_backup() {
    log "INFO" "Compressing backup"

    echo -e "${BLUE}Compressing backup...${NC}"

    local archive_name="$BACKUP_ROOT/${BACKUP_NAME}.tar.gz"

    cd "$BACKUP_ROOT"
    tar -czf "$archive_name" "$BACKUP_NAME" || log "ERROR" "Failed to compress backup"

    # Remove uncompressed backup directory
    rm -rf "$BACKUP_DIR"

    log "INFO" "Backup compressed: $archive_name"
    echo -e "${GREEN}Backup archive created: $archive_name${NC}"
}

# Send notifications
send_notifications() {
    log "INFO" "Sending backup notifications"

    local status=$1
    local size=$(du -sh "$BACKUP_ROOT/${BACKUP_NAME}.tar.gz" 2>/dev/null | cut -f1)

    # Email notification (if configured)
    if command -v mail &> /dev/null && [[ -n "${BACKUP_NOTIFICATION_EMAIL:-}" ]]; then
        echo "Backup completed at $(date)
Status: $status
Size: $size
Location: $BACKUP_ROOT/${BACKUP_NAME}.tar.gz
Log: $LOG_FILE" | mail -s "RAG Chatbot Backup $status" "$BACKUP_NOTIFICATION_EMAIL"
    fi

    # Slack notification (if configured)
    if [[ -n "${SLACK_WEBHOOK_URL:-}" ]]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"RAG Chatbot Backup $status - Size: $size\"}" \
            "$SLACK_WEBHOOK_URL" 2>/dev/null || log "WARNING" "Slack notification failed"
    fi
}

# Main backup function
main() {
    echo -e "${BLUE}Personal RAG Chatbot - System Backup${NC}"
    echo "Started at: $(date)"
    echo "Backup location: $BACKUP_DIR"
    echo

    local start_time=$(date +%s)
    local status="SUCCESS"

    log "INFO" "Starting system backup"

    # Execute backup steps
    setup_backup_dirs
    backup_application_data
    backup_configuration
    backup_logs
    backup_database
    create_manifest

    if verify_backup; then
        compress_backup
        cleanup_old_backups
    else
        status="FAILED"
        log "ERROR" "Backup verification failed"
    fi

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    log "INFO" "Backup completed in ${duration}s with status: $status"

    # Send notifications
    send_notifications "$status"

    echo
    echo -e "${GREEN}Backup completed${NC}"
    echo "Status: $status"
    echo "Duration: ${duration}s"
    echo "Archive: $BACKUP_ROOT/${BACKUP_NAME}.tar.gz"
    echo "Log: $LOG_FILE"
}

# Error handling
trap 'echo -e "\n${RED}Backup failed at $(date)${NC}"; exit 1' ERR

# Run main function
main "$@"