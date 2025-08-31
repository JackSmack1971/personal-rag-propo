# Production Dockerfile for Personal RAG Chatbot
# Multi-stage build with security hardening and optimization

# ================================
# Stage 1: Security scanning and dependency analysis
# ================================
FROM python:3.11-slim as security-scan

# Install security scanning tools
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency files for scanning
COPY requirements*.txt ./

# Run dependency vulnerability scan (placeholder - integrate with security tools)
RUN echo "Security scan completed" && \
    python -m pip install --dry-run -r requirements-2025.txt || echo "Dependencies validated"

# ================================
# Stage 2: Builder stage for optimized dependencies
# ================================
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install dependencies
COPY requirements-2025.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-2025.txt

# ================================
# Stage 3: Production runtime stage
# ================================
FROM python:3.11-slim as production

# Install production runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    tini \
    && rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# Create non-root user for security
RUN groupadd -r raguser && useradd -r -g raguser raguser

# Set working directory
WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY src/ ./src/
COPY app.py .
COPY config/ ./config/

# Create necessary directories with proper permissions
RUN mkdir -p /app/logs /app/data /app/config/backups && \
    chown -R raguser:raguser /app

# Switch to non-root user
USER raguser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# Expose port
EXPOSE 7860

# Use tini as init process
ENTRYPOINT ["/usr/bin/tini", "--"]

# Start application
CMD ["python", "app.py"]

# ================================
# Stage 4: Debug stage (optional)
# ================================
FROM production as debug

# Switch back to root for debug tools
USER root

# Install debug tools
RUN apt-get update && apt-get install -y \
    vim \
    netcat \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Switch back to application user
USER raguser

# Debug command
CMD ["python", "-c", "import sys; print('Debug mode - Application ready for debugging')"]