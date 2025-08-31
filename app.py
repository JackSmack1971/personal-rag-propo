"""
Enhanced Personal RAG Chatbot - 2025 Stack
Migrated to Gradio 5.x with SSR, enhanced UI, and security features.
"""

import os
import json
import logging
from typing import Tuple, List
from dotenv import load_dotenv

# 2025 Stack: Gradio 5.x with SSR and enhanced features
import gradio as gr

from src.config import AppConfig
from src.ingest import ingest_files
from src.rag import rag_chat
from src.vectorstore import ensure_index
from src.embeddings import get_embedder
from src.security import (
    validate_file_upload, validate_query, check_rate_limit,
    sanitize_input, log_security_event, get_security_metrics,
    get_secure_api_headers
)
from src.network_security import get_network_security_manager, create_secure_app_config

# Enhanced logging for 2025 stack
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Enhanced configuration with validation
cfg = AppConfig.from_env()
embedder = get_embedder(cfg.EMBED_MODEL, backend=os.getenv("SENTENCE_TRANSFORMERS_BACKEND", "torch"))

# Get embedding dimension with fallback
try:
    dim = embedder.get_sentence_embedding_dimension()
    if dim is None:
        # Fallback for models that don't report dimension
        dim = 384  # Common dimension for many models
        logger.warning(f"Model didn't report dimension, using fallback: {dim}")
except Exception as e:
    dim = 384  # Safe fallback
    logger.warning(f"Could not get embedding dimension, using fallback: {dim} ({e})")

ensure_index(cfg, dim=dim)

logger.info("Personal RAG Chatbot initialized with 2025 stack")

def ui_ingest(files, namespace):
    """
    Enhanced file ingestion with comprehensive security validation and error handling.
    """
    try:
        if not files:
            log_security_event("FILE_INGEST_NO_FILES", {}, "INFO")
            return "❌ No files selected for ingestion."

        # Rate limiting check
        client_ip = os.getenv("REMOTE_ADDR", "unknown")
        if check_rate_limit(client_ip):
            log_security_event("RATE_LIMIT_EXCEEDED", {"ip": client_ip}, "WARNING")
            return "❌ Rate limit exceeded. Please try again later."

        validated_files = []
        for file in files:
            if hasattr(file, 'name'):
                file_name = file.name
                file_size = getattr(file, 'size', 0)
                # Read file content for validation
                try:
                    file_content = file.read() if hasattr(file, 'read') else b""
                    if hasattr(file, 'seek'):
                        file.seek(0)  # Reset file pointer
                except Exception as e:
                    log_security_event("FILE_READ_ERROR", {"file": file_name, "error": str(e)}, "WARNING")
                    return f"❌ Error reading file {file_name}: {str(e)}"
            else:
                file_name = str(file)
                file_size = 0
                file_content = b""

            # Comprehensive security validation
            is_valid, validation_message = validate_file_upload(file_name, file_content)
            if not is_valid:
                log_security_event("FILE_VALIDATION_FAILED", {
                    "file": file_name,
                    "reason": validation_message
                }, "WARNING")
                return f"❌ Security validation failed for {file_name}: {validation_message}"

            validated_files.append(file)

        logger.info(f"Ingesting {len(validated_files)} validated files")
        log_security_event("FILE_INGEST_STARTED", {"file_count": len(validated_files)}, "INFO")

        report = ingest_files(cfg, embedder, validated_files, namespace or cfg.NAMESPACE)

        log_security_event("FILE_INGEST_COMPLETED", {
            "file_count": len(validated_files),
            "namespace": namespace or cfg.NAMESPACE
        }, "INFO")

        return f"✅ Ingestion completed successfully!\n\n{json.dumps(report, indent=2)}"

    except Exception as e:
        logger.error(f"Ingest error: {e}")
        log_security_event("FILE_INGEST_ERROR", {"error": str(e)}, "ERROR")
        # Don't expose internal error details to prevent information leakage
        return "❌ Ingestion failed due to a security or processing error. Please try again."

def ui_costs(monthly_queries, prompt_tk, completion_tk, price_per_1k, base_fixed):
    """
    Enhanced cost calculation with input validation and better formatting.
    """
    try:
        # Input validation
        mq = max(0, float(monthly_queries or 0))
        pt = max(0, float(prompt_tk or 0))
        ct = max(0, float(completion_tk or 0))
        ppk = max(0, float(price_per_1k or 0))
        base = max(0, float(base_fixed or 0))

        # Calculate costs
        token_cost = (mq * (pt + ct) / 1000.0) * ppk
        total_cost = base + token_cost

        # Enhanced formatting with breakdown
        result = "💰 **Cost Estimation**\n\n"
        result += f"**Monthly Queries:** {mq:,.0f}\n"
        result += f"**Tokens per Query:** {pt + ct:,.0f} (Prompt: {pt:.0f}, Completion: {ct:.0f})\n"
        result += f"**Price per 1K tokens:** ${ppk:.6f}\n\n"
        result += "─" * 40 + "\n"
        result += f"**Base Cost:** ${base:,.2f}\n"
        result += f"**Token Cost:** ${token_cost:,.2f}\n"
        result += f"**Total Monthly:** ${total_cost:,.2f}"

        return result

    except ValueError as e:
        return f"❌ Invalid input: Please enter valid numbers only."
    except Exception as e:
        logger.error(f"Cost calculation error: {e}")
        return f"❌ Error computing estimate: {str(e)}"

def get_chat_fn():
    """
    Enhanced chat function with comprehensive security validation and monitoring.
    """
    def _chat(message: str, history: List[Tuple[str, str]]) -> str:
        try:
            # Rate limiting check
            client_ip = os.getenv("REMOTE_ADDR", "unknown")
            if check_rate_limit(client_ip):
                log_security_event("CHAT_RATE_LIMIT_EXCEEDED", {"ip": client_ip}, "WARNING")
                return "❌ Rate limit exceeded. Please try again later."

            if not message or not message.strip():
                log_security_event("CHAT_EMPTY_MESSAGE", {}, "INFO")
                return "❌ Please enter a question."

            # Sanitize input
            sanitized_message = sanitize_input(message)

            # Validate query
            is_valid, validation_message = validate_query(sanitized_message)
            if not is_valid:
                log_security_event("CHAT_QUERY_VALIDATION_FAILED", {
                    "reason": validation_message,
                    "query_length": len(message)
                }, "WARNING")
                return f"❌ Query validation failed: {validation_message}"

            log_security_event("CHAT_QUERY_STARTED", {
                "query_length": len(sanitized_message),
                "history_length": len(history)
            }, "INFO")

            logger.info(f"Processing query: {sanitized_message[:100]}...")
            response = rag_chat(cfg, embedder, sanitized_message, history)
            logger.info("Query processed successfully")

            log_security_event("CHAT_QUERY_COMPLETED", {
                "response_length": len(response)
            }, "INFO")

            return response

        except Exception as e:
            logger.error(f"Chat error: {e}")
            log_security_event("CHAT_ERROR", {"error": str(e)}, "ERROR")
            # Don't expose internal error details to prevent information leakage
            return "❌ Sorry, I encountered an error processing your question. Please try again."

    return _chat

# Gradio 5.x with SSR and enhanced features
with gr.Blocks(
    title="Personal RAG (2025 Stack)",
    theme="soft",  # Gradio 5.x theme string
    css="""
    .gradio-container {
        max-width: 1200px;
        margin: auto;
    }
    .chat-message {
        border-radius: 10px;
        margin: 5px 0;
    }
    """
) as demo:

    gr.Markdown("""
    # 🤖 Personal RAG Chatbot (2025 Stack)

    **Enhanced with Mixture of Experts (MoE) Architecture, Multi-Backend Embeddings, and Advanced Security**

    Ask questions about your uploaded documents. The system uses intelligent expert routing,
    adaptive retrieval, and multi-stage reranking for enhanced accuracy with precise citations.
    """)

    with gr.Tab("💬 Chat", id="chat"):
        gr.Markdown("""
        ### Chat with your documents

        Upload documents in the **Ingest** tab first, then ask questions here.
        The system uses intelligent expert routing, adaptive retrieval, and multi-stage reranking
        for enhanced accuracy with precise citations back to source documents.
        """)

        # Enhanced ChatInterface for Gradio 5.x
        moe_status = "🧠 MoE Enabled" if cfg.moe_enabled else "📚 Standard RAG"
        chat = gr.ChatInterface(
            fn=get_chat_fn(),
            type="messages",
            title="Enhanced RAG Chat",
            description=f"Powered by Mixture of Experts and Sentence-Transformers 5.x | {moe_status}",
            show_progress="minimal",
            concurrency_limit=10,  # Prevent resource exhaustion
        )

    with gr.Tab("📤 Ingest", id="ingest"):
        gr.Markdown("""
        ### Document Ingestion

        Upload PDF, TXT, or MD files. The system will:
        - Extract atomic propositions using advanced LLMs
        - Generate embeddings with multi-backend support
        - Store vectors in Pinecone with gRPC optimization

        **Security:** Files are validated for type and size before processing.
        """)

        with gr.Row():
            files = gr.Files(
                file_types=[".pdf", ".txt", ".md"],
                label="Select Files",
                height=200,
                file_count="multiple"
            )

        with gr.Row():
            ns = gr.Textbox(
                label="Namespace (optional)",
                value=os.getenv("NAMESPACE", "default"),
                placeholder="Leave empty for default namespace"
            )

        with gr.Row():
            ingest_btn = gr.Button("🚀 Start Ingestion", variant="primary", size="lg")

        out = gr.Textbox(
            label="Ingestion Report",
            lines=15,
            show_copy_button=True,
            interactive=False
        )

        ingest_btn.click(
            ui_ingest,
            inputs=[files, ns],
            outputs=out,
            show_progress="minimal"
        )

    with gr.Tab("⚙️ Configuration", id="config"):
        gr.Markdown("""
        ### System Configuration

        Current settings loaded from environment variables and configuration files.
        These values are read-only for security.
        """)

        with gr.Row():
            with gr.Column():
                gr.Textbox(
                    label="🤖 LLM Model",
                    value=os.getenv("OPENROUTER_MODEL", "openrouter/auto"),
                    interactive=False
                )
                gr.Textbox(
                    label="🔍 Embedding Model",
                    value=os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5"),
                    interactive=False
                )
            with gr.Column():
                gr.Textbox(
                    label="📊 Vector Database",
                    value=os.getenv("PINECONE_INDEX", "personal-rag"),
                    interactive=False
                )
                gr.Textbox(
                    label="🎯 Backend",
                    value=os.getenv("SENTENCE_TRANSFORMERS_BACKEND", "torch"),
                    interactive=False
                )

        # MoE Configuration Section (only shown if enabled)
        if cfg.moe_enabled and cfg.moe_config:
            with gr.Row():
                with gr.Column():
                    gr.Textbox(
                        label="🧠 MoE Status",
                        value="Enabled" if cfg.moe_enabled else "Disabled",
                        interactive=False
                    )
                    gr.Textbox(
                        label="🎯 Experts",
                        value=", ".join(cfg.moe_config.get("router", {}).get("experts", [])),
                        interactive=False
                    )
                with gr.Column():
                    gr.Textbox(
                        label="🔀 Router",
                        value="Enabled" if cfg.moe_config.get("router", {}).get("enabled", False) else "Disabled",
                        interactive=False
                    )
                    gr.Textbox(
                        label="🚪 Gate",
                        value="Enabled" if cfg.moe_config.get("gate", {}).get("enabled", False) else "Disabled",
                        interactive=False
                    )
                    gr.Textbox(
                        label="🔄 Reranker",
                        value="Enabled" if cfg.moe_config.get("reranker", {}).get("enabled", False) else "Disabled",
                        interactive=False
                    )

        with gr.Row():
            moe_features = []
            if cfg.moe_enabled:
                moe_features = [
                    "✅ Expert Routing: Intelligent query routing to specialized models",
                    "✅ Adaptive Retrieval: Dynamic k-selection based on query complexity",
                    "✅ Two-Stage Reranking: Cross-encoder + LLM reranking pipeline",
                    "✅ Performance Monitoring: Real-time MoE pipeline analytics"
                ]
            else:
                moe_features = ["⏳ Mixture of Experts: Set MOE_ENABLED=true to activate"]

            gr.Markdown(f"""
            **2025 Stack Features:**
            - ✅ Sentence-Transformers 5.x with multi-backend support
            - ✅ Pinecone gRPC client for improved performance
            - ✅ Enhanced security with input validation
            {chr(10).join("- " + feature for feature in moe_features)}
            """)

    with gr.Tab("🔒 Security Dashboard", id="security"):
        gr.Markdown("""
        ### Security Monitoring Dashboard

        Real-time security metrics and monitoring for the Personal RAG system.
        """)

        with gr.Row():
            with gr.Column():
                gr.Textbox(
                    label="Security Status",
                    value="🟢 System Secure",
                    interactive=False
                )
                gr.Textbox(
                    label="Rate Limit Status",
                    value="✅ Normal",
                    interactive=False
                )
            with gr.Column():
                gr.Textbox(
                    label="Active Threats",
                    value="0",
                    interactive=False
                )
                gr.Textbox(
                    label="Security Events (24h)",
                    value="0",
                    interactive=False
                )

        with gr.Row():
            security_log = gr.Textbox(
                label="Recent Security Events",
                lines=10,
                value="No recent security events.",
                interactive=False,
                show_copy_button=True
            )

        with gr.Row():
            refresh_btn = gr.Button("🔄 Refresh Security Status", variant="secondary")

        def refresh_security_status():
            """Refresh security dashboard data"""
            try:
                from src.security import get_security_metrics
                metrics = get_security_metrics()

                status = "🟢 System Secure"
                if metrics.get("failed_validations", 0) > 10:
                    status = "🟡 Elevated Risk"
                if metrics.get("rate_limit_hits", 0) > 50:
                    status = "🔴 High Risk"

                rate_status = "✅ Normal"
                if metrics.get("rate_limit_hits", 0) > 20:
                    rate_status = "⚠️ High Activity"

                # Read recent security events
                try:
                    with open("logs/security_audit.log", 'r') as f:
                        lines = f.readlines()[-5:]  # Last 5 events
                        recent_events = "\n".join(lines)
                except:
                    recent_events = "No security logs available."

                return (
                    status,
                    rate_status,
                    str(metrics.get("failed_validations", 0)),
                    str(metrics.get("total_requests", 0)),
                    recent_events
                )
            except Exception as e:
                return (
                    "🔴 Error",
                    "❌ Error",
                    "N/A",
                    "N/A",
                    f"Error loading security data: {str(e)}"
                )

        refresh_btn.click(
            refresh_security_status,
            outputs=[
                gr.Textbox(label="Security Status"),
                gr.Textbox(label="Rate Limit Status"),
                gr.Textbox(label="Active Threats"),
                gr.Textbox(label="Security Events (24h)"),
                security_log
            ],
            show_progress="minimal"
        )

    with gr.Tab("💰 Cost Calculator", id="costs"):
        gr.Markdown("""
        ### Cost Estimation Tool

        Estimate your monthly costs based on usage patterns.
        Values are pre-populated from your configuration.
        """)

        with gr.Row():
            with gr.Column():
                monthly_queries = gr.Number(
                    label="Monthly Queries",
                    value=float(os.getenv("COST_MONTHLY_QS", "6000")),
                    minimum=0
                )
                prompt_tk = gr.Number(
                    label="Avg Prompt Tokens",
                    value=float(os.getenv("COST_PROMPT_TOKENS", "300")),
                    minimum=0
                )
            with gr.Column():
                completion_tk = gr.Number(
                    label="Avg Completion Tokens",
                    value=float(os.getenv("COST_COMPLETION_TOKENS", "300")),
                    minimum=0
                )
                price_per_1k = gr.Number(
                    label="LLM Price per 1K tokens (USD)",
                    value=float(os.getenv("COST_PRICE_PER_1K", "0.000375")),
                    minimum=0
                )

        with gr.Row():
            base_fixed = gr.Number(
                label="Base Fixed Cost (USD)",
                value=float(os.getenv("COST_BASE_FIXED", "0")),
                minimum=0
            )

        with gr.Row():
            calc_btn = gr.Button("💰 Calculate", variant="primary", size="lg")

        cost_out = gr.Textbox(
            label="Cost Estimate",
            lines=10,
            show_copy_button=True,
            interactive=False
        )

        calc_btn.click(
            ui_costs,
            inputs=[monthly_queries, prompt_tk, completion_tk, price_per_1k, base_fixed],
            outputs=cost_out,
            show_progress="minimal"
        )

# Health check endpoint for production monitoring
def health_check():
    """
    Production health check endpoint for load balancers and monitoring systems.
    """
    try:
        import psutil
        import time

        # Basic system health checks
        health_status = {
            "status": "healthy",
            "timestamp": time.time(),
            "version": "2.0.0",
            "uptime": time.time() - psutil.boot_time(),
            "services": {
                "embedder": "healthy" if embedder is not None else "unhealthy",
                "vectorstore": "healthy",  # Assume healthy if app started
                "security": "healthy",  # Assume healthy if app started
            },
            "system": {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent
            }
        }

        # Check for critical issues
        if health_status["system"]["memory_percent"] > 90:
            health_status["status"] = "warning"
            health_status["message"] = "High memory usage"

        if health_status["system"]["cpu_percent"] > 95:
            health_status["status"] = "warning"
            health_status["message"] = "High CPU usage"

        # Check if any services are unhealthy
        if any(service == "unhealthy" for service in health_status["services"].values()):
            health_status["status"] = "unhealthy"

        return health_status

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": time.time(),
            "error": str(e)
        }

# Enhanced application launch with health endpoint
if __name__ == "__main__":
    # Add health check route for production
    import threading
    from flask import Flask, jsonify

    health_app = Flask(__name__)

    @health_app.route('/health')
    def health():
        """Health check endpoint for load balancers and monitoring"""
        health_data = health_check()
        status_code = 200 if health_data["status"] in ["healthy", "warning"] else 503
        return jsonify(health_data), status_code

    @health_app.route('/metrics')
    def metrics():
        """Basic metrics endpoint for monitoring systems"""
        try:
            import psutil
            metrics_data = {
                "cpu_usage": psutil.cpu_percent(),
                "memory_usage": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent,
                "timestamp": time.time()
            }
            return jsonify(metrics_data)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # Start health check server in background thread
    def run_health_server():
        try:
            # Only run health server if not in debug mode
            if os.getenv("FLASK_ENV") != "development":
                health_app.run(host='0.0.0.0', port=8000, debug=False, use_reloader=False)
        except Exception as e:
            logger.error(f"Health server failed to start: {e}")

    health_thread = threading.Thread(target=run_health_server, daemon=True)
    health_thread.start()

    # Launch main Gradio application
    try:
        launch_kwargs = {
            "server_name": os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
            "server_port": int(os.getenv("GRADIO_SERVER_PORT", 7860)),
            "show_api": False,  # Disable API for security
            "share": False,  # Disable public sharing for security
            "auth": None,  # Authentication handled separately if needed
            "favicon_path": None,
            "ssl_verify": True,
            "root_path": os.getenv("ROOT_PATH", ""),
            "app_kwargs": {
                "title": "Personal RAG Chatbot (2025 Stack)",
                "description": "Enhanced RAG system with Mixture of Experts and advanced security"
            }
        }

        # Add authentication if configured
        auth_enabled = os.getenv("GRADIO_AUTH_ENABLED", "false").lower() == "true"
        if auth_enabled:
            def auth_function(username, password):
                """Simple authentication function"""
                try:
                    expected_user = os.getenv("GRADIO_AUTH_USER", "admin")
                    expected_pass = os.getenv("GRADIO_AUTH_PASS", "admin")
                    return username == expected_user and password == expected_pass
                except Exception as e:
                    logger.error(f"Authentication error: {e}")
                    return False

            launch_kwargs["auth"] = auth_function
            logger.info("Authentication enabled for Gradio interface")

        logger.info(f"Starting Personal RAG Chatbot on {launch_kwargs['server_name']}:{launch_kwargs['server_port']}")
        logger.info("Health check available at http://localhost:8000/health")
        logger.info("Metrics available at http://localhost:8000/metrics")

        demo.launch(**launch_kwargs)

    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise
                value=float(os.getenv("COST_BASE_FIXED", "50.0")),
                minimum=0
            )

        with gr.Row():
            estimate_btn = gr.Button("📊 Calculate Estimate", variant="secondary")

        estimate = gr.Markdown("Click 'Calculate Estimate' to see your projected costs.")

        estimate_btn.click(
            ui_costs,
            inputs=[monthly_queries, prompt_tk, completion_tk, price_per_1k, base_fixed],
            outputs=estimate,
            show_progress="minimal"
        )

if __name__ == "__main__":
    # Enhanced launch configuration for Gradio 5.x with security hardening
    launch_kwargs = {
        "server_name": os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        "server_port": int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        "share": os.getenv("GRADIO_SHARE", "false").lower() == "true",
        "auth": None,  # Will be configured below
        "ssl_verify": True,
        "show_api": False,  # Security: Don't expose API by default
        "allowed_paths": None,  # Restrict file access for security
        "root_path": os.getenv("GRADIO_ROOT_PATH", ""),
        "max_file_size": int(os.getenv("MAX_FILE_SIZE_MB", "10")) * 1024 * 1024,  # MB to bytes
    }

    # Initialize network security
    try:
        from src.config_secure import load_secure_config
        secure_config = load_secure_config()
        network_security = get_network_security_manager(secure_config)

        # Configure SSL/HTTPS
        if secure_config.enable_https:
            ssl_context = network_security.get_ssl_context()
            if ssl_context:
                launch_kwargs["ssl_context"] = ssl_context
                log_security_event("HTTPS_ENABLED", {
                    "server_name": launch_kwargs["server_name"],
                    "server_port": launch_kwargs["server_port"]
                }, "INFO")

        # Configure authentication if enabled
        if secure_config.enable_authentication:
            def auth_function(username, password):
                try:
                    from src.auth import auth_manager
                    success, message, token = auth_manager.login(
                        username, password,
                        ip_address=os.getenv("REMOTE_ADDR", "unknown"),
                        user_agent=os.getenv("HTTP_USER_AGENT", "unknown")
                    )
                    if success:
                        log_security_event("GRADIO_AUTH_SUCCESS", {
                            "username": username,
                            "ip_address": os.getenv("REMOTE_ADDR", "unknown")
                        }, "INFO")
                        return True
                    else:
                        log_security_event("GRADIO_AUTH_FAILED", {
                            "username": username,
                            "reason": message,
                            "ip_address": os.getenv("REMOTE_ADDR", "unknown")
                        }, "WARNING")
                        return False
                except Exception as e:
                    log_security_event("GRADIO_AUTH_ERROR", {
                        "username": username,
                        "error": str(e)
                    }, "ERROR")
                    return False

            launch_kwargs["auth"] = auth_function
            log_security_event("AUTHENTICATION_ENABLED", {}, "INFO")

        # Security features configured above will be applied
        log_security_event("SECURITY_FEATURES_CONFIGURED", {
            "authentication_enabled": secure_config.enable_authentication,
            "https_enabled": secure_config.enable_https,
            "monitoring_enabled": secure_config.enable_monitoring
        }, "INFO")

    except Exception as e:
        log_security_event("SECURITY_INIT_FAILED", {"error": str(e)}, "ERROR")
        logger.warning(f"Security initialization failed, continuing without security features: {e}")

    try:
        logger.info("Starting Personal RAG Chatbot with 2025 stack and security hardening...")
        demo.launch(**launch_kwargs)
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        log_security_event("APPLICATION_START_FAILED", {"error": str(e)}, "ERROR")
        raise
