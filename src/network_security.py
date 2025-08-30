"""
Network Security Module for Personal RAG Chatbot
Implements HTTPS, security headers, CORS protection, and network-level security controls.

Author: SPARC Security Architect
Date: 2025-08-30
"""

import os
import ssl
import hashlib
import secrets
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
from urllib.parse import urlparse

from .security import log_security_event

class HTTPSManager:
    """HTTPS configuration and certificate management"""

    def __init__(self):
        self.cert_dir = Path("config/certs")
        self.cert_dir.mkdir(exist_ok=True)

    def create_ssl_context(self, cert_file: Optional[str] = None,
                          key_file: Optional[str] = None,
                          ca_certs: Optional[str] = None) -> ssl.SSLContext:
        """Create secure SSL context"""

        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)

        # Set minimum TLS version
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.maximum_version = ssl.TLSVersion.TLSv1_3

        # Load certificate and key if provided
        if cert_file and key_file:
            context.load_cert_chain(cert_file, key_file)

        # Load CA certificates for client verification
        if ca_certs:
            context.load_verify_locations(ca_certs)
            context.verify_mode = ssl.CERT_REQUIRED

        # Configure cipher suites (prefer secure ones)
        context.set_ciphers(
            'ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20'
        )

        # Disable compression to prevent CRIME attacks
        context.options |= ssl.OP_NO_COMPRESSION

        # Enable session resumption
        context.options |= ssl.OP_NO_TICKET

        log_security_event("SSL_CONTEXT_CREATED", {
            "tls_min_version": str(context.minimum_version),
            "tls_max_version": str(context.maximum_version),
            "cert_loaded": cert_file is not None,
            "client_verify": ca_certs is not None
        }, "INFO")

        return context

    def generate_self_signed_cert(self, domain: str = "localhost",
                                 validity_days: int = 365) -> Tuple[str, str]:
        """Generate self-signed certificate for development"""

        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from datetime import datetime, timedelta

        # Generate private key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )

        # Create certificate
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "Development"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "Local"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Personal RAG"),
            x509.NameAttribute(NameOID.COMMON_NAME, domain),
        ])

        cert = x509.CertificateBuilder().subject_name(
            subject
        ).issuer_name(
            issuer
        ).public_key(
            private_key.public_key()
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.utcnow()
        ).not_valid_after(
            datetime.utcnow() + timedelta(days=validity_days)
        ).add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName(domain),
                x509.DNSName(f"*.{domain}"),
                x509.DNSName("localhost"),
                x509.DNSName("127.0.0.1"),
            ]),
            critical=False,
        ).sign(private_key, hashes.SHA256())

        # Save certificate and key
        cert_file = self.cert_dir / f"{domain}.crt"
        key_file = self.cert_dir / f"{domain}.key"

        with open(cert_file, 'wb') as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))

        with open(key_file, 'wb') as f:
            f.write(private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ))

        log_security_event("SELF_SIGNED_CERT_GENERATED", {
            "domain": domain,
            "validity_days": validity_days,
            "cert_file": str(cert_file),
            "key_file": str(key_file)
        }, "INFO")

        return str(cert_file), str(key_file)

class SecurityHeadersManager:
    """Security headers management and configuration"""

    def __init__(self):
        self._headers_cache: Dict[str, Dict[str, str]] = {}

    def get_security_headers(self, environment: str = "production") -> Dict[str, str]:
        """Get comprehensive security headers"""

        cache_key = f"headers_{environment}"
        if cache_key in self._headers_cache:
            return self._headers_cache[cache_key]

        headers = {}

        # Content Security Policy
        if environment == "production":
            headers["Content-Security-Policy"] = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
                "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
                "font-src 'self' https://fonts.gstatic.com; "
                "img-src 'self' data: https:; "
                "connect-src 'self' https://openrouter.ai https://api.pinecone.io; "
                "frame-ancestors 'none'; "
                "base-uri 'self'; "
                "form-action 'self'"
            )
        else:
            # More permissive for development
            headers["Content-Security-Policy"] = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net; "
                "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
                "connect-src 'self' https://openrouter.ai https://api.pinecone.io ws://localhost:* http://localhost:*"
            )

        # HTTP Strict Transport Security
        if environment == "production":
            headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains; preload"

        # X-Frame-Options
        headers["X-Frame-Options"] = "DENY"

        # X-Content-Type-Options
        headers["X-Content-Type-Options"] = "nosniff"

        # Referrer-Policy
        headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Permissions-Policy
        headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=(), "
            "payment=(), usb=(), magnetometer=()"
        )

        # Cross-Origin-Embedder-Policy
        headers["Cross-Origin-Embedder-Policy"] = "require-corp"

        # Cross-Origin-Opener-Policy
        headers["Cross-Origin-Opener-Policy"] = "same-origin"

        # X-Permitted-Cross-Domain-Policies
        headers["X-Permitted-Cross-Domain-Policies"] = "none"

        # Remove server information
        headers["Server"] = ""

        self._headers_cache[cache_key] = headers

        log_security_event("SECURITY_HEADERS_GENERATED", {
            "environment": environment,
            "header_count": len(headers)
        }, "DEBUG")

        return headers

    def get_cors_headers(self, origin: str, allowed_origins: List[str],
                        request_method: str = "GET") -> Dict[str, str]:
        """Get CORS headers for cross-origin requests"""

        headers = {}

        # Check if origin is allowed
        if origin in allowed_origins:
            headers["Access-Control-Allow-Origin"] = origin
            headers["Access-Control-Allow-Credentials"] = "true"
            headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
            headers["Access-Control-Allow-Headers"] = (
                "Content-Type, Authorization, X-Requested-With, "
                "X-API-Key, X-Request-ID"
            )
            headers["Access-Control-Max-Age"] = "86400"  # 24 hours

            log_security_event("CORS_HEADERS_GRANTED", {
                "origin": origin,
                "method": request_method
            }, "DEBUG")
        else:
            log_security_event("CORS_HEADERS_DENIED", {
                "origin": origin,
                "method": request_method,
                "allowed_origins": allowed_origins
            }, "WARNING")

        return headers

    def validate_origin(self, origin: str, allowed_origins: List[str]) -> bool:
        """Validate request origin against allowed origins"""

        if not origin:
            return False

        try:
            parsed_origin = urlparse(origin)
            if parsed_origin.scheme not in ['http', 'https']:
                return False

            # Check exact match
            if origin in allowed_origins:
                return True

            # Check wildcard patterns
            for allowed in allowed_origins:
                if allowed.startswith('*'):
                    domain = allowed[1:]  # Remove *
                    if parsed_origin.netloc.endswith(domain):
                        return True

            return False

        except Exception as e:
            log_security_event("ORIGIN_VALIDATION_ERROR", {
                "origin": origin,
                "error": str(e)
            }, "WARNING")
            return False

class CORSPolicyManager:
    """CORS policy management and enforcement"""

    def __init__(self, allowed_origins: List[str]):
        self.allowed_origins = allowed_origins
        self.security_headers = SecurityHeadersManager()

    def handle_cors_request(self, request_headers: Dict[str, str]) -> Dict[str, str]:
        """Handle CORS preflight and actual requests"""

        origin = request_headers.get('Origin', '')
        request_method = request_headers.get('Access-Control-Request-Method', 'GET')

        # Validate origin
        if not self.security_headers.validate_origin(origin, self.allowed_origins):
            return {}

        # Return CORS headers
        return self.security_headers.get_cors_headers(
            origin, self.allowed_origins, request_method
        )

    def is_cors_request(self, request_headers: Dict[str, str]) -> bool:
        """Check if request is a CORS request"""
        return 'Origin' in request_headers

class NetworkFirewall:
    """Network-level firewall and request filtering"""

    def __init__(self, trusted_proxies: Optional[List[str]] = None):
        self.trusted_proxies = trusted_proxies or []
        self.blocked_ips: set = set()
        self.suspicious_patterns = [
            r'\.\./',  # Path traversal
            r'<script',  # XSS attempts
            r'union\s+select',  # SQL injection
            r'eval\s*\(',  # Code injection
            r'javascript:',  # JavaScript injection
        ]

    def should_block_request(self, client_ip: str, request_path: str,
                           request_headers: Dict[str, str],
                           user_agent: str = "") -> Tuple[bool, str]:
        """Determine if request should be blocked"""

        # Check blocked IPs
        if client_ip in self.blocked_ips:
            return True, "IP address blocked"

        # Check suspicious patterns in path
        for pattern in self.suspicious_patterns:
            if re.search(pattern, request_path, re.IGNORECASE):
                self.blocked_ips.add(client_ip)
                log_security_event("REQUEST_BLOCKED_SUSPICIOUS_PATTERN", {
                    "client_ip": client_ip,
                    "pattern": pattern,
                    "path": request_path
                }, "WARNING")
                return True, f"Suspicious pattern detected: {pattern}"

        # Check user agent
        suspicious_uas = ['sqlmap', 'nmap', 'nikto', 'dirbuster']
        ua_lower = user_agent.lower()
        for ua in suspicious_uas:
            if ua in ua_lower:
                self.blocked_ips.add(client_ip)
                log_security_event("REQUEST_BLOCKED_SUSPICIOUS_UA", {
                    "client_ip": client_ip,
                    "user_agent": user_agent
                }, "WARNING")
                return True, f"Suspicious user agent: {ua}"

        # Check for common attack headers
        attack_headers = [
            'x-forwarded-for', 'x-real-ip', 'x-client-ip',
            'x-forwarded-host', 'x-host'
        ]

        for header in attack_headers:
            if header in request_headers and request_headers[header]:
                # Verify if proxy is trusted
                header_value = request_headers[header]
                if not self._is_trusted_proxy(client_ip):
                    log_security_event("HEADER_SPOOFING_DETECTED", {
                        "client_ip": client_ip,
                        "header": header,
                        "value": header_value
                    }, "WARNING")
                    return True, f"Header spoofing detected: {header}"

        return False, ""

    def _is_trusted_proxy(self, client_ip: str) -> bool:
        """Check if client IP is from a trusted proxy"""
        return client_ip in self.trusted_proxies

    def get_client_ip(self, request_headers: Dict[str, str],
                     remote_addr: str) -> str:
        """Get real client IP considering trusted proxies"""

        # Check X-Forwarded-For header
        x_forwarded_for = request_headers.get('X-Forwarded-For', '')
        if x_forwarded_for and self._is_trusted_proxy(remote_addr):
            # Take the first IP (original client)
            ips = [ip.strip() for ip in x_forwarded_for.split(',')]
            if ips:
                return ips[0]

        # Check X-Real-IP header
        x_real_ip = request_headers.get('X-Real-IP', '')
        if x_real_ip and self._is_trusted_proxy(remote_addr):
            return x_real_ip.strip()

        return remote_addr

    def block_ip(self, ip_address: str, reason: str = ""):
        """Block an IP address"""
        self.blocked_ips.add(ip_address)
        log_security_event("IP_BLOCKED", {
            "ip_address": ip_address,
            "reason": reason
        }, "WARNING")

    def unblock_ip(self, ip_address: str):
        """Unblock an IP address"""
        self.blocked_ips.discard(ip_address)
        log_security_event("IP_UNBLOCKED", {
            "ip_address": ip_address
        }, "INFO")

class NetworkSecurityManager:
    """Main network security coordinator"""

    def __init__(self, config: Any):
        self.config = config
        self.https_manager = HTTPSManager()
        self.security_headers = SecurityHeadersManager()
        self.cors_manager = CORSPolicyManager(config.cors_allowed_origins or [])
        self.firewall = NetworkFirewall(config.trusted_proxies or [])

    def get_ssl_context(self) -> Optional[ssl.SSLContext]:
        """Get SSL context for HTTPS"""

        if not self.config.enable_https:
            return None

        cert_file = os.getenv("SSL_CERT_FILE")
        key_file = os.getenv("SSL_KEY_FILE")

        if not cert_file or not key_file:
            # Generate self-signed cert for development
            cert_file, key_file = self.https_manager.generate_self_signed_cert()

        return self.https_manager.create_ssl_context(cert_file, key_file)

    def get_response_headers(self, request_headers: Dict[str, str] = None) -> Dict[str, str]:
        """Get all appropriate security headers for response"""

        headers = self.security_headers.get_security_headers(self.config.environment)

        # Add CORS headers if it's a CORS request
        if request_headers and self.cors_manager.is_cors_request(request_headers):
            cors_headers = self.cors_manager.handle_cors_request(request_headers)
            headers.update(cors_headers)

        return headers

    def validate_request(self, client_ip: str, request_path: str,
                        request_headers: Dict[str, str]) -> Tuple[bool, str]:
        """Validate incoming request"""

        # Get real client IP
        real_ip = self.firewall.get_client_ip(request_headers, client_ip)
        user_agent = request_headers.get('User-Agent', '')

        # Check firewall rules
        should_block, reason = self.firewall.should_block_request(
            real_ip, request_path, request_headers, user_agent
        )

        if should_block:
            return False, reason

        return True, ""

    def log_network_event(self, event_type: str, details: Dict[str, Any]):
        """Log network security events"""
        log_security_event(f"NETWORK_{event_type}", details, "INFO")

# Global network security instance
_network_security = None

def get_network_security_manager(config: Any) -> NetworkSecurityManager:
    """Get or create network security manager instance"""
    global _network_security
    if _network_security is None:
        _network_security = NetworkSecurityManager(config)
    return _network_security

def create_secure_app_config() -> Dict[str, Any]:
    """Create secure application configuration for web frameworks"""

    config = {
        "ssl_context": None,
        "security_headers": {},
        "cors_origins": ["http://localhost:7860", "https://localhost:7860"],
        "trusted_proxies": [],
        "session_cookie_secure": True,
        "session_cookie_httponly": True,
        "session_cookie_samesite": "Lax",
        "max_content_length": 10 * 1024 * 1024,  # 10MB
        "max_form_memory_size": 1024 * 1024,  # 1MB
    }

    # Load from environment
    if os.getenv("ENABLE_HTTPS", "true").lower() == "true":
        config["ssl_context"] = "auto"  # Will be handled by NetworkSecurityManager

    cors_origins = os.getenv("CORS_ALLOWED_ORIGINS", "")
    if cors_origins:
        config["cors_origins"] = [origin.strip() for origin in cors_origins.split(",")]

    trusted_proxies = os.getenv("TRUSTED_PROXIES", "")
    if trusted_proxies:
        config["trusted_proxies"] = [proxy.strip() for proxy in trusted_proxies.split(",")]

    return config