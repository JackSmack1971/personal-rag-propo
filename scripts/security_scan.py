#!/usr/bin/env python3
"""
Security Vulnerability Scanner for Personal RAG Chatbot

This script performs security validation including:
- Dependency vulnerability scanning
- Configuration security checks
- Input validation testing
- Access control verification

Author: SPARC QA Analyst
Date: 2025-08-30
"""

import sys
import os
import re
import json
import hashlib
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def log_message(message, level="INFO"):
    """Log a message with timestamp"""
    timestamp = datetime.now().isoformat()
    print(f"[{timestamp}] {level}: {message}")

def scan_dependencies():
    """Scan dependencies for known vulnerabilities"""
    log_message("Scanning dependencies for vulnerabilities...")

    vulnerabilities = []

    try:
        import pkg_resources
        installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}

        # Check for known vulnerable versions
        vulnerable_packages = {
            "requests": ["<2.32.5"],  # CVE-2024-47081
            "urllib3": ["<1.26.19", "<2.2.2"],  # Multiple CVEs
            "certifi": ["<2024.7.4"],  # Certificate validation issues
            "cryptography": ["<42.0.8"],  # Multiple CVEs
        }

        for package, vulnerable_versions in vulnerable_packages.items():
            if package in installed_packages:
                current_version = installed_packages[package]
                for vuln_version in vulnerable_versions:
                    if check_version_vulnerable(current_version, vuln_version):
                        vulnerabilities.append({
                            "package": package,
                            "current_version": current_version,
                            "vulnerable_pattern": vuln_version,
                            "severity": "HIGH",
                            "description": f"Package {package} has known vulnerabilities in versions {vuln_version}"
                        })
                        break

        log_message(f"Dependency scan complete. Found {len(vulnerabilities)} vulnerabilities.")

    except Exception as e:
        log_message(f"Error during dependency scan: {e}", "ERROR")

    return vulnerabilities

def check_version_vulnerable(current_version, vulnerable_pattern):
    """Check if current version matches vulnerable pattern"""
    # Simple version comparison - in production use proper version comparison
    if vulnerable_pattern.startswith("<"):
        vuln_ver = vulnerable_pattern[1:]
        return current_version < vuln_ver
    elif vulnerable_pattern.startswith("<="):
        vuln_ver = vulnerable_pattern[2:]
        return current_version <= vuln_ver
    return False

def scan_configuration_files():
    """Scan configuration files for security issues"""
    log_message("Scanning configuration files...")

    issues = []

    # Check .env file
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        with open(env_file, 'r') as f:
            content = f.read()

        # Check for hardcoded secrets
        if "password" in content.lower() and not content.strip().startswith("#"):
            issues.append({
                "file": ".env",
                "issue": "Potential hardcoded password",
                "severity": "HIGH",
                "description": "Found 'password' in .env file - ensure it's properly masked"
            })

        # Check for API keys in plain text
        api_key_patterns = [
            r"sk-[a-zA-Z0-9]{48}",  # OpenAI style
            r"pk_[a-zA-Z0-9_]{100,}",  # Stripe style
            r"[a-zA-Z0-9]{32,}",  # Generic long keys
        ]

        for pattern in api_key_patterns:
            if re.search(pattern, content):
                issues.append({
                    "file": ".env",
                    "issue": "Potential API key exposure",
                    "severity": "CRITICAL",
                    "description": "Found pattern matching API key format in .env file"
                })
                break

    # Check Python files for security issues
    python_files = list(Path(__file__).parent.parent.glob("**/*.py"))
    for py_file in python_files:
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Check for eval usage
            if "eval(" in content:
                issues.append({
                    "file": str(py_file),
                    "issue": "Use of eval() function",
                    "severity": "HIGH",
                    "description": "Found eval() usage which can execute arbitrary code"
                })

            # Check for exec usage
            if "exec(" in content:
                issues.append({
                    "file": str(py_file),
                    "issue": "Use of exec() function",
                    "severity": "HIGH",
                    "description": "Found exec() usage which can execute arbitrary code"
                })

            # Check for SQL injection patterns
            sql_patterns = [r"SELECT.*\+", r"INSERT.*\+", r"UPDATE.*\+", r"DELETE.*\+"]
            for pattern in sql_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    issues.append({
                        "file": str(py_file),
                        "issue": "Potential SQL injection",
                        "severity": "HIGH",
                        "description": f"Found string concatenation in SQL query: {pattern}"
                    })

            # Check for command injection
            if "subprocess" in content and ("shell=True" in content or "Popen" in content):
                issues.append({
                    "file": str(py_file),
                    "issue": "Potential command injection",
                    "severity": "HIGH",
                    "description": "Found subprocess usage that may be vulnerable to command injection"
                })

        except Exception as e:
            log_message(f"Error scanning {py_file}: {e}", "WARNING")

    log_message(f"Configuration scan complete. Found {len(issues)} issues.")
    return issues

def scan_input_validation():
    """Test input validation mechanisms"""
    log_message("Testing input validation...")

    issues = []

    # Test file upload validation
    try:
        from src.config import AppConfig
        config = AppConfig()

        # Check if file type restrictions are in place
        allowed_extensions = getattr(config, 'ALLOWED_FILE_EXTENSIONS', None)
        if not allowed_extensions:
            issues.append({
                "component": "File Upload",
                "issue": "No file type restrictions",
                "severity": "MEDIUM",
                "description": "No file extension restrictions configured for uploads"
            })

        # Check file size limits
        max_file_size = getattr(config, 'MAX_FILE_SIZE_MB', None)
        if not max_file_size or max_file_size > 100:
            issues.append({
                "component": "File Upload",
                "issue": "No file size limits",
                "severity": "MEDIUM",
                "description": "No file size limits configured for uploads"
            })

    except Exception as e:
        log_message(f"Error testing input validation: {e}", "WARNING")

    log_message(f"Input validation test complete. Found {len(issues)} issues.")
    return issues

def scan_network_security():
    """Test network security configurations"""
    log_message("Testing network security...")

    issues = []

    try:
        from src.config import AppConfig
        config = AppConfig()

        # Check HTTPS configuration
        if hasattr(config, 'REQUIRE_HTTPS'):
            if not getattr(config, 'REQUIRE_HTTPS', True):
                issues.append({
                    "component": "Network Security",
                    "issue": "HTTPS not enforced",
                    "severity": "HIGH",
                    "description": "HTTPS is not required for all connections"
                })

        # Check CORS configuration
        if hasattr(config, 'CORS_ORIGINS'):
            cors_origins = getattr(config, 'CORS_ORIGINS', [])
            if "*" in cors_origins:
                issues.append({
                    "component": "Network Security",
                    "issue": "CORS allows all origins",
                    "severity": "MEDIUM",
                    "description": "CORS configuration allows requests from any origin"
                })

        # Check API key validation
        api_key = getattr(config, 'OPENROUTER_API_KEY', '')
        if not api_key or api_key.startswith('your-') or api_key == '':
            issues.append({
                "component": "API Security",
                "issue": "API key not configured",
                "severity": "HIGH",
                "description": "OpenRouter API key is not properly configured"
            })

    except Exception as e:
        log_message(f"Error testing network security: {e}", "WARNING")

    log_message(f"Network security test complete. Found {len(issues)} issues.")
    return issues

def generate_security_report(vulnerabilities, config_issues, validation_issues, network_issues):
    """Generate comprehensive security report"""
    log_message("Generating security report...")

    report = {
        "scan_timestamp": datetime.now().isoformat(),
        "summary": {
            "total_vulnerabilities": len(vulnerabilities),
            "total_config_issues": len(config_issues),
            "total_validation_issues": len(validation_issues),
            "total_network_issues": len(network_issues),
            "overall_risk_level": "LOW"
        },
        "vulnerabilities": vulnerabilities,
        "configuration_issues": config_issues,
        "validation_issues": validation_issues,
        "network_issues": network_issues,
        "recommendations": []
    }

    # Calculate overall risk level
    all_issues = vulnerabilities + config_issues + validation_issues + network_issues
    critical_count = sum(1 for issue in all_issues if issue.get("severity") == "CRITICAL")
    high_count = sum(1 for issue in all_issues if issue.get("severity") == "HIGH")

    if critical_count > 0:
        report["summary"]["overall_risk_level"] = "CRITICAL"
    elif high_count > 2:
        report["summary"]["overall_risk_level"] = "HIGH"
    elif high_count > 0:
        report["summary"]["overall_risk_level"] = "MEDIUM"
    else:
        report["summary"]["overall_risk_level"] = "LOW"

    # Generate recommendations
    if vulnerabilities:
        report["recommendations"].append("Update vulnerable dependencies to latest secure versions")
    if config_issues:
        report["recommendations"].append("Review and fix configuration security issues")
    if validation_issues:
        report["recommendations"].append("Implement proper input validation mechanisms")
    if network_issues:
        report["recommendations"].append("Review and strengthen network security configurations")

    return report

def main():
    """Run comprehensive security scan"""
    log_message("🔒 Starting Security Vulnerability Scan")
    log_message("=" * 60)

    start_time = datetime.now()

    # Run all security checks
    vulnerabilities = scan_dependencies()
    config_issues = scan_configuration_files()
    validation_issues = scan_input_validation()
    network_issues = scan_network_security()

    # Generate report
    report = generate_security_report(
        vulnerabilities, config_issues, validation_issues, network_issues
    )

    end_time = datetime.now()
    scan_duration = (end_time - start_time).total_seconds()

    # Save report
    report_file = Path(__file__).parent.parent / "qa_security_scan_results.xml"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n")
        f.write("<security-scan-report>\n")
        f.write(f"  <timestamp>{report['scan_timestamp']}</timestamp>\n")
        f.write(f"  <duration>{scan_duration:.2f}</duration>\n")
        f.write("  <summary>\n")
        f.write(f"    <total-vulnerabilities>{report['summary']['total_vulnerabilities']}</total-vulnerabilities>\n")
        f.write(f"    <total-config-issues>{report['summary']['total_config_issues']}</total-config-issues>\n")
        f.write(f"    <total-validation-issues>{report['summary']['total_validation_issues']}</total-validation-issues>\n")
        f.write(f"    <total-network-issues>{report['summary']['total_network_issues']}</total-network-issues>\n")
        f.write(f"    <overall-risk-level>{report['summary']['overall_risk_level']}</overall-risk-level>\n")
        f.write("  </summary>\n")

        # Write vulnerabilities
        f.write("  <vulnerabilities>\n")
        for vuln in vulnerabilities:
            f.write("    <vulnerability>\n")
            f.write(f"      <package>{vuln['package']}</package>\n")
            f.write(f"      <current-version>{vuln['current_version']}</current-version>\n")
            f.write(f"      <severity>{vuln['severity']}</severity>\n")
            f.write(f"      <description>{vuln['description']}</description>\n")
            f.write("    </vulnerability>\n")
        f.write("  </vulnerabilities>\n")

        # Write configuration issues
        f.write("  <configuration-issues>\n")
        for issue in config_issues:
            f.write("    <issue>\n")
            f.write(f"      <file>{issue['file']}</file>\n")
            f.write(f"      <severity>{issue['severity']}</severity>\n")
            f.write(f"      <description>{issue['description']}</description>\n")
            f.write("    </issue>\n")
        f.write("  </configuration-issues>\n")

        # Write recommendations
        f.write("  <recommendations>\n")
        for rec in report["recommendations"]:
            f.write(f"    <recommendation>{rec}</recommendation>\n")
        f.write("  </recommendations>\n")

        f.write("</security-scan-report>\n")

    log_message("=" * 60)
    log_message("🔒 SECURITY SCAN COMPLETE")
    log_message(f"Duration: {scan_duration:.2f} seconds")
    log_message(f"Risk Level: {report['summary']['overall_risk_level']}")
    log_message(f"Vulnerabilities: {report['summary']['total_vulnerabilities']}")
    log_message(f"Config Issues: {report['summary']['total_config_issues']}")
    log_message(f"Validation Issues: {report['summary']['total_validation_issues']}")
    log_message(f"Network Issues: {report['summary']['total_network_issues']}")
    log_message(f"Report saved to: {report_file}")

    return 0

if __name__ == "__main__":
    sys.exit(main())