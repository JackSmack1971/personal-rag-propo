#!/usr/bin/env python3
"""
Master QA Test Runner for Personal RAG Chatbot

This script orchestrates the complete QA testing process:
1. Runs individual test suites
2. Executes performance benchmarks
3. Performs security scanning
4. Generates comprehensive reports

Author: SPARC QA Analyst
Date: 2025-08-30
"""

import sys
import os
import time
import json
import subprocess
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def log_message(message, level="INFO"):
    """Log a message with timestamp"""
    timestamp = datetime.now().isoformat()
    print(f"[{timestamp}] {level}: {message}")

def run_script(script_name, description):
    """Run a Python script and capture results"""
    log_message(f"Running {description}...")

    script_path = Path(__file__).parent / script_name
    if not script_path.exists():
        log_message(f"Script not found: {script_name}", "ERROR")
        return False

    start_time = time.time()
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )

        end_time = time.time()
        duration = end_time - start_time

        log_message(f"{description} completed in {duration:.2f}s")

        if result.returncode != 0:
            log_message(f"{description} failed with return code {result.returncode}", "WARNING")
            if result.stderr:
                log_message(f"STDERR: {result.stderr[:500]}...", "WARNING")

        return result.returncode == 0

    except subprocess.TimeoutExpired:
        log_message(f"{description} timed out after 10 minutes", "ERROR")
        return False
    except Exception as e:
        log_message(f"Error running {description}: {e}", "ERROR")
        return False

def collect_system_info():
    """Collect system information"""
    try:
        import platform
        import psutil

        return {
            "platform": platform.platform(),
            "python_version": sys.version.split()[0],
            "cpu_count": os.cpu_count(),
            "memory_gb": psutil.virtual_memory().total / (1024**3),
            "timestamp": datetime.now().isoformat()
        }
    except:
        return {"error": "Could not collect system info"}

def check_dependencies():
    """Check if key dependencies are available"""
    dependencies = [
        "gradio",
        "sentence_transformers",
        "torch",
        "pinecone",
        "numpy",
        "pandas"
    ]

    missing = []
    for dep in dependencies:
        try:
            __import__(dep.replace('_', ''))
        except ImportError:
            missing.append(dep)

    return missing

def main():
    """Run complete QA testing suite"""
    log_message("🎯 STARTING COMPREHENSIVE QA TESTING SUITE")
    log_message("=" * 80)

    start_time = datetime.now()

    # Collect initial information
    system_info = collect_system_info()
    missing_deps = check_dependencies()

    log_message("System Information:")
    log_message(f"  Platform: {system_info.get('platform', 'Unknown')}")
    log_message(f"  Python: {system_info.get('python_version', 'Unknown')}")
    log_message(f"  CPU Cores: {system_info.get('cpu_count', 'Unknown')}")
    log_message(f"  Memory: {system_info.get('memory_gb', 'Unknown'):.1f} GB")

    if missing_deps:
        log_message(f"⚠️  Missing dependencies: {', '.join(missing_deps)}")
        log_message("Tests may fail due to missing dependencies")
    else:
        log_message("✅ All key dependencies available")

    # Define test execution sequence
    test_sequence = [
        ("run_individual_tests.py", "Individual Test Execution"),
        ("performance_benchmark.py", "Performance Benchmarking"),
        ("security_scan.py", "Security Vulnerability Scanning"),
        ("validate_moe.py", "MoE Validation"),
    ]

    results = {}
    overall_success = True

    # Execute each test script
    for script_name, description in test_sequence:
        success = run_script(script_name, description)
        results[script_name] = {
            "description": description,
            "success": success,
            "timestamp": datetime.now().isoformat()
        }

        if not success:
            overall_success = False

        # Brief pause between tests
        time.sleep(2)

    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()

    # Generate summary report
    successful_tests = sum(1 for r in results.values() if r["success"])
    total_tests = len(results)

    log_message("\n" + "=" * 80)
    log_message("📊 QA TESTING SUITE COMPLETE")
    log_message(f"Duration: {total_duration:.2f} seconds")
    log_message(f"Tests Passed: {successful_tests}/{total_tests}")

    # Detailed results
    log_message("\nTest Results:")
    for script_name, result in results.items():
        status = "✅ PASSED" if result["success"] else "❌ FAILED"
        log_message(f"  {result['description']}: {status}")

    # Check for generated files
    expected_files = [
        "qa_execution_logs.txt",
        "qa_performance_results.json",
        "qa_security_scan_results.xml",
        "qa_test_coverage_report.html"
    ]

    log_message("\nGenerated Files:")
    for filename in expected_files:
        file_path = Path(__file__).parent.parent / filename
        if file_path.exists():
            size = file_path.stat().st_size
            log_message(f"  ✅ {filename} ({size} bytes)")
        else:
            log_message(f"  ❌ {filename} - NOT GENERATED")

    # Overall assessment
    if overall_success and successful_tests == total_tests:
        log_message("\n🎉 ALL QA TESTS PASSED!")
        log_message("✅ System is ready for production")
        return 0
    else:
        log_message(f"\n⚠️  {total_tests - successful_tests} tests failed")
        log_message("🔧 Review the generated reports for details")
        log_message("📋 Check qa_execution_logs.txt for detailed error information")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        log_message("QA testing interrupted by user", "WARNING")
        sys.exit(130)
    except Exception as e:
        log_message(f"Unexpected error in QA testing: {e}", "ERROR")
        sys.exit(1)