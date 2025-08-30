#!/usr/bin/env python3
"""
Individual Test Runner for Personal RAG Chatbot

This script runs individual test files and captures their output for QA reporting.

Author: SPARC QA Analyst
Date: 2025-08-30
"""

import sys
import os
import time
import importlib.util
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def log_message(message, level="INFO"):
    """Log a message with timestamp"""
    timestamp = datetime.now().isoformat()
    print(f"[{timestamp}] {level}: {message}")

def run_test_file(test_file_path):
    """Run a single test file and capture results"""
    log_message(f"Running test file: {test_file_path}")

    start_time = time.time()

    try:
        # Load the test module
        spec = importlib.util.spec_from_file_location("test_module", test_file_path)
        test_module = importlib.util.module_from_spec(spec)

        # Capture stdout and stderr
        import io
        from contextlib import redirect_stdout, redirect_stderr

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            spec.loader.exec_module(test_module)

        stdout_output = stdout_capture.getvalue()
        stderr_output = stderr_capture.getvalue()

        end_time = time.time()
        duration = end_time - start_time

        # Determine success based on output
        success = "FAILED" not in stderr_output and "Traceback" not in stderr_output

        result = {
            "test_file": str(test_file_path),
            "success": success,
            "duration": duration,
            "stdout": stdout_output,
            "stderr": stderr_output,
            "timestamp": datetime.now().isoformat()
        }

        if success:
            log_message(f"✅ Test passed: {Path(test_file_path).name} ({duration:.2f}s)")
        else:
            log_message(f"❌ Test failed: {Path(test_file_path).name} ({duration:.2f}s)")

        return result

    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        log_message(f"💥 Test error in {Path(test_file_path).name}: {str(e)} ({duration:.2f}s)")

        return {
            "test_file": str(test_file_path),
            "success": False,
            "duration": duration,
            "stdout": "",
            "stderr": str(e),
            "timestamp": datetime.now().isoformat()
        }

def main():
    """Run individual test files"""
    log_message("🎯 Starting Individual Test Execution")
    log_message("=" * 60)

    # Define test files to run
    test_files = [
        "tests/test_acceptance.py",
        "tests/test_moe_components.py",
        "tests/test_moe_integration.py",
        "tests/test_performance_benchmarking.py",
        "tests/test_security_validation.py"
    ]

    results = []
    total_start_time = time.time()

    for test_file in test_files:
        test_path = Path(__file__).parent.parent / test_file
        if test_path.exists():
            result = run_test_file(test_path)
            results.append(result)
            time.sleep(0.5)  # Brief pause between tests
        else:
            log_message(f"⚠️  Test file not found: {test_file}")
            results.append({
                "test_file": test_file,
                "success": False,
                "duration": 0,
                "stdout": "",
                "stderr": "File not found",
                "timestamp": datetime.now().isoformat()
            })

    total_end_time = time.time()
    total_duration = total_end_time - total_start_time

    # Calculate summary
    successful_tests = sum(1 for r in results if r["success"])
    total_tests = len(results)

    # Save results
    results_file = Path(__file__).parent.parent / "qa_test_coverage_report.html"
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("<!DOCTYPE html>\n<html>\n<head>\n")
        f.write("<title>QA Test Coverage Report</title>\n")
        f.write("<style>\n")
        f.write("body { font-family: Arial, sans-serif; margin: 20px; }\n")
        f.write(".success { color: green; }\n")
        f.write(".failure { color: red; }\n")
        f.write(".summary { background: #f0f0f0; padding: 10px; margin: 10px 0; }\n")
        f.write("pre { background: #f8f8f8; padding: 10px; border: 1px solid #ddd; }\n")
        f.write("</style>\n</head>\n<body>\n")
        f.write("<h1>QA Test Coverage Report</h1>\n")
        f.write(f"<p>Generated: {datetime.now().isoformat()}</p>\n")
        f.write(f"<p>Total Duration: {total_duration:.2f} seconds</p>\n")
        f.write("<div class='summary'>\n")
        f.write(f"<h2>Summary: {successful_tests}/{total_tests} tests passed</h2>\n")
        f.write("</div>\n")

        for result in results:
            status_class = "success" if result["success"] else "failure"
            f.write(f"<h3 class='{status_class}'>{Path(result['test_file']).name}</h3>\n")
            f.write(f"<p>Duration: {result['duration']:.2f}s</p>\n")
            if result["stdout"]:
                f.write("<h4>Output:</h4>\n")
                f.write(f"<pre>{result['stdout']}</pre>\n")
            if result["stderr"]:
                f.write("<h4>Errors:</h4>\n")
                f.write(f"<pre class='failure'>{result['stderr']}</pre>\n")

        f.write("</body>\n</html>\n")

    log_message("=" * 60)
    log_message("📊 INDIVIDUAL TEST EXECUTION COMPLETE")
    log_message(f"Total tests: {total_tests}")
    log_message(f"Passed: {successful_tests}")
    log_message(f"Failed: {total_tests - successful_tests}")
    log_message(".2f")
    log_message(f"Report saved to: {results_file}")

    return 0 if successful_tests == total_tests else 1

if __name__ == "__main__":
    sys.exit(main())