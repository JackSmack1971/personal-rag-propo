#!/usr/bin/env python3
"""
Comprehensive QA Test Runner for Personal RAG Chatbot

This script executes all QA tests with detailed logging and evidence collection.
Runs acceptance tests, MoE component tests, performance benchmarks, and security validation.

Author: SPARC QA Analyst
Date: 2025-08-30
"""

import sys
import os
import time
import subprocess
import json
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def log_message(message, level="INFO"):
    """Log a message with timestamp"""
    timestamp = datetime.now().isoformat()
    print(f"[{timestamp}] {level}: {message}")

def run_test_suite(test_file, test_name):
    """Run a test suite and capture results"""
    log_message(f"Starting {test_name} tests...")

    start_time = time.time()
    try:
        # Import the test module and run it
        test_module = test_file.replace('.py', '').replace('/', '.').replace('\\', '.')
        if test_module.startswith('tests.'):
            test_module = test_module[6:]  # Remove 'tests.' prefix

        # Execute the test file as a subprocess to capture output
        test_path = Path(__file__).parent.parent / test_file
        result = subprocess.run(
            [sys.executable, str(test_path)],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        end_time = time.time()
        duration = end_time - start_time

        test_result = {
            "test_name": test_name,
            "test_file": test_file,
            "start_time": datetime.fromtimestamp(start_time).isoformat(),
            "end_time": datetime.fromtimestamp(end_time).isoformat(),
            "duration_seconds": duration,
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "success": result.returncode == 0
        }

        if result.returncode == 0:
            log_message(f"✅ {test_name} tests PASSED ({duration:.2f}s)")
        else:
            log_message(f"❌ {test_name} tests FAILED ({duration:.2f}s)")

        return test_result

    except subprocess.TimeoutExpired:
        end_time = time.time()
        duration = end_time - start_time
        log_message(f"⏰ {test_name} tests TIMED OUT ({duration:.2f}s)")
        return {
            "test_name": test_name,
            "test_file": test_file,
            "start_time": datetime.fromtimestamp(start_time).isoformat(),
            "end_time": datetime.fromtimestamp(end_time).isoformat(),
            "duration_seconds": duration,
            "return_code": -1,
            "stdout": "",
            "stderr": "Test execution timed out",
            "success": False
        }
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        log_message(f"💥 {test_name} tests ERROR: {str(e)} ({duration:.2f}s)")
        return {
            "test_name": test_name,
            "test_file": test_file,
            "start_time": datetime.fromtimestamp(start_time).isoformat(),
            "end_time": datetime.fromtimestamp(end_time).isoformat(),
            "duration_seconds": duration,
            "return_code": -1,
            "stdout": "",
            "stderr": str(e),
            "success": False
        }

def run_moe_validation():
    """Run MoE validation script"""
    log_message("Running MoE validation tests...")

    start_time = time.time()
    try:
        # Run the validation script
        validation_script = Path(__file__).parent / "validate_moe.py"
        result = subprocess.run(
            [sys.executable, str(validation_script)],
            capture_output=True,
            text=True,
            timeout=180  # 3 minute timeout
        )

        end_time = time.time()
        duration = end_time - start_time

        test_result = {
            "test_name": "MoE Validation",
            "test_file": "scripts/validate_moe.py",
            "start_time": datetime.fromtimestamp(start_time).isoformat(),
            "end_time": datetime.fromtimestamp(end_time).isoformat(),
            "duration_seconds": duration,
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "success": result.returncode == 0
        }

        if result.returncode == 0:
            log_message(f"✅ MoE validation PASSED ({duration:.2f}s)")
        else:
            log_message(f"❌ MoE validation FAILED ({duration:.2f}s)")

        return test_result

    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        log_message(f"💥 MoE validation ERROR: {str(e)} ({duration:.2f}s)")
        return {
            "test_name": "MoE Validation",
            "test_file": "scripts/validate_moe.py",
            "start_time": datetime.fromtimestamp(start_time).isoformat(),
            "end_time": datetime.fromtimestamp(end_time).isoformat(),
            "duration_seconds": duration,
            "return_code": -1,
            "stdout": "",
            "stderr": str(e),
            "success": False
        }

def collect_system_info():
    """Collect system information for test context"""
    log_message("Collecting system information...")

    try:
        import platform
        import psutil

        system_info = {
            "platform": platform.platform(),
            "python_version": sys.version,
            "cpu_count": os.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "memory_available_gb": psutil.virtual_memory().available / (1024**3),
            "disk_free_gb": psutil.disk_usage('/').free / (1024**3)
        }

        log_message(f"System: {system_info['platform']}")
        log_message(f"Python: {system_info['python_version'].split()[0]}")
        log_message(f"CPU cores: {system_info['cpu_count']}")
        log_message(f"Memory: {system_info['memory_available_gb']:.1f}GB available")

        return system_info

    except Exception as e:
        log_message(f"Warning: Could not collect system info: {e}")
        return {"error": str(e)}

def check_dependencies():
    """Check if required dependencies are installed"""
    log_message("Checking dependencies...")

    dependencies = [
        "gradio",
        "sentence_transformers",
        "torch",
        "pinecone",
        "pypdf",
        "numpy",
        "pandas",
        "requests",
        "python_dotenv",
        "tqdm"
    ]

    missing_deps = []
    for dep in dependencies:
        try:
            __import__(dep.replace('_', ''))
            log_message(f"✅ {dep}")
        except ImportError:
            log_message(f"❌ {dep} - MISSING")
            missing_deps.append(dep)

    return missing_deps

def main():
    """Run all QA tests"""
    log_message("🚀 Starting Comprehensive QA Test Execution")
    log_message("=" * 60)

    # Initialize results
    test_results = []
    overall_start_time = time.time()

    # Collect system information
    system_info = collect_system_info()
    test_results.append({
        "type": "system_info",
        "data": system_info,
        "timestamp": datetime.now().isoformat()
    })

    # Check dependencies
    missing_deps = check_dependencies()
    test_results.append({
        "type": "dependency_check",
        "missing_dependencies": missing_deps,
        "timestamp": datetime.now().isoformat()
    })

    if missing_deps:
        log_message(f"⚠️  Warning: {len(missing_deps)} dependencies missing: {', '.join(missing_deps)}")
        log_message("Tests may fail due to missing dependencies")

    # Define test suites to run
    test_suites = [
        ("tests/test_acceptance.py", "Acceptance Tests"),
        ("tests/test_moe_components.py", "MoE Component Unit Tests"),
        ("tests/test_moe_integration.py", "MoE Integration Tests"),
        ("tests/test_performance_benchmarking.py", "Performance Benchmarking Tests"),
        ("tests/test_security_validation.py", "Security Validation Tests"),
    ]

    # Run test suites
    for test_file, test_name in test_suites:
        if Path(test_file).exists():
            result = run_test_suite(test_file, test_name)
            test_results.append(result)
            time.sleep(1)  # Brief pause between tests
        else:
            log_message(f"⚠️  Test file not found: {test_file}")

    # Run MoE validation
    moe_result = run_moe_validation()
    test_results.append(moe_result)

    # Calculate summary
    overall_end_time = time.time()
    total_duration = overall_end_time - overall_start_time

    successful_tests = sum(1 for r in test_results if isinstance(r, dict) and r.get("success", False))
    total_tests = len([r for r in test_results if isinstance(r, dict) and "success" in r])

    # Save results to file
    results_file = Path(__file__).parent.parent / "qa_execution_logs.txt"
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write(f"QA Test Execution Results - {datetime.now().isoformat()}\n")
        f.write("=" * 60 + "\n\n")

        f.write("SYSTEM INFORMATION:\n")
        f.write(json.dumps(system_info, indent=2) + "\n\n")

        f.write("DEPENDENCY CHECK:\n")
        if missing_deps:
            f.write(f"Missing dependencies: {', '.join(missing_deps)}\n")
        else:
            f.write("All dependencies available\n")
        f.write("\n")

        f.write("TEST RESULTS SUMMARY:\n")
        f.write(f"Total tests run: {total_tests}\n")
        f.write(f"Successful tests: {successful_tests}\n")
        f.write(f"Failed tests: {total_tests - successful_tests}\n")
        f.write(".2f")
        f.write("\n\n")

        f.write("DETAILED TEST RESULTS:\n")
        for result in test_results:
            if isinstance(result, dict) and "success" in result:
                f.write(f"\n--- {result['test_name']} ---\n")
                f.write(f"File: {result['test_file']}\n")
                f.write(f"Duration: {result['duration_seconds']:.2f}s\n")
                f.write(f"Success: {result['success']}\n")
                if result['stdout']:
                    f.write("STDOUT:\n" + result['stdout'] + "\n")
                if result['stderr']:
                    f.write("STDERR:\n" + result['stderr'] + "\n")

    log_message("=" * 60)
    log_message("📊 QA TEST EXECUTION COMPLETE")
    log_message(f"Total tests: {total_tests}")
    log_message(f"Passed: {successful_tests}")
    log_message(f"Failed: {total_tests - successful_tests}")
    log_message(".2f")
    log_message(f"Results saved to: {results_file}")

    if successful_tests == total_tests:
        log_message("🎉 ALL TESTS PASSED!")
        return 0
    else:
        log_message(f"❌ {total_tests - successful_tests} tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())