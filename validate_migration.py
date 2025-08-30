#!/usr/bin/env python3
"""
2025 Stack Migration Validation Script
Tests all critical dependencies and imports for the enhanced stack.

Author: SPARC Code Implementer
Date: 2025-08-30
"""

import sys
import traceback
from typing import List, Tuple, Dict

def test_import(module_name: str, description: str = "") -> Tuple[bool, str]:
    """Test importing a module and return success status with message."""
    try:
        __import__(module_name)
        return True, f"[OK] {module_name} - {description or 'OK'}"
    except ImportError as e:
        return False, f"[FAIL] {module_name} - Import failed: {e}"
    except Exception as e:
        return False, f"[ERROR] {module_name} - Error: {e}"

def test_version_check(module_name: str, expected_version: str = "") -> Tuple[bool, str]:
    """Test module version if available."""
    try:
        module = __import__(module_name)
        version = getattr(module, '__version__', 'unknown')
        if expected_version and version != 'unknown':
            if version.startswith(expected_version):
                return True, f"[OK] {module_name} v{version} - Version OK"
            else:
                return False, f"[WARN] {module_name} v{version} - Expected {expected_version}"
        return True, f"[OK] {module_name} v{version} - Version check"
    except Exception as e:
        return False, f"[ERROR] {module_name} - Version check failed: {e}"

def main():
    """Run comprehensive validation of 2025 stack dependencies."""

    print("2025 Stack Migration Validation")
    print("=" * 50)

    # Core framework dependencies
    core_frameworks = [
        ("gradio", "UI Framework - SSR enabled"),
        ("sentence_transformers", "ML Embeddings - Multi-backend"),
        ("torch", "ML Framework - Enhanced CUDA"),
        ("pinecone", "Vector Database - gRPC client"),
    ]

    # Data processing dependencies
    data_processing = [
        ("pypdf", "PDF Processing - Enhanced security"),
        ("numpy", "Numerical Computing - Performance"),
        ("pandas", "Data Analysis - String dtype"),
    ]

    # Utility dependencies
    utilities = [
        ("requests", "HTTP Client - Security fixes"),
        ("dotenv", "Environment Management"),
        ("tqdm", "Progress Bars - Enhanced features"),
    ]

    # Test results
    results = []
    all_passed = True

    print("\nCORE FRAMEWORKS:")
    print("-" * 30)
    for module, desc in core_frameworks:
        success, message = test_import(module, desc)
        if not success:
            all_passed = False
        results.append((module, success, message))
        print(message)

    print("\nDATA PROCESSING:")
    print("-" * 30)
    for module, desc in data_processing:
        success, message = test_import(module, desc)
        if not success:
            all_passed = False
        results.append((module, success, message))
        print(message)

    print("\nUTILITIES:")
    print("-" * 30)
    for module, desc in utilities:
        success, message = test_import(module, desc)
        if not success:
            all_passed = False
        results.append((module, success, message))
        print(message)

    # Version checks for critical components
    print("\nVERSION VALIDATION:")
    print("-" * 30)

    version_checks = [
        ("gradio", "5.42"),
        ("sentence_transformers", "5.1"),
        ("torch", "2.8"),
        ("pinecone", "7.0"),
        ("pypdf", "6.0"),
        ("numpy", "2.3"),
        ("pandas", "2.3"),
    ]

    for module, expected in version_checks:
        success, message = test_version_check(module, expected)
        if not success:
            all_passed = False
        results.append((f"{module}_version", success, message))
        print(message)

    # Test local modules
    print("\nLOCAL MODULES:")
    print("-" * 30)

    local_modules = [
        ("src.config", "Configuration management"),
        ("src.embeddings", "Embedding system"),
        ("src.vectorstore", "Vector database operations"),
        ("src.rag", "RAG pipeline"),
        ("src.security", "Security framework"),
    ]

    for module, desc in local_modules:
        success, message = test_import(module, desc)
        if not success:
            all_passed = False
        results.append((module, success, message))
        print(message)

    # Summary
    print("\n" + "=" * 50)
    if all_passed:
        print("MIGRATION SUCCESSFUL!")
        print("All 2025 stack dependencies validated successfully")
        print("System is ready for enhanced features")
        sys.exit(0)
    else:
        print("MIGRATION ISSUES DETECTED!")
        print("Some dependencies failed validation")
        print("\nFailed components:")
        for module, success, message in results:
            if not success and "[FAIL]" in message:
                print(f"  - {module}: {message}")

        print("\nNext steps:")
        print("1. Check pip install logs for errors")
        print("2. Verify Python version compatibility")
        print("3. Check system dependencies (CUDA, etc.)")
        print("4. Review error messages above for specific issues")

        sys.exit(1)

if __name__ == "__main__":
    main()