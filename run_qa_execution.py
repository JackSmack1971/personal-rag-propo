#!/usr/bin/env python3
"""
QA Test Execution Script

This script executes the comprehensive QA testing suite by importing and running
the master QA runner and all test scripts.

Author: SPARC QA Analyst
Date: 2025-08-30
"""

import sys
import os
from pathlib import Path

# Add scripts directory to path
scripts_dir = Path(__file__).parent / "scripts"
sys.path.insert(0, str(scripts_dir))

def main():
    """Execute QA tests by importing the master runner"""
    try:
        print("🚀 Starting QA Test Execution...")
        print("=" * 60)

        # Import and run the master QA runner
        from master_qa_runner import main as run_qa

        exit_code = run_qa()
        return exit_code

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("This may be due to missing dependencies or module issues")
        return 1
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)