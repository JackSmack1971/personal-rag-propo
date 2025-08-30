#!/usr/bin/env python3
"""
Quality Gate Validation Script

Executes quality gate validation on QA artifacts to ensure authenticity and completeness.
This script prevents documentation-only QA bypass by enforcing genuine testing execution.

Usage:
    python scripts/validate_quality_gates.py [--artifacts-dir ARTIFACTS_DIR] [--config CONFIG_FILE]

Author: SPARC Quality Assurance Team
Date: 2025-08-30
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from quality_gates import QualityGateEngine, create_default_config, QualityGateConfig

def load_config(config_path: Optional[str] = None) -> QualityGateConfig:
    """Load quality gate configuration from file or use defaults"""
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)

            return QualityGateConfig(
                execution_required_artifacts=config_data.get('execution_required_artifacts', []),
                performance_required_artifacts=config_data.get('performance_required_artifacts', []),
                security_required_artifacts=config_data.get('security_required_artifacts', []),
                coverage_required_artifacts=config_data.get('coverage_required_artifacts', []),
                adversarial_required_artifacts=config_data.get('adversarial_required_artifacts', []),
                minimum_authenticity_score=config_data.get('minimum_authenticity_score', 0.90),
                minimum_execution_evidence_score=config_data.get('minimum_execution_evidence_score', 0.95),
                minimum_performance_variance=config_data.get('minimum_performance_variance', 0.05),
                minimum_security_coverage=config_data.get('minimum_security_coverage', 0.95),
                minimum_coverage_threshold=config_data.get('minimum_coverage_threshold', 0.70)
            )
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Failed to load config from {config_path}: {e}")
            print("Using default configuration...")

    return create_default_config()

def discover_qa_artifacts(artifacts_dir: str = '.') -> Dict[str, str]:
    """Discover QA artifacts in the specified directory"""
    artifacts_path = Path(artifacts_dir)
    qa_artifacts = {}

    # Define expected artifact patterns
    artifact_patterns = {
        'qa_execution_logs.txt': 'qa_execution_logs.txt',
        'qa_test_results.json': 'qa_test_results.json',
        'qa_system_metrics.json': 'qa_system_metrics.json',
        'qa_performance_results.json': 'qa_performance_results.json',
        'qa_system_specs.json': 'qa_system_specs.json',
        'qa_performance_timeline.csv': 'qa_performance_timeline.csv',
        'qa_security_scan_results.xml': 'qa_security_scan_results.xml',
        'qa_vulnerability_details.json': 'qa_vulnerability_details.json',
        'qa_security_test_logs.txt': 'qa_security_test_logs.txt',
        'qa_test_coverage_report.html': 'qa_test_coverage_report.html',
        'qa_coverage_breakdown.json': 'qa_coverage_breakdown.json',
        'qa_uncovered_code_analysis.md': 'qa_uncovered_code_analysis.md',
        'adversarial_verification_report.md': 'adversarial_verification_report.md',
        'artifact_authenticity_analysis.json': 'artifact_authenticity_analysis.json',
        'verification_evidence_log.txt': 'verification_evidence_log.txt'
    }

    print(f"Discovering QA artifacts in: {artifacts_path.absolute()}")

    for artifact_name, filename in artifact_patterns.items():
        artifact_path = artifacts_path / filename
        if artifact_path.exists():
            qa_artifacts[artifact_name] = str(artifact_path)
            print(f"✓ Found: {artifact_name} -> {artifact_path}")
        else:
            print(f"✗ Missing: {artifact_name} -> {artifact_path}")

    return qa_artifacts

def validate_artifacts_presence(qa_artifacts: Dict[str, str], config: QualityGateConfig) -> bool:
    """Validate that required artifacts are present"""
    print("\n" + "="*60)
    print("ARTIFACT PRESENCE VALIDATION")
    print("="*60)

    all_required = set()
    all_required.update(config.execution_required_artifacts)
    all_required.update(config.performance_required_artifacts)
    all_required.update(config.security_required_artifacts)
    all_required.update(config.coverage_required_artifacts)
    all_required.update(config.adversarial_required_artifacts)

    missing_artifacts = []
    for artifact in all_required:
        if artifact not in qa_artifacts:
            missing_artifacts.append(artifact)

    if missing_artifacts:
        print("❌ MISSING REQUIRED ARTIFACTS:")
        for artifact in missing_artifacts:
            print(f"   - {artifact}")
        print(f"\nTotal missing: {len(missing_artifacts)}/{len(all_required)}")
        return False
    else:
        print("✅ ALL REQUIRED ARTIFACTS PRESENT")
        print(f"Total artifacts: {len(qa_artifacts)}/{len(all_required)}")
        return True

def run_quality_gate_validation(qa_artifacts: Dict[str, str], config: QualityGateConfig) -> bool:
    """Run the complete quality gate validation"""
    print("\n" + "="*60)
    print("QUALITY GATE VALIDATION EXECUTION")
    print("="*60)

    engine = QualityGateEngine(config)

    try:
        result = engine.validate_all_gates(qa_artifacts)

        print("\nVALIDATION RESULTS:")
        print(f"Overall Status: {result.overall_status}")
        print(f"Timestamp: {result.timestamp}")

        print("\nGate Results:")
        for gate_name, gate_result in result.gate_results.items():
            status_icon = "✅" if gate_result.status == "PASS" else "❌" if gate_result.status == "FAIL" else "⚠️"
            print(f"   {status_icon} {gate_name}: {gate_result.score:.3f}")
        print("\nCritical Issues:")
        if result.critical_issues:
            for issue in result.critical_issues:
                print(f"   - {issue}")
        else:
            print("   No critical issues found.")

        print("\nRecommendations:")
        if result.recommendations:
            for rec in result.recommendations:
                print(f"   - {rec}")
        else:
            print("   No recommendations.")

        # Generate detailed report
        report = engine.generate_report(result)
        report_path = Path("qa_quality_gate_report.md")
        with open(report_path, 'w') as f:
            f.write(report)

        print(f"\n📄 Detailed report saved to: {report_path.absolute()}")

        return result.overall_status == "PASS"

    except Exception as e:
        print(f"❌ QUALITY GATE VALIDATION FAILED: {e}")
        return False

def generate_summary_report(qa_artifacts: Dict[str, str], validation_passed: bool) -> None:
    """Generate a summary report of the validation"""
    summary = f"""# QA Quality Gate Validation Summary

**Date:** 2025-08-30
**Validation Status:** {'✅ PASSED' if validation_passed else '❌ FAILED'}

## Artifacts Summary

**Total Artifacts Found:** {len(qa_artifacts)}

### Artifact Inventory
"""

    for artifact_name, path in qa_artifacts.items():
        file_size = os.path.getsize(path) if os.path.exists(path) else 0
        summary += f"- **{artifact_name}**: {Path(path).name} ({file_size} bytes)\n"

    summary += f"""

## Validation Outcome

{'✅ All quality gates passed. QA artifacts are authentic and complete.' if validation_passed else '❌ Quality gate validation failed. QA artifacts require remediation.'}

## Next Steps

{'Production deployment can proceed with confidence.' if validation_passed else 'Address critical issues and re-run validation before deployment.'}

---
*Generated by SPARC Quality Assurance System*
"""

    with open("qa_quality_gate_summary.md", "w") as f:
        f.write(summary)

    print(f"\n📋 Summary report saved to: qa_quality_gate_summary.md")

def main():
    parser = argparse.ArgumentParser(description="Validate QA quality gates")
    parser.add_argument(
        "--artifacts-dir",
        default=".",
        help="Directory containing QA artifacts (default: current directory)"
    )
    parser.add_argument(
        "--config",
        help="Path to quality gate configuration file (optional)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    print("🔍 SPARC QA Quality Gate Validator v1.0")
    print("="*50)
    print(f"Artifacts Directory: {Path(args.artifacts_dir).absolute()}")
    print(f"Configuration: {args.config or 'Default'}")
    print(f"Verbose Mode: {'Enabled' if args.verbose else 'Disabled'}")
    print()

    # Load configuration
    config = load_config(args.config)

    # Discover artifacts
    qa_artifacts = discover_qa_artifacts(args.artifacts_dir)

    if not qa_artifacts:
        print("❌ No QA artifacts found!")
        print("Expected artifacts:")
        all_artifacts = (create_default_config().execution_required_artifacts +
                        create_default_config().performance_required_artifacts +
                        create_default_config().security_required_artifacts +
                        create_default_config().coverage_required_artifacts +
                        create_default_config().adversarial_required_artifacts)
        for artifact in all_artifacts:
            print(f"   - {artifact}")
        return 1

    # Validate artifact presence
    artifacts_valid = validate_artifacts_presence(qa_artifacts, config)

    if not artifacts_valid:
        print("\n❌ ARTIFACT VALIDATION FAILED")
        print("Cannot proceed with quality gate validation.")
        return 1

    # Run quality gate validation
    validation_passed = run_quality_gate_validation(qa_artifacts, config)

    # Generate summary report
    generate_summary_report(qa_artifacts, validation_passed)

    print("\n" + "="*60)
    if validation_passed:
        print("🎉 QUALITY GATE VALIDATION COMPLETED SUCCESSFULLY!")
        print("QA artifacts are authentic and meet all quality standards.")
    else:
        print("⚠️  QUALITY GATE VALIDATION FAILED!")
        print("QA artifacts require remediation before proceeding.")
    print("="*60)

    return 0 if validation_passed else 1

if __name__ == "__main__":
    sys.exit(main())