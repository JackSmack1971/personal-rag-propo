#!/usr/bin/env python3
"""
QA Quality Gates Implementation v1.0

Critical system to prevent documentation-only QA bypass and ensure genuine testing execution.
Validates authenticity of QA artifacts and enforces mandatory quality standards.

Author: SPARC Quality Assurance Team
Date: 2025-08-30
"""

import json
import os
import re
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QualityGateConfig:
    """Configuration for quality gate validation"""
    execution_required_artifacts: List[str]
    performance_required_artifacts: List[str]
    security_required_artifacts: List[str]
    coverage_required_artifacts: List[str]
    adversarial_required_artifacts: List[str]

    minimum_authenticity_score: float = 0.90
    minimum_execution_evidence_score: float = 0.95
    minimum_performance_variance: float = 0.05
    minimum_security_coverage: float = 0.95
    minimum_coverage_threshold: float = 0.70

@dataclass
class GateResult:
    """Result of a quality gate validation"""
    gate_name: str
    status: str  # 'PASS', 'FAIL', 'WARNING'
    score: float
    evidence: Dict[str, Any]
    issues: List[str]
    recommendations: List[str]

@dataclass
class QualityGateResult:
    """Overall quality gate validation result"""
    overall_status: str
    gate_results: Dict[str, GateResult]
    timestamp: str
    recommendations: List[str]
    critical_issues: List[str]

class BaseValidator:
    """Base class for quality gate validators"""

    def __init__(self, config: QualityGateConfig):
        self.config = config

    def validate(self, artifacts: Dict[str, str]) -> GateResult:
        """Validate artifacts for this gate"""
        raise NotImplementedError

    def _check_artifact_exists(self, artifact_path: str, artifacts: Dict[str, str]) -> bool:
        """Check if required artifact exists"""
        return artifact_path in artifacts and os.path.exists(artifacts[artifact_path])

    def _load_json_artifact(self, artifact_path: str, artifacts: Dict[str, str]) -> Optional[Dict]:
        """Load and parse JSON artifact"""
        if not self._check_artifact_exists(artifact_path, artifacts):
            return None

        try:
            with open(artifacts[artifact_path], 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load JSON artifact {artifact_path}: {e}")
            return None

    def _load_text_artifact(self, artifact_path: str, artifacts: Dict[str, str]) -> Optional[str]:
        """Load text artifact"""
        if not self._check_artifact_exists(artifact_path, artifacts):
            return None

        try:
            with open(artifacts[artifact_path], 'r') as f:
                return f.read()
        except IOError as e:
            logger.error(f"Failed to load text artifact {artifact_path}: {e}")
            return None

    def _calculate_authenticity_score(self, content: str) -> float:
        """Calculate authenticity score based on content patterns"""
        score = 0.0
        total_checks = 0

        # Check for realistic timestamps
        timestamp_pattern = r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}'
        if re.search(timestamp_pattern, content):
            score += 0.3
        total_checks += 0.3

        # Check for technical error messages
        error_patterns = [
            r'ImportError|ModuleNotFoundError|AttributeError',
            r'Exception|Error|Failed',
            r'line \d+|File ".*?\.py"',
            r'Traceback \(most recent call last\)'
        ]
        error_score = sum(0.1 for pattern in error_patterns if re.search(pattern, content))
        score += min(error_score, 0.3)
        total_checks += 0.3

        # Check for realistic file paths
        path_pattern = r'[a-zA-Z]:[/\\]|/home/|/usr/|\.py|\.json|\.txt'
        if re.search(path_pattern, content):
            score += 0.2
        total_checks += 0.2

        # Check for system metrics
        metric_patterns = [
            r'\d+\.\d+ seconds|\d+ MB|\d+ CPU|\d+ queries/sec',
            r'PID \d+|Process|Thread',
            r'Memory|CPU|Disk|Network'
        ]
        metric_score = sum(0.05 for pattern in metric_patterns if re.search(pattern, content))
        score += min(metric_score, 0.2)
        total_checks += 0.2

        return min(score / total_checks, 1.0) if total_checks > 0 else 0.0

class ExecutionValidator(BaseValidator):
    """Validates test execution authenticity"""

    def validate(self, artifacts: Dict[str, str]) -> GateResult:
        issues = []
        recommendations = []
        evidence = {}

        # Check required artifacts
        required = ['qa_execution_logs.txt', 'qa_test_results.json', 'qa_system_metrics.json']
        missing_artifacts = [art for art in required if not self._check_artifact_exists(art, artifacts)]

        if missing_artifacts:
            issues.append(f"Missing required artifacts: {missing_artifacts}")
            return GateResult(
                gate_name='execution',
                status='FAIL',
                score=0.0,
                evidence={'missing_artifacts': missing_artifacts},
                issues=issues,
                recommendations=['Generate all required execution artifacts']
            )

        # Load and analyze execution logs
        logs_content = self._load_text_artifact('qa_execution_logs.txt', artifacts)
        if logs_content:
            authenticity_score = self._calculate_authenticity_score(logs_content)
            evidence['log_authenticity'] = authenticity_score

            # Check for command execution evidence
            command_patterns = [
                r'python.*\.py|pytest|pip install|npm|docker',
                r'Collecting|Installing|Building|Running',
                r'Test session starts|tests ran|passed|failed'
            ]
            command_evidence = sum(1 for pattern in command_patterns if re.search(pattern, logs_content))
            evidence['command_execution_score'] = min(command_evidence / len(command_patterns), 1.0)

            # Check log structure and completeness
            log_lines = len(logs_content.split('\n'))
            evidence['log_completeness'] = min(log_lines / 100, 1.0)  # Expect at least 100 lines

            if authenticity_score < self.config.minimum_authenticity_score:
                issues.append(".2f")
                recommendations.append("Re-execute tests with proper logging enabled")

            if evidence['command_execution_score'] < 0.7:
                issues.append("Insufficient command execution evidence in logs")
                recommendations.append("Ensure all test commands are logged with output")

        # Load test results
        test_results = self._load_json_artifact('qa_test_results.json', artifacts)
        if test_results:
            evidence['test_results_structure'] = self._validate_test_results_structure(test_results)
        else:
            issues.append("Invalid or missing test results JSON")

        # Calculate overall score
        scores = [evidence.get('log_authenticity', 0),
                 evidence.get('command_execution_score', 0),
                 evidence.get('log_completeness', 0),
                 evidence.get('test_results_structure', 0)]

        overall_score = sum(scores) / len(scores) if scores else 0.0

        status = 'PASS' if overall_score >= self.config.minimum_execution_evidence_score else 'FAIL'

        return GateResult(
            gate_name='execution',
            status=status,
            score=overall_score,
            evidence=evidence,
            issues=issues,
            recommendations=recommendations
        )

    def _validate_test_results_structure(self, results: Dict) -> float:
        """Validate test results JSON structure"""
        required_fields = ['timestamp', 'test_suites', 'summary']
        score = 0.0

        for field in required_fields:
            if field in results:
                score += 0.25

        # Check for realistic test data
        if 'test_suites' in results and isinstance(results['test_suites'], list):
            if len(results['test_suites']) > 0:
                score += 0.25

        # Check timestamp validity
        if 'timestamp' in results:
            try:
                datetime.fromisoformat(results['timestamp'].replace('Z', '+00:00'))
                score += 0.25
            except:
                pass

        return min(score, 1.0)

class PerformanceValidator(BaseValidator):
    """Validates performance benchmark authenticity"""

    def validate(self, artifacts: Dict[str, str]) -> GateResult:
        issues = []
        recommendations = []
        evidence = {}

        # Check required artifacts
        required = ['qa_performance_results.json', 'qa_system_specs.json', 'qa_performance_timeline.csv']
        missing_artifacts = [art for art in required if not self._check_artifact_exists(art, artifacts)]

        if missing_artifacts:
            issues.append(f"Missing required artifacts: {missing_artifacts}")
            return GateResult(
                gate_name='performance',
                status='FAIL',
                score=0.0,
                evidence={'missing_artifacts': missing_artifacts},
                issues=issues,
                recommendations=['Execute performance benchmarks and generate artifacts']
            )

        # Load performance results
        perf_results = self._load_json_artifact('qa_performance_results.json', artifacts)
        if perf_results:
            evidence.update(self._analyze_performance_results(perf_results))
        else:
            issues.append("Invalid or missing performance results JSON")

        # Load system specs
        system_specs = self._load_json_artifact('qa_system_specs.json', artifacts)
        if system_specs:
            evidence['system_specs_valid'] = self._validate_system_specs(system_specs)
        else:
            issues.append("Invalid or missing system specifications")

        # Load timeline data
        timeline_content = self._load_text_artifact('qa_performance_timeline.csv', artifacts)
        if timeline_content:
            evidence['timeline_valid'] = self._validate_timeline_data(timeline_content)
        else:
            issues.append("Invalid or missing performance timeline data")

        # Calculate overall score
        scores = [evidence.get('benchmark_authenticity', 0),
                 evidence.get('variance_analysis', 0),
                 evidence.get('system_specs_valid', 0),
                 evidence.get('timeline_valid', 0)]

        overall_score = sum(scores) / len(scores) if scores else 0.0

        status = 'PASS' if overall_score >= self.config.minimum_authenticity_score else 'FAIL'

        return GateResult(
            gate_name='performance',
            status=status,
            score=overall_score,
            evidence=evidence,
            issues=issues,
            recommendations=recommendations
        )

    def _analyze_performance_results(self, results: Dict) -> Dict[str, Any]:
        """Analyze performance results for authenticity"""
        evidence = {}

        # Check for realistic variance in results
        if 'benchmarks' in results:
            variances = []
            for benchmark in results['benchmarks']:
                if 'runs' in benchmark and len(benchmark['runs']) > 1:
                    run_times = [run.get('duration', 0) for run in benchmark['runs']]
                    if run_times:
                        variance = self._calculate_variance(run_times)
                        variances.append(variance)

            if variances:
                avg_variance = sum(variances) / len(variances)
                evidence['variance_analysis'] = min(avg_variance / self.config.minimum_performance_variance, 1.0)
                evidence['benchmark_authenticity'] = 1.0 if avg_variance > 0.01 else 0.5  # Some variance expected
            else:
                evidence['variance_analysis'] = 0.0
                evidence['benchmark_authenticity'] = 0.3  # Single runs suspicious

        return evidence

    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate coefficient of variation"""
        if not values or len(values) < 2:
            return 0.0

        mean = sum(values) / len(values)
        if mean == 0:
            return 0.0

        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return (variance ** 0.5) / mean  # Coefficient of variation

    def _validate_system_specs(self, specs: Dict) -> float:
        """Validate system specifications"""
        required_specs = ['cpu', 'memory', 'os', 'python_version']
        score = 0.0

        for spec in required_specs:
            if spec in specs and specs[spec]:
                score += 0.25

        return min(score, 1.0)

    def _validate_timeline_data(self, content: str) -> float:
        """Validate performance timeline CSV data"""
        lines = content.strip().split('\n')
        if len(lines) < 2:  # Header + at least one data row
            return 0.0

        # Check for timestamp column
        header = lines[0].lower()
        if 'timestamp' not in header and 'time' not in header:
            return 0.3

        # Check data rows for valid format
        data_rows = lines[1:]
        valid_rows = 0
        for row in data_rows:
            if ',' in row and len(row.split(',')) >= 2:
                valid_rows += 1

        return min(valid_rows / len(data_rows), 1.0) if data_rows else 0.0

class SecurityValidator(BaseValidator):
    """Validates security testing authenticity"""

    def validate(self, artifacts: Dict[str, str]) -> GateResult:
        issues = []
        recommendations = []
        evidence = {}

        # Check required artifacts
        required = ['qa_security_scan_results.xml', 'qa_vulnerability_details.json', 'qa_security_test_logs.txt']
        missing_artifacts = [art for art in required if not self._check_artifact_exists(art, artifacts)]

        if missing_artifacts:
            issues.append(f"Missing required artifacts: {missing_artifacts}")
            return GateResult(
                gate_name='security',
                status='FAIL',
                score=0.0,
                evidence={'missing_artifacts': missing_artifacts},
                issues=issues,
                recommendations=['Execute security scanning and generate artifacts']
            )

        # Load security scan results
        scan_content = self._load_text_artifact('qa_security_scan_results.xml', artifacts)
        if scan_content:
            evidence['scan_authenticity'] = self._calculate_authenticity_score(scan_content)
            evidence['vulnerability_coverage'] = self._analyze_security_scan(scan_content)
        else:
            issues.append("Invalid or missing security scan results")

        # Load vulnerability details
        vuln_details = self._load_json_artifact('qa_vulnerability_details.json', artifacts)
        if vuln_details:
            evidence['vulnerability_details_valid'] = self._validate_vulnerability_details(vuln_details)
        else:
            issues.append("Invalid or missing vulnerability details")

        # Load security test logs
        test_logs = self._load_text_artifact('qa_security_test_logs.txt', artifacts)
        if test_logs:
            evidence['security_test_authenticity'] = self._calculate_authenticity_score(test_logs)
            evidence['test_execution_evidence'] = self._analyze_security_test_logs(test_logs)
        else:
            issues.append("Invalid or missing security test logs")

        # Calculate overall score
        scores = [evidence.get('scan_authenticity', 0),
                 evidence.get('vulnerability_coverage', 0),
                 evidence.get('vulnerability_details_valid', 0),
                 evidence.get('security_test_authenticity', 0),
                 evidence.get('test_execution_evidence', 0)]

        overall_score = sum(scores) / len(scores) if scores else 0.0

        status = 'PASS' if overall_score >= self.config.minimum_authenticity_score else 'FAIL'

        return GateResult(
            gate_name='security',
            status=status,
            score=overall_score,
            evidence=evidence,
            issues=issues,
            recommendations=recommendations
        )

    def _analyze_security_scan(self, content: str) -> float:
        """Analyze security scan results for coverage"""
        # Look for common vulnerability patterns
        vuln_patterns = [
            r'CVE-\d{4}-\d+',
            r'severity.*(?:critical|high|medium|low)',
            r'vulnerability|exploit|weakness',
            r'CWE-\d+',
            r'file.*?\.py.*?:.*?\d+'
        ]

        coverage_score = sum(0.2 for pattern in vuln_patterns if re.search(pattern, content, re.IGNORECASE))
        return min(coverage_score, 1.0)

    def _validate_vulnerability_details(self, details: Dict) -> float:
        """Validate vulnerability details structure"""
        if not isinstance(details, dict) or 'vulnerabilities' not in details:
            return 0.0

        vulnerabilities = details['vulnerabilities']
        if not isinstance(vulnerabilities, list):
            return 0.3

        if len(vulnerabilities) == 0:
            return 1.0  # No vulnerabilities is valid

        # Check structure of first vulnerability
        if vulnerabilities:
            vuln = vulnerabilities[0]
            required_fields = ['id', 'severity', 'description']
            field_score = sum(0.2 for field in required_fields if field in vuln)
            return min(field_score, 1.0)

        return 0.5

    def _analyze_security_test_logs(self, content: str) -> float:
        """Analyze security test logs for execution evidence"""
        # Look for security testing patterns
        test_patterns = [
            r'security.*scan|vulnerability.*scan',
            r'nmap|nessus|owasp|bandit',
            r'security.*test|penetration.*test',
            r'exploit|attack|injection',
            r'sql.*injection|xss|csrf'
        ]

        execution_score = sum(0.2 for pattern in test_patterns if re.search(pattern, content, re.IGNORECASE))
        return min(execution_score, 1.0)

class CoverageValidator(BaseValidator):
    """Validates test coverage authenticity"""

    def validate(self, artifacts: Dict[str, str]) -> GateResult:
        issues = []
        recommendations = []
        evidence = {}

        # Check required artifacts
        required = ['qa_test_coverage_report.html', 'qa_coverage_breakdown.json', 'qa_uncovered_code_analysis.md']
        missing_artifacts = [art for art in required if not self._check_artifact_exists(art, artifacts)]

        if missing_artifacts:
            issues.append(f"Missing required artifacts: {missing_artifacts}")
            return GateResult(
                gate_name='coverage',
                status='FAIL',
                score=0.0,
                evidence={'missing_artifacts': missing_artifacts},
                issues=issues,
                recommendations=['Execute test coverage analysis and generate artifacts']
            )

        # Load coverage report
        coverage_html = self._load_text_artifact('qa_test_coverage_report.html', artifacts)
        if coverage_html:
            evidence['coverage_authenticity'] = self._calculate_authenticity_score(coverage_html)
            evidence['coverage_percentage'] = self._extract_coverage_percentage(coverage_html)
            evidence['file_coverage_valid'] = self._validate_file_coverage(coverage_html)
        else:
            issues.append("Invalid or missing coverage report")

        # Load coverage breakdown
        coverage_breakdown = self._load_json_artifact('qa_coverage_breakdown.json', artifacts)
        if coverage_breakdown:
            evidence['breakdown_structure_valid'] = self._validate_coverage_breakdown(coverage_breakdown)
        else:
            issues.append("Invalid or missing coverage breakdown")

        # Load uncovered analysis
        uncovered_analysis = self._load_text_artifact('qa_uncovered_code_analysis.md', artifacts)
        if uncovered_analysis:
            evidence['uncovered_analysis_authenticity'] = self._calculate_authenticity_score(uncovered_analysis)
            evidence['uncovered_analysis_completeness'] = self._validate_uncovered_analysis(uncovered_analysis)
        else:
            issues.append("Invalid or missing uncovered code analysis")

        # Calculate overall score
        scores = [evidence.get('coverage_authenticity', 0),
                 evidence.get('coverage_percentage', 0),
                 evidence.get('file_coverage_valid', 0),
                 evidence.get('breakdown_structure_valid', 0),
                 evidence.get('uncovered_analysis_authenticity', 0),
                 evidence.get('uncovered_analysis_completeness', 0)]

        overall_score = sum(scores) / len(scores) if scores else 0.0

        # Check minimum coverage threshold
        coverage_pct = evidence.get('coverage_percentage', 0)
        if coverage_pct < self.config.minimum_coverage_threshold:
            issues.append(".1f")
            recommendations.append("Increase test coverage to meet minimum threshold")

        status = 'PASS' if overall_score >= self.config.minimum_authenticity_score else 'FAIL'

        return GateResult(
            gate_name='coverage',
            status=status,
            score=overall_score,
            evidence=evidence,
            issues=issues,
            recommendations=recommendations
        )

    def _extract_coverage_percentage(self, html_content: str) -> float:
        """Extract coverage percentage from HTML report"""
        # Look for coverage percentage patterns
        patterns = [
            r'coverage.*?:.*?(\d+(?:\.\d+)?)%',
            r'total.*?(\d+(?:\.\d+)?)%',
            r'(\d+(?:\.\d+)?)%.*?(?:covered|coverage)'
        ]

        for pattern in patterns:
            match = re.search(pattern, html_content, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1)) / 100.0
                except ValueError:
                    continue

        return 0.0

    def _validate_file_coverage(self, html_content: str) -> float:
        """Validate file coverage data in HTML"""
        # Look for file listing patterns
        file_patterns = [
            r'<tr>.*?<td>.*?\.py</td>.*?</tr>',
            r'class="file".*?\.py',
            r'href=".*?\.py.*?"',
            r'<a.*?\.py.*?>.*?</a>'
        ]

        file_evidence = sum(1 for pattern in file_patterns if re.search(pattern, html_content))
        return min(file_evidence / len(file_patterns), 1.0)

    def _validate_coverage_breakdown(self, breakdown: Dict) -> float:
        """Validate coverage breakdown structure"""
        if not isinstance(breakdown, dict):
            return 0.0

        required_keys = ['files', 'total_coverage', 'timestamp']
        key_score = sum(0.2 for key in required_keys if key in breakdown)

        # Check files structure
        if 'files' in breakdown and isinstance(breakdown['files'], list):
            key_score += 0.2
            if breakdown['files'] and isinstance(breakdown['files'][0], dict):
                key_score += 0.2

        return min(key_score, 1.0)

    def _validate_uncovered_analysis(self, content: str) -> float:
        """Validate uncovered code analysis completeness"""
        # Look for analysis patterns
        analysis_patterns = [
            r'uncovered|not.*covered|missing.*test',
            r'line.*?\d+|function|class',
            r'reason|justification|explanation',
            r'plan|strategy|approach'
        ]

        analysis_score = sum(0.25 for pattern in analysis_patterns if re.search(pattern, content, re.IGNORECASE))
        return min(analysis_score, 1.0)

class AdversarialValidator(BaseValidator):
    """Performs adversarial verification of all artifacts"""

    def validate(self, artifacts: Dict[str, str]) -> GateResult:
        issues = []
        recommendations = []
        evidence = {}

        # Check required artifacts
        required = ['adversarial_verification_report.md', 'artifact_authenticity_analysis.json', 'verification_evidence_log.txt']
        missing_artifacts = [art for art in required if not self._check_artifact_exists(art, artifacts)]

        if missing_artifacts:
            issues.append(f"Missing required artifacts: {missing_artifacts}")
            return GateResult(
                gate_name='adversarial',
                status='FAIL',
                score=0.0,
                evidence={'missing_artifacts': missing_artifacts},
                issues=issues,
                recommendations=['Execute adversarial verification and generate artifacts']
            )

        # Load verification report
        verification_report = self._load_text_artifact('adversarial_verification_report.md', artifacts)
        if verification_report:
            evidence['verification_completeness'] = self._analyze_verification_report(verification_report)
            evidence['verification_authenticity'] = self._calculate_authenticity_score(verification_report)
        else:
            issues.append("Invalid or missing adversarial verification report")

        # Load authenticity analysis
        authenticity_analysis = self._load_json_artifact('artifact_authenticity_analysis.json', artifacts)
        if authenticity_analysis:
            evidence['authenticity_scoring_valid'] = self._validate_authenticity_analysis(authenticity_analysis)
            evidence['cross_artifact_consistency'] = self._analyze_cross_artifact_consistency(authenticity_analysis)
        else:
            issues.append("Invalid or missing authenticity analysis")

        # Load verification evidence log
        evidence_log = self._load_text_artifact('verification_evidence_log.txt', artifacts)
        if evidence_log:
            evidence['evidence_log_completeness'] = self._analyze_evidence_log(evidence_log)
            evidence['evidence_log_authenticity'] = self._calculate_authenticity_score(evidence_log)
        else:
            issues.append("Invalid or missing verification evidence log")

        # Perform cross-validation
        evidence['cross_validation_score'] = self._perform_cross_validation(artifacts)

        # Calculate overall score
        scores = [evidence.get('verification_completeness', 0),
                 evidence.get('verification_authenticity', 0),
                 evidence.get('authenticity_scoring_valid', 0),
                 evidence.get('cross_artifact_consistency', 0),
                 evidence.get('evidence_log_completeness', 0),
                 evidence.get('evidence_log_authenticity', 0),
                 evidence.get('cross_validation_score', 0)]

        overall_score = sum(scores) / len(scores) if scores else 0.0

        status = 'PASS' if overall_score >= self.config.minimum_authenticity_score else 'FAIL'

        return GateResult(
            gate_name='adversarial',
            status=status,
            score=overall_score,
            evidence=evidence,
            issues=issues,
            recommendations=recommendations
        )

    def _analyze_verification_report(self, content: str) -> float:
        """Analyze verification report completeness"""
        # Look for required sections
        required_sections = [
            r'#.*?(?:verification|adversarial|authenticity)',
            r'##.*?(?:findings|results|analysis)',
            r'###.*?(?:methodology|evidence|conclusion)',
            r'(?:score|percentage|authentic|fabricated)'
        ]

        section_score = sum(0.25 for section in required_sections if re.search(section, content, re.IGNORECASE))
        return min(section_score, 1.0)

    def _validate_authenticity_analysis(self, analysis: Dict) -> float:
        """Validate authenticity analysis structure"""
        if not isinstance(analysis, dict):
            return 0.0

        required_keys = ['artifacts', 'overall_score', 'timestamp']
        key_score = sum(0.2 for key in required_keys if key in analysis)

        # Check artifacts structure
        if 'artifacts' in analysis and isinstance(analysis['artifacts'], dict):
            key_score += 0.2
            artifact_count = len(analysis['artifacts'])
            key_score += min(artifact_count / 5, 0.2)  # Expect at least 5 artifacts

        # Check overall score validity
        if 'overall_score' in analysis:
            score = analysis['overall_score']
            if isinstance(score, (int, float)) and 0 <= score <= 1:
                key_score += 0.2

        return min(key_score, 1.0)

    def _analyze_cross_artifact_consistency(self, analysis: Dict) -> float:
        """Analyze cross-artifact consistency"""
        if 'artifacts' not in analysis or not isinstance(analysis['artifacts'], dict):
            return 0.0

        artifacts = analysis['artifacts']
        if len(artifacts) < 2:
            return 0.3  # Need multiple artifacts for consistency analysis

        # Check for reasonable score distribution
        scores = [art.get('score', 0) for art in artifacts.values() if isinstance(art, dict)]
        if not scores:
            return 0.0

        # Look for suspicious patterns (all perfect scores, etc.)
        if all(score == 1.0 for score in scores):
            return 0.5  # Suspiciously perfect

        if all(score == 0.0 for score in scores):
            return 0.5  # Suspiciously terrible

        # Check variance
        mean_score = sum(scores) / len(scores)
        variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
        cv = (variance ** 0.5) / mean_score if mean_score > 0 else 0

        # Moderate variance is good (not too uniform, not too scattered)
        if 0.1 <= cv <= 0.5:
            return 0.9
        elif cv < 0.1:
            return 0.6  # Too uniform, suspicious
        else:
            return 0.7  # Too scattered, but still valid

    def _analyze_evidence_log(self, content: str) -> float:
        """Analyze evidence log completeness"""
        # Look for evidence patterns
        evidence_patterns = [
            r'(?:step|phase|check|verify|validate).*?\d+',
            r'(?:timestamp|time|date).*?\d{4}',
            r'(?:evidence|proof|confirmation)',
            r'(?:cross.?reference|correlation|consistency)',
            r'(?:pattern|anomaly|discrepancy)'
        ]

        evidence_score = sum(0.2 for pattern in evidence_patterns if re.search(pattern, content, re.IGNORECASE))
        return min(evidence_score, 1.0)

    def _perform_cross_validation(self, artifacts: Dict[str, str]) -> float:
        """Perform cross-validation between artifacts"""
        validation_score = 0.0
        checks = 0

        # Check if referenced files actually exist
        for artifact_name, artifact_path in artifacts.items():
            if os.path.exists(artifact_path):
                validation_score += 0.2
            checks += 0.2

        # Check timestamp consistency across artifacts
        timestamps = []
        for artifact_path in artifacts.values():
            if os.path.exists(artifact_path):
                try:
                    with open(artifact_path, 'r') as f:
                        content = f.read()
                        # Look for ISO timestamps
                        ts_matches = re.findall(r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}', content)
                        if ts_matches:
                            timestamps.extend(ts_matches[:3])  # Take first few timestamps
                except:
                    pass

        if len(timestamps) >= 2:
            # Check if timestamps are reasonably close (within same day)
            try:
                parsed_ts = [datetime.fromisoformat(ts.replace(' ', 'T').replace('Z', '+00:00')) for ts in timestamps[:5]]
                time_span = max(parsed_ts) - min(parsed_ts)
                if time_span <= timedelta(hours=24):
                    validation_score += 0.3
                else:
                    validation_score += 0.1  # Too spread out
                checks += 0.3
            except:
                validation_score += 0.1
                checks += 0.3

        # Check content consistency
        contents = []
        for artifact_path in artifacts.values():
            if os.path.exists(artifact_path):
                try:
                    with open(artifact_path, 'r') as f:
                        contents.append(f.read()[:1000])  # First 1000 chars
                except:
                    pass

        if len(contents) >= 2:
            # Simple consistency check - look for common technical terms
            common_terms = ['error', 'test', 'validation', 'artifact', 'score']
            consistency_scores = []

            for i, content1 in enumerate(contents):
                for j, content2 in enumerate(contents):
                    if i != j:
                        common_count = sum(1 for term in common_terms if term in content1.lower() and term in content2.lower())
                        consistency_scores.append(common_count / len(common_terms))

            if consistency_scores:
                avg_consistency = sum(consistency_scores) / len(consistency_scores)
                validation_score += avg_consistency * 0.3
                checks += 0.3

        return validation_score / checks if checks > 0 else 0.0

class QualityGateEngine:
    """Main quality gate validation engine"""

    def __init__(self, config: QualityGateConfig):
        self.config = config
        self.validators = {
            'execution': ExecutionValidator(config),
            'performance': PerformanceValidator(config),
            'security': SecurityValidator(config),
            'coverage': CoverageValidator(config),
            'adversarial': AdversarialValidator(config)
        }
        self.logger = logging.getLogger(__name__)

    def validate_all_gates(self, qa_artifacts: Dict[str, str]) -> QualityGateResult:
        """Validate all quality gates for QA artifacts"""

        self.logger.info("Starting quality gate validation...")

        gate_results = {}
        critical_issues = []
        all_recommendations = []

        for gate_name, validator in self.validators.items():
            self.logger.info(f"Validating {gate_name} gate...")

            try:
                result = validator.validate(qa_artifacts)
                gate_results[gate_name] = result

                if result.status == 'FAIL':
                    critical_issues.extend(result.issues)

                all_recommendations.extend(result.recommendations)

                self.logger.info(f"{gate_name} gate: {result.status} (score: {result.score:.3f})")

            except Exception as e:
                self.logger.error(f"Error validating {gate_name} gate: {e}")
                gate_results[gate_name] = GateResult(
                    gate_name=gate_name,
                    status='ERROR',
                    score=0.0,
                    evidence={'error': str(e)},
                    issues=[f"Validation error: {e}"],
                    recommendations=['Fix validation system error']
                )
                critical_issues.append(f"Validation system error in {gate_name} gate")

        # Determine overall status
        failed_gates = [name for name, result in gate_results.items() if result.status in ['FAIL', 'ERROR']]
        overall_status = 'FAIL' if failed_gates else 'PASS'

        result = QualityGateResult(
            overall_status=overall_status,
            gate_results=gate_results,
            timestamp=datetime.now().isoformat(),
            recommendations=list(set(all_recommendations)),  # Remove duplicates
            critical_issues=critical_issues
        )

        self.logger.info(f"Quality gate validation complete: {overall_status}")

        return result

    def generate_report(self, result: QualityGateResult) -> str:
        """Generate detailed quality gate report"""

        report_lines = [
            "# Quality Gate Validation Report",
            "",
            f"**Timestamp:** {result.timestamp}",
            f"**Overall Status:** {result.overall_status}",
            "",
            "## Gate Results Summary",
            "",
            "| Gate | Status | Score | Issues |",
            "|------|--------|-------|--------|"
        ]

        for gate_name, gate_result in result.gate_results.items():
            issues_count = len(gate_result.issues)
            report_lines.append(f"| {gate_name} | {gate_result.status} | {gate_result.score:.3f} | {issues_count} |")

        report_lines.extend([
            "",
            "## Critical Issues",
            ""
        ])

        if result.critical_issues:
            for issue in result.critical_issues:
                report_lines.append(f"- {issue}")
        else:
            report_lines.append("No critical issues found.")

        report_lines.extend([
            "",
            "## Recommendations",
            ""
        ])

        if result.recommendations:
            for rec in result.recommendations:
                report_lines.append(f"- {rec}")
        else:
            report_lines.append("No recommendations.")

        report_lines.extend([
            "",
            "## Detailed Gate Results",
            ""
        ])

        for gate_name, gate_result in result.gate_results.items():
            report_lines.extend([
                f"### {gate_name.title()} Gate",
                "",
                f"**Status:** {gate_result.status}",
                f"**Score:** {gate_result.score:.3f}",
                "",
                "#### Evidence",
            ])

            for key, value in gate_result.evidence.items():
                if isinstance(value, float):
                    report_lines.append(f"- {key}: {value:.3f}")
                else:
                    report_lines.append(f"- {key}: {value}")

            if gate_result.issues:
                report_lines.extend([
                    "",
                    "#### Issues",
                ])
                for issue in gate_result.issues:
                    report_lines.append(f"- {issue}")

            if gate_result.recommendations:
                report_lines.extend([
                    "",
                    "#### Recommendations",
                ])
                for rec in gate_result.recommendations:
                    report_lines.append(f"- {rec}")

            report_lines.append("")

        return "\n".join(report_lines)

def create_default_config() -> QualityGateConfig:
    """Create default quality gate configuration"""
    return QualityGateConfig(
        execution_required_artifacts=['qa_execution_logs.txt', 'qa_test_results.json', 'qa_system_metrics.json'],
        performance_required_artifacts=['qa_performance_results.json', 'qa_system_specs.json', 'qa_performance_timeline.csv'],
        security_required_artifacts=['qa_security_scan_results.xml', 'qa_vulnerability_details.json', 'qa_security_test_logs.txt'],
        coverage_required_artifacts=['qa_test_coverage_report.html', 'qa_coverage_breakdown.json', 'qa_uncovered_code_analysis.md'],
        adversarial_required_artifacts=['adversarial_verification_report.md', 'artifact_authenticity_analysis.json', 'verification_evidence_log.txt']
    )

if __name__ == "__main__":
    # Example usage
    config = create_default_config()
    engine = QualityGateEngine(config)

    # Example artifacts (replace with actual paths)
    qa_artifacts = {
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

    result = engine.validate_all_gates(qa_artifacts)
    report = engine.generate_report(result)

    print(report)

    # Save report
    with open('qa_quality_gate_report.md', 'w') as f:
        f.write(report)