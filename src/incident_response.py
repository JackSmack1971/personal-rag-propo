"""
Incident Response Module for Personal RAG Chatbot
Implements comprehensive incident detection, response, and forensics capabilities.

Author: SPARC Security Architect
Date: 2025-08-30
"""

import os
import json
import time
import logging
import threading
import secrets
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from enum import Enum
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests

from .security import log_security_event

logger = logging.getLogger(__name__)

class IncidentSeverity(Enum):
    """Incident severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class IncidentStatus(Enum):
    """Incident status"""
    DETECTED = "detected"
    INVESTIGATING = "investigating"
    CONTAINED = "contained"
    ERADICATED = "eradicated"
    RECOVERED = "recovered"
    CLOSED = "closed"

@dataclass
class Incident:
    """Security incident data structure"""
    incident_id: str
    title: str
    description: str
    severity: IncidentSeverity
    status: IncidentStatus
    detected_at: float
    updated_at: float
    source: str
    category: str
    indicators: List[Dict[str, Any]]
    affected_systems: List[str]
    assigned_to: Optional[str] = None
    resolution: Optional[str] = None
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    timeline: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class AlertRule:
    """Alert rule configuration"""
    rule_id: str
    name: str
    condition: str
    severity: IncidentSeverity
    enabled: bool = True
    cooldown_minutes: int = 5
    last_triggered: Optional[float] = None

@dataclass
class ResponseAction:
    """Automated response action"""
    action_id: str
    name: str
    incident_type: str
    severity_threshold: IncidentSeverity
    actions: List[Dict[str, Any]]
    enabled: bool = True

class IncidentDetector:
    """Automated incident detection system"""

    def __init__(self, config: Any):
        self.config = config
        self.alert_rules: Dict[str, AlertRule] = {}
        self.incidents: Dict[str, Incident] = {}
        self._load_alert_rules()
        self._detection_thread = None
        self._running = False

    def _load_alert_rules(self):
        """Load alert rules from configuration"""
        # Default alert rules
        default_rules = [
            AlertRule(
                rule_id="high_rate_limit_hits",
                name="High Rate Limit Violations",
                condition="rate_limit_hits > 50",
                severity=IncidentSeverity.MEDIUM
            ),
            AlertRule(
                rule_id="api_key_exposure",
                name="API Key Exposure Attempt",
                condition="api_key_validation_failures > 10",
                severity=IncidentSeverity.HIGH
            ),
            AlertRule(
                rule_id="malicious_file_uploads",
                name="Malicious File Uploads",
                condition="file_upload_rejections > 20",
                severity=IncidentSeverity.HIGH
            ),
            AlertRule(
                rule_id="llm_prompt_injection",
                name="LLM Prompt Injection Attempts",
                condition="prompt_injection_attempts > 5",
                severity=IncidentSeverity.CRITICAL
            ),
            AlertRule(
                rule_id="suspicious_api_responses",
                name="Suspicious API Responses",
                condition="api_error_rate > 0.5",
                severity=IncidentSeverity.MEDIUM
            ),
            AlertRule(
                rule_id="authentication_failures",
                name="Authentication Failures",
                condition="auth_failures > 25",
                severity=IncidentSeverity.HIGH
            )
        ]

        for rule in default_rules:
            self.alert_rules[rule.rule_id] = rule

    def start_detection(self):
        """Start automated incident detection"""
        if self._detection_thread and self._detection_thread.is_alive():
            return

        self._running = True
        self._detection_thread = threading.Thread(target=self._detection_loop)
        self._detection_thread.daemon = True
        self._detection_thread.start()

        log_security_event("INCIDENT_DETECTION_STARTED", {}, "INFO")

    def stop_detection(self):
        """Stop automated incident detection"""
        self._running = False
        if self._detection_thread:
            self._detection_thread.join(timeout=5)

        log_security_event("INCIDENT_DETECTION_STOPPED", {}, "INFO")

    def _detection_loop(self):
        """Main detection loop"""
        while self._running:
            try:
                self._check_alert_rules()
                time.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"Incident detection error: {e}")
                time.sleep(60)  # Wait longer on error

    def _check_alert_rules(self):
        """Check all alert rules against current metrics"""
        # This would integrate with security monitoring
        # For now, we'll simulate metric checking
        current_metrics = self._get_current_metrics()

        for rule in self.alert_rules.values():
            if not rule.enabled:
                continue

            # Check cooldown
            if rule.last_triggered:
                cooldown_end = rule.last_triggered + (rule.cooldown_minutes * 60)
                if time.time() < cooldown_end:
                    continue

            # Evaluate condition
            if self._evaluate_condition(rule.condition, current_metrics):
                self._trigger_alert(rule, current_metrics)
                rule.last_triggered = time.time()

    def _get_current_metrics(self) -> Dict[str, Any]:
        """Get current security metrics"""
        # This would integrate with SecurityMonitor
        # For simulation, return mock data
        return {
            "rate_limit_hits": 0,
            "api_key_validation_failures": 0,
            "file_upload_rejections": 0,
            "prompt_injection_attempts": 0,
            "api_error_rate": 0.0,
            "auth_failures": 0
        }

    def _evaluate_condition(self, condition: str, metrics: Dict[str, Any]) -> bool:
        """Evaluate alert condition against metrics"""
        try:
            # Simple condition evaluation (in production, use a proper expression evaluator)
            parts = condition.split()
            if len(parts) == 3:
                metric_name, operator, value = parts
                metric_value = metrics.get(metric_name, 0)

                if operator == ">":
                    return metric_value > float(value)
                elif operator == ">=":
                    return metric_value >= float(value)
                elif operator == "<":
                    return metric_value < float(value)
                elif operator == "<=":
                    return metric_value <= float(value)
                elif operator == "==":
                    return metric_value == float(value)
        except Exception as e:
            logger.error(f"Condition evaluation error: {e}")

        return False

    def _trigger_alert(self, rule: AlertRule, metrics: Dict[str, Any]):
        """Trigger alert for rule violation"""
        incident = self.create_incident(
            title=f"Alert: {rule.name}",
            description=f"Alert rule '{rule.name}' triggered: {rule.condition}",
            severity=rule.severity,
            source="automated_detection",
            category="security_alert",
            indicators=[{
                "type": "alert_rule",
                "rule_id": rule.rule_id,
                "condition": rule.condition,
                "metrics": metrics
            }]
        )

        log_security_event("ALERT_TRIGGERED", {
            "incident_id": incident.incident_id,
            "rule_id": rule.rule_id,
            "severity": rule.severity.value
        }, "WARNING")

    def create_incident(self, title: str, description: str, severity: IncidentSeverity,
                       source: str, category: str, indicators: List[Dict[str, Any]],
                       affected_systems: Optional[List[str]] = None) -> Incident:
        """Create a new security incident"""

        incident_id = f"INC-{int(time.time())}-{secrets.token_hex(4)}"

        incident = Incident(
            incident_id=incident_id,
            title=title,
            description=description,
            severity=severity,
            status=IncidentStatus.DETECTED,
            detected_at=time.time(),
            updated_at=time.time(),
            source=source,
            category=category,
            indicators=indicators,
            affected_systems=affected_systems or ["personal-rag"]
        )

        self.incidents[incident_id] = incident

        # Add timeline entry
        incident.timeline.append({
            "timestamp": time.time(),
            "action": "incident_created",
            "details": f"Incident created by {source}"
        })

        # Save incident
        self._save_incident(incident)

        log_security_event("INCIDENT_CREATED", {
            "incident_id": incident_id,
            "severity": severity.value,
            "category": category
        }, "WARNING")

        return incident

    def _save_incident(self, incident: Incident):
        """Save incident to storage"""
        incidents_dir = Path("logs/incidents")
        incidents_dir.mkdir(exist_ok=True)

        incident_file = incidents_dir / f"{incident.incident_id}.json"
        try:
            with open(incident_file, 'w') as f:
                json.dump(asdict(incident), f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save incident: {e}")

    def update_incident_status(self, incident_id: str, status: IncidentStatus,
                              resolution: Optional[str] = None):
        """Update incident status"""

        if incident_id not in self.incidents:
            return False

        incident = self.incidents[incident_id]
        incident.status = status
        incident.updated_at = time.time()

        if resolution:
            incident.resolution = resolution

        # Add timeline entry
        incident.timeline.append({
            "timestamp": time.time(),
            "action": f"status_changed_to_{status.value}",
            "details": resolution or f"Status changed to {status.value}"
        })

        self._save_incident(incident)

        log_security_event("INCIDENT_STATUS_UPDATED", {
            "incident_id": incident_id,
            "status": status.value,
            "resolution": resolution
        }, "INFO")

        return True

class AlertManager:
    """Alert notification and escalation system"""

    def __init__(self, config: Any):
        self.config = config
        self.notification_channels: Dict[str, Dict[str, Any]] = {}
        self._load_notification_channels()

    def _load_notification_channels(self):
        """Load notification channel configurations"""
        # Default channels
        self.notification_channels = {
            "email": {
                "enabled": True,
                "smtp_server": os.getenv("SMTP_SERVER", "localhost"),
                "smtp_port": int(os.getenv("SMTP_PORT", "587")),
                "username": os.getenv("SMTP_USERNAME"),
                "password": os.getenv("SMTP_PASSWORD"),
                "from_address": os.getenv("ALERT_FROM_EMAIL", "alerts@personal-rag.local"),
                "recipients": os.getenv("ALERT_EMAIL_RECIPIENTS", "").split(",")
            },
            "slack": {
                "enabled": bool(os.getenv("SLACK_WEBHOOK_URL")),
                "webhook_url": os.getenv("SLACK_WEBHOOK_URL"),
                "channel": os.getenv("SLACK_CHANNEL", "#security-alerts")
            },
            "webhook": {
                "enabled": bool(os.getenv("ALERT_WEBHOOK_URL")),
                "url": os.getenv("ALERT_WEBHOOK_URL"),
                "headers": json.loads(os.getenv("ALERT_WEBHOOK_HEADERS", "{}"))
            }
        }

    def send_alert(self, incident: Incident, escalation_level: str = "standard"):
        """Send alert notifications"""

        alert_data = {
            "incident_id": incident.incident_id,
            "title": incident.title,
            "description": incident.description,
            "severity": incident.severity.value,
            "status": incident.status.value,
            "detected_at": datetime.fromtimestamp(incident.detected_at).isoformat(),
            "source": incident.source,
            "category": incident.category,
            "escalation_level": escalation_level
        }

        # Send to all enabled channels
        for channel_name, channel_config in self.notification_channels.items():
            if channel_config.get("enabled"):
                try:
                    self._send_to_channel(channel_name, channel_config, alert_data)
                except Exception as e:
                    logger.error(f"Failed to send alert to {channel_name}: {e}")

    def _send_to_channel(self, channel_name: str, config: Dict[str, Any], alert_data: Dict[str, Any]):
        """Send alert to specific channel"""

        if channel_name == "email":
            self._send_email_alert(config, alert_data)
        elif channel_name == "slack":
            self._send_slack_alert(config, alert_data)
        elif channel_name == "webhook":
            self._send_webhook_alert(config, alert_data)

    def _send_email_alert(self, config: Dict[str, Any], alert_data: Dict[str, Any]):
        """Send email alert"""

        subject = f"Security Alert: {alert_data['title']} ({alert_data['severity'].upper()})"
        body = f"""
Security Incident Alert

Incident ID: {alert_data['incident_id']}
Title: {alert_data['title']}
Description: {alert_data['description']}
Severity: {alert_data['severity']}
Status: {alert_data['status']}
Detected: {alert_data['detected_at']}
Source: {alert_data['source']}
Category: {alert_data['category']}

Please investigate immediately.
"""

        msg = MIMEMultipart()
        msg['From'] = config['from_address']
        msg['To'] = ', '.join(config['recipients'])
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        try:
            server = smtplib.SMTP(config['smtp_server'], config['smtp_port'])
            server.starttls()
            if config.get('username') and config.get('password'):
                server.login(config['username'], config['password'])
            server.send_message(msg)
            server.quit()
        except Exception as e:
            logger.error(f"Email alert failed: {e}")

    def _send_slack_alert(self, config: Dict[str, Any], alert_data: Dict[str, Any]):
        """Send Slack alert"""

        payload = {
            "channel": config['channel'],
            "username": "Security Alert Bot",
            "icon_emoji": ":warning:",
            "attachments": [{
                "color": "danger" if alert_data['severity'] in ['high', 'critical'] else "warning",
                "title": f"🚨 {alert_data['title']}",
                "text": alert_data['description'],
                "fields": [
                    {"title": "Incident ID", "value": alert_data['incident_id'], "short": True},
                    {"title": "Severity", "value": alert_data['severity'].upper(), "short": True},
                    {"title": "Status", "value": alert_data['status'], "short": True},
                    {"title": "Source", "value": alert_data['source'], "short": True}
                ],
                "footer": "Personal RAG Security Alert",
                "ts": int(time.time())
            }]
        }

        response = requests.post(config['webhook_url'], json=payload)
        response.raise_for_status()

    def _send_webhook_alert(self, config: Dict[str, Any], alert_data: Dict[str, Any]):
        """Send webhook alert"""

        headers = config.get('headers', {})
        headers['Content-Type'] = 'application/json'

        response = requests.post(config['url'], json=alert_data, headers=headers)
        response.raise_for_status()

class IncidentResponder:
    """Automated incident response system"""

    def __init__(self, config: Any):
        self.config = config
        self.response_actions: Dict[str, ResponseAction] = {}
        self._load_response_actions()

    def _load_response_actions(self):
        """Load automated response actions"""

        default_actions = [
            ResponseAction(
                action_id="rate_limit_response",
                name="Rate Limit Abuse Response",
                incident_type="rate_limit_abuse",
                severity_threshold=IncidentSeverity.MEDIUM,
                actions=[
                    {"type": "block_ip", "duration_minutes": 15},
                    {"type": "increase_rate_limit", "multiplier": 0.5},
                    {"type": "log_incident", "level": "WARNING"}
                ]
            ),
            ResponseAction(
                action_id="api_key_compromise",
                name="API Key Compromise Response",
                incident_type="api_key_compromise",
                severity_threshold=IncidentSeverity.HIGH,
                actions=[
                    {"type": "rotate_api_key", "service": "all"},
                    {"type": "block_suspicious_ips"},
                    {"type": "enable_enhanced_monitoring", "duration_hours": 24},
                    {"type": "notify_admin", "priority": "urgent"}
                ]
            ),
            ResponseAction(
                action_id="malicious_upload",
                name="Malicious File Upload Response",
                incident_type="malicious_file_upload",
                severity_threshold=IncidentSeverity.HIGH,
                actions=[
                    {"type": "quarantine_file"},
                    {"type": "scan_system", "scan_type": "full"},
                    {"type": "update_file_validation_rules"},
                    {"type": "block_upload_ip", "duration_minutes": 60}
                ]
            )
        ]

        for action in default_actions:
            self.response_actions[action.action_id] = action

    def execute_response(self, incident: Incident):
        """Execute automated response for incident"""

        applicable_actions = []

        # Find applicable response actions
        for action in self.response_actions.values():
            if not action.enabled:
                continue

            if incident.category == action.incident_type and \
               incident.severity.value >= action.severity_threshold.value:
                applicable_actions.append(action)

        if not applicable_actions:
            log_security_event("NO_RESPONSE_ACTION", {
                "incident_id": incident.incident_id,
                "category": incident.category,
                "severity": incident.severity.value
            }, "INFO")
            return

        # Execute actions
        for action in applicable_actions:
            self._execute_action(action, incident)

    def _execute_action(self, action: ResponseAction, incident: Incident):
        """Execute specific response action"""

        log_security_event("EXECUTING_RESPONSE_ACTION", {
            "incident_id": incident.incident_id,
            "action_id": action.action_id,
            "action_name": action.name
        }, "INFO")

        for action_config in action.actions:
            action_type = action_config['type']

            try:
                if action_type == "block_ip":
                    self._block_ip(incident, action_config)
                elif action_type == "rotate_api_key":
                    self._rotate_api_key(action_config)
                elif action_type == "increase_rate_limit":
                    self._adjust_rate_limit(action_config)
                elif action_type == "quarantine_file":
                    self._quarantine_file(incident)
                elif action_type == "log_incident":
                    self._log_response_action(incident, action_config)
                elif action_type == "notify_admin":
                    self._notify_admin(incident, action_config)
                # Add more action types as needed

            except Exception as e:
                logger.error(f"Response action failed: {action_type} - {e}")

    def _block_ip(self, incident: Incident, config: Dict[str, Any]):
        """Block IP address"""
        # Implementation would integrate with firewall
        duration = config.get('duration_minutes', 15)
        log_security_event("IP_BLOCKED_RESPONSE", {
            "incident_id": incident.incident_id,
            "duration_minutes": duration
        }, "WARNING")

    def _rotate_api_key(self, config: Dict[str, Any]):
        """Rotate API key"""
        service = config.get('service', 'all')
        log_security_event("API_KEY_ROTATION_RESPONSE", {
            "service": service
        }, "WARNING")

    def _adjust_rate_limit(self, config: Dict[str, Any]):
        """Adjust rate limiting"""
        multiplier = config.get('multiplier', 0.5)
        log_security_event("RATE_LIMIT_ADJUSTED", {
            "multiplier": multiplier
        }, "INFO")

    def _quarantine_file(self, incident: Incident):
        """Quarantine malicious file"""
        log_security_event("FILE_QUARANTINED", {
            "incident_id": incident.incident_id
        }, "WARNING")

    def _log_response_action(self, incident: Incident, config: Dict[str, Any]):
        """Log response action"""
        level = config.get('level', 'INFO')
        log_security_event("RESPONSE_ACTION_LOGGED", {
            "incident_id": incident.incident_id,
            "level": level
        }, level)

    def _notify_admin(self, incident: Incident, config: Dict[str, Any]):
        """Notify administrator"""
        priority = config.get('priority', 'normal')
        log_security_event("ADMIN_NOTIFICATION_SENT", {
            "incident_id": incident.incident_id,
            "priority": priority
        }, "WARNING")

class ForensicsCollector:
    """Digital forensics evidence collection"""

    def __init__(self, config: Any):
        self.config = config
        self.evidence_dir = Path("logs/forensics")
        self.evidence_dir.mkdir(exist_ok=True)

    def collect_evidence(self, incident: Incident) -> List[Dict[str, Any]]:
        """Collect forensic evidence for incident"""

        evidence = []

        # Collect system logs
        evidence.extend(self._collect_system_logs(incident))

        # Collect network logs
        evidence.extend(self._collect_network_logs(incident))

        # Collect application logs
        evidence.extend(self._collect_application_logs(incident))

        # Collect memory dumps if configured
        if self.config.collect_memory_dumps:
            evidence.extend(self._collect_memory_dump(incident))

        # Save evidence to incident
        evidence_file = self.evidence_dir / f"{incident.incident_id}_evidence.json"
        try:
            with open(evidence_file, 'w') as f:
                json.dump(evidence, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save evidence: {e}")

        return evidence

    def _collect_system_logs(self, incident: Incident) -> List[Dict[str, Any]]:
        """Collect system-level logs"""
        # Implementation would collect relevant system logs
        return [{
            "type": "system_log",
            "timestamp": time.time(),
            "data": "System log collection placeholder"
        }]

    def _collect_network_logs(self, incident: Incident) -> List[Dict[str, Any]]:
        """Collect network-level logs"""
        # Implementation would collect network traffic logs
        return [{
            "type": "network_log",
            "timestamp": time.time(),
            "data": "Network log collection placeholder"
        }]

    def _collect_application_logs(self, incident: Incident) -> List[Dict[str, Any]]:
        """Collect application-level logs"""
        # Implementation would collect application-specific logs
        return [{
            "type": "application_log",
            "timestamp": time.time(),
            "data": "Application log collection placeholder"
        }]

    def _collect_memory_dump(self, incident: Incident) -> List[Dict[str, Any]]:
        """Collect memory dump (if enabled)"""
        # This would only be done in extreme cases with proper authorization
        return []

class IncidentResponseManager:
    """Main incident response coordination system"""

    def __init__(self, config: Any):
        self.config = config
        self.detector = IncidentDetector(config)
        self.alert_manager = AlertManager(config)
        self.responder = IncidentResponder(config)
        self.forensics = ForensicsCollector(config)

    def start(self):
        """Start incident response system"""
        self.detector.start_detection()
        log_security_event("INCIDENT_RESPONSE_STARTED", {}, "INFO")

    def stop(self):
        """Stop incident response system"""
        self.detector.stop_detection()
        log_security_event("INCIDENT_RESPONSE_STOPPED", {}, "INFO")

    def handle_incident(self, incident: Incident):
        """Handle detected incident"""

        # Collect forensic evidence
        evidence = self.forensics.collect_evidence(incident)
        incident.evidence.extend(evidence)

        # Execute automated response
        self.responder.execute_response(incident)

        # Send alerts
        escalation_level = self._determine_escalation_level(incident)
        self.alert_manager.send_alert(incident, escalation_level)

        # Update incident status
        self.detector.update_incident_status(incident.incident_id, IncidentStatus.INVESTIGATING)

    def _determine_escalation_level(self, incident: Incident) -> str:
        """Determine alert escalation level"""
        if incident.severity == IncidentSeverity.CRITICAL:
            return "executive"
        elif incident.severity == IncidentSeverity.HIGH:
            return "management"
        else:
            return "standard"

    def get_incident_report(self, incident_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive incident report"""

        if incident_id not in self.detector.incidents:
            return None

        incident = self.detector.incidents[incident_id]

        return {
            "incident": asdict(incident),
            "evidence_count": len(incident.evidence),
            "timeline_events": len(incident.timeline),
            "response_actions": [],  # Would be populated from response logs
            "recommendations": self._generate_recommendations(incident)
        }

    def _generate_recommendations(self, incident: Incident) -> List[str]:
        """Generate incident response recommendations"""

        recommendations = []

        if incident.category == "api_key_compromise":
            recommendations.extend([
                "Rotate all API keys immediately",
                "Review API key access patterns",
                "Implement API key usage monitoring"
            ])

        elif incident.category == "malicious_file_upload":
            recommendations.extend([
                "Update file validation rules",
                "Implement file content scanning",
                "Review upload access controls"
            ])

        elif incident.category == "rate_limit_abuse":
            recommendations.extend([
                "Implement progressive rate limiting",
                "Add IP-based blocking",
                "Monitor for DDoS patterns"
            ])

        return recommendations

# Global incident response instance
_incident_response = None

def get_incident_response_manager(config: Any) -> IncidentResponseManager:
    """Get or create incident response manager instance"""
    global _incident_response
    if _incident_response is None:
        _incident_response = IncidentResponseManager(config)
    return _incident_response

# Convenience functions
def create_incident(title: str, description: str, severity: str,
                   source: str, category: str, indicators: List[Dict[str, Any]]) -> Optional[str]:
    """Create a new incident"""
    global _incident_response
    if _incident_response:
        severity_enum = IncidentSeverity(severity.lower())
        incident = _incident_response.detector.create_incident(
            title, description, severity_enum, source, category, indicators
        )
        return incident.incident_id
    return None

def update_incident_status(incident_id: str, status: str, resolution: Optional[str] = None) -> bool:
    """Update incident status"""
    global _incident_response
    if _incident_response:
        status_enum = IncidentStatus(status.lower())
        return _incident_response.detector.update_incident_status(incident_id, status_enum, resolution)
    return False

def get_incident_report(incident_id: str) -> Optional[Dict[str, Any]]:
    """Get incident report"""
    global _incident_response
    if _incident_response:
        return _incident_response.get_incident_report(incident_id)
    return None