"""
Authentication and Authorization Module for Personal RAG Chatbot
Implements secure user authentication, session management, and role-based access control.

Author: SPARC Security Architect
Date: 2025-08-30
"""

import os
import re
import hashlib
import secrets
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import jwt
from pathlib import Path

from .security import log_security_event

@dataclass
class User:
    """User account information"""
    user_id: str
    username: str
    email: str
    role: str
    is_active: bool = True
    created_at: Optional[float] = None
    last_login: Optional[float] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()

@dataclass
class Session:
    """User session information"""
    session_id: str
    user_id: str
    created_at: float
    expires_at: float
    ip_address: str
    user_agent: str
    is_active: bool = True

@dataclass
class AuthConfig:
    """Authentication configuration"""
    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    session_timeout_minutes: int = 30
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 15
    password_min_length: int = 12
    require_special_chars: bool = True
    require_numbers: bool = True
    enable_mfa: bool = False

class UserManager:
    """User account management"""

    def __init__(self, config: AuthConfig):
        self.config = config
        self.users_file = Path("config/users.json")
        self.users_file.parent.mkdir(exist_ok=True)
        self._users: Dict[str, User] = {}
        self._passwords: Dict[str, str] = {}  # In production, use proper password hashing
        self._load_users()

    def _load_users(self):
        """Load users from storage"""
        try:
            if self.users_file.exists():
                with open(self.users_file, 'r') as f:
                    data = json.load(f)
                    for user_data in data.get('users', []):
                        user = User(**user_data)
                        self._users[user.user_id] = user
                    self._passwords = data.get('passwords', {})
        except Exception as e:
            log_security_event("USER_LOAD_ERROR", {"error": str(e)}, "ERROR")

    def _save_users(self):
        """Save users to storage"""
        try:
            data = {
                'users': [vars(user) for user in self._users.values()],
                'passwords': self._passwords
            }
            with open(self.users_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            log_security_event("USER_SAVE_ERROR", {"error": str(e)}, "ERROR")

    def create_user(self, username: str, email: str, password: str, role: str = "user") -> Tuple[bool, str]:
        """Create a new user account"""

        # Validate input
        if not self._validate_username(username):
            return False, "Invalid username format"

        if not self._validate_email(email):
            return False, "Invalid email format"

        if not self._validate_password(password):
            return False, f"Password must be at least {self.config.password_min_length} characters"

        # Check if user already exists
        if any(user.username == username for user in self._users.values()):
            return False, "Username already exists"

        if any(user.email == email for user in self._users.values()):
            return False, "Email already registered"

        # Create user
        user_id = secrets.token_hex(16)
        hashed_password = self._hash_password(password)

        user = User(
            user_id=user_id,
            username=username,
            email=email,
            role=role
        )

        self._users[user_id] = user
        self._passwords[user_id] = hashed_password
        self._save_users()

        log_security_event("USER_CREATED", {
            "user_id": user_id,
            "username": username,
            "role": role
        }, "INFO")

        return True, "User created successfully"

    def authenticate_user(self, username: str, password: str, ip_address: str = "unknown") -> Tuple[bool, Optional[User]]:
        """Authenticate user credentials"""

        # Find user
        user = None
        for u in self._users.values():
            if u.username == username and u.is_active:
                user = u
                break

        if not user:
            log_security_event("AUTH_FAILED_USER_NOT_FOUND", {
                "username": username,
                "ip_address": ip_address
            }, "WARNING")
            return False, None

        # Check password
        hashed_password = self._hash_password(password)
        if self._passwords.get(user.user_id) != hashed_password:
            log_security_event("AUTH_FAILED_INVALID_PASSWORD", {
                "user_id": user.user_id,
                "username": username,
                "ip_address": ip_address
            }, "WARNING")
            return False, None

        # Update last login
        user.last_login = time.time()
        self._save_users()

        log_security_event("AUTH_SUCCESS", {
            "user_id": user.user_id,
            "username": username,
            "ip_address": ip_address
        }, "INFO")

        return True, user

    def _validate_username(self, username: str) -> bool:
        """Validate username format"""
        if not username or len(username) < 3 or len(username) > 50:
            return False
        # Allow alphanumeric, underscore, and hyphen
        return bool(re.match(r'^[a-zA-Z0-9_-]+$', username))

    def _validate_email(self, email: str) -> bool:
        """Validate email format"""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))

    def _validate_password(self, password: str) -> bool:
        """Validate password strength"""
        if len(password) < self.config.password_min_length:
            return False

        if self.config.require_special_chars and not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            return False

        if self.config.require_numbers and not re.search(r'\d', password):
            return False

        return True

    def _hash_password(self, password: str) -> str:
        """Hash password (in production, use proper password hashing like bcrypt)"""
        # WARNING: This is a simple hash for demonstration
        # In production, use bcrypt, scrypt, or argon2
        return hashlib.sha256(password.encode()).hexdigest()

class SessionManager:
    """Session management for authenticated users"""

    def __init__(self, config: AuthConfig):
        self.config = config
        self.sessions_file = Path("config/sessions.json")
        self.sessions_file.parent.mkdir(exist_ok=True)
        self._sessions: Dict[str, Session] = {}
        self._load_sessions()

    def _load_sessions(self):
        """Load sessions from storage"""
        try:
            if self.sessions_file.exists():
                with open(self.sessions_file, 'r') as f:
                    data = json.load(f)
                    for session_data in data.get('sessions', []):
                        session = Session(**session_data)
                        self._sessions[session.session_id] = session
        except Exception as e:
            log_security_event("SESSION_LOAD_ERROR", {"error": str(e)}, "ERROR")

    def _save_sessions(self):
        """Save sessions to storage"""
        try:
            data = {
                'sessions': [vars(session) for session in self._sessions.values()]
            }
            with open(self.sessions_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            log_security_event("SESSION_SAVE_ERROR", {"error": str(e)}, "ERROR")

    def create_session(self, user: User, ip_address: str = "unknown",
                      user_agent: str = "unknown") -> str:
        """Create a new session for user"""

        session_id = secrets.token_hex(32)
        now = time.time()
        expires_at = now + (self.config.session_timeout_minutes * 60)

        session = Session(
            session_id=session_id,
            user_id=user.user_id,
            created_at=now,
            expires_at=expires_at,
            ip_address=ip_address,
            user_agent=user_agent
        )

        self._sessions[session_id] = session
        self._save_sessions()

        log_security_event("SESSION_CREATED", {
            "session_id": session_id,
            "user_id": user.user_id,
            "ip_address": ip_address
        }, "INFO")

        return session_id

    def validate_session(self, session_id: str) -> Optional[Session]:
        """Validate session and return session if valid"""

        if session_id not in self._sessions:
            return None

        session = self._sessions[session_id]

        if not session.is_active:
            return None

        if time.time() > session.expires_at:
            session.is_active = False
            self._save_sessions()
            log_security_event("SESSION_EXPIRED", {
                "session_id": session_id,
                "user_id": session.user_id
            }, "INFO")
            return None

        return session

    def destroy_session(self, session_id: str):
        """Destroy a session"""

        if session_id in self._sessions:
            session = self._sessions[session_id]
            session.is_active = False
            self._save_sessions()

            log_security_event("SESSION_DESTROYED", {
                "session_id": session_id,
                "user_id": session.user_id
            }, "INFO")

    def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        now = time.time()
        expired_sessions = [
            session_id for session_id, session in self._sessions.items()
            if not session.is_active or now > session.expires_at
        ]

        for session_id in expired_sessions:
            del self._sessions[session_id]

        if expired_sessions:
            self._save_sessions()
            log_security_event("SESSIONS_CLEANED", {
                "expired_count": len(expired_sessions)
            }, "INFO")

class AuthorizationManager:
    """Role-based authorization management"""

    def __init__(self):
        self.roles_file = Path("config/roles.json")
        self.roles_file.parent.mkdir(exist_ok=True)
        self._role_permissions: Dict[str, List[str]] = {}
        self._load_roles()

    def _load_roles(self):
        """Load role permissions from storage"""
        try:
            if self.roles_file.exists():
                with open(self.roles_file, 'r') as f:
                    self._role_permissions = json.load(f)
            else:
                # Default role permissions
                self._role_permissions = {
                    "admin": [
                        "file.upload", "file.delete", "user.manage",
                        "system.config", "security.view", "chat.unlimited"
                    ],
                    "user": [
                        "file.upload", "chat.basic", "profile.view"
                    ],
                    "viewer": [
                        "chat.view", "profile.view"
                    ]
                }
                self._save_roles()
        except Exception as e:
            log_security_event("ROLE_LOAD_ERROR", {"error": str(e)}, "ERROR")

    def _save_roles(self):
        """Save role permissions to storage"""
        try:
            with open(self.roles_file, 'w') as f:
                json.dump(self._role_permissions, f, indent=2)
        except Exception as e:
            log_security_event("ROLE_SAVE_ERROR", {"error": str(e)}, "ERROR")

    def has_permission(self, user_role: str, permission: str) -> bool:
        """Check if user role has specific permission"""
        if user_role not in self._role_permissions:
            return False

        return permission in self._role_permissions[user_role]

    def get_role_permissions(self, role: str) -> List[str]:
        """Get all permissions for a role"""
        return self._role_permissions.get(role, [])

class JWTManager:
    """JWT token management for API authentication"""

    def __init__(self, config: AuthConfig):
        self.config = config

    def create_token(self, user: User, session_id: str) -> str:
        """Create JWT token for user"""

        payload = {
            "user_id": user.user_id,
            "username": user.username,
            "role": user.role,
            "session_id": session_id,
            "iat": int(time.time()),
            "exp": int(time.time()) + (self.config.session_timeout_minutes * 60)
        }

        token = jwt.encode(payload, self.config.jwt_secret_key, algorithm=self.config.jwt_algorithm)
        return token

    def validate_token(self, token: str) -> Optional[Dict]:
        """Validate JWT token and return payload"""

        try:
            payload = jwt.decode(token, self.config.jwt_secret_key,
                               algorithms=[self.config.jwt_algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            log_security_event("JWT_EXPIRED", {}, "WARNING")
            return None
        except jwt.InvalidTokenError:
            log_security_event("JWT_INVALID", {}, "WARNING")
            return None

class AuthManager:
    """Main authentication and authorization manager"""

    def __init__(self, config: AuthConfig):
        self.config = config
        self.user_manager = UserManager(config)
        self.session_manager = SessionManager(config)
        self.auth_manager = AuthorizationManager()
        self.jwt_manager = JWTManager(config)

    def login(self, username: str, password: str, ip_address: str = "unknown",
             user_agent: str = "unknown") -> Tuple[bool, str, Optional[str]]:
        """User login process"""

        # Authenticate user
        success, user = self.user_manager.authenticate_user(username, password, ip_address)
        if not success or not user:
            return False, "Invalid credentials", None

        # Create session
        session_id = self.session_manager.create_session(user, ip_address, user_agent)

        # Create JWT token
        token = self.jwt_manager.create_token(user, session_id)

        return True, "Login successful", token

    def logout(self, token: str):
        """User logout process"""

        payload = self.jwt_manager.validate_token(token)
        if payload and "session_id" in payload:
            self.session_manager.destroy_session(payload["session_id"])

    def validate_request(self, token: str, required_permission: Optional[str] = None) -> Tuple[bool, Optional[Dict]]:
        """Validate authentication and authorization for request"""

        # Validate token
        payload = self.jwt_manager.validate_token(token)
        if not payload:
            return False, None

        # Validate session
        session = self.session_manager.validate_session(payload.get("session_id", ""))
        if not session:
            return False, None

        # Check permission if required
        if required_permission:
            user_role = payload.get("role", "user")
            if not self.auth_manager.has_permission(user_role, required_permission):
                log_security_event("AUTH_PERMISSION_DENIED", {
                    "user_id": payload.get("user_id"),
                    "permission": required_permission,
                    "role": user_role
                }, "WARNING")
                return False, None

        return True, payload

    def create_default_admin(self):
        """Create default admin user if no users exist"""
        if not self.user_manager._users:
            success, message = self.user_manager.create_user(
                username="admin",
                email="admin@example.com",
                password="Admin123!@#",
                role="admin"
            )
            if success:
                log_security_event("DEFAULT_ADMIN_CREATED", {
                    "username": "admin",
                    "message": "Default admin user created"
                }, "INFO")
            else:
                log_security_event("DEFAULT_ADMIN_CREATE_FAILED", {
                    "message": message
                }, "ERROR")

# Global authentication configuration
_auth_config = AuthConfig(
    jwt_secret_key=os.getenv("JWT_SECRET_KEY", secrets.token_hex(32)),
    enable_mfa=os.getenv("ENABLE_MFA", "false").lower() == "true"
)

# Global authentication manager instance
auth_manager = AuthManager(_auth_config)

# Initialize with default admin if needed
auth_manager.create_default_admin()

def login_required(permission: Optional[str] = None):
    """Decorator for requiring authentication and authorization"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Extract token from request context (would be implemented based on framework)
            token = kwargs.get('token') or getattr(args[0] if args else None, 'token', None)

            if not token:
                raise PermissionError("Authentication required")

            valid, payload = auth_manager.validate_request(token, permission)
            if not valid:
                raise PermissionError("Authentication or authorization failed")

            # Add user context to kwargs
            kwargs['user_context'] = payload
            return func(*args, **kwargs)
        return wrapper
    return decorator