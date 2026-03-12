"""Web Application Reconnaissance Agent

Deep-scans a web application to map the attack surface before active
exploitation.  Discovers endpoints, technologies, hidden files,
API schemas, and input vectors.

Designed for OWASP Juice Shop and similar deliberately vulnerable
web apps, but works against any HTTP target.

Math deltas:
  k_gain = endpoints_found * 0.15 + tech_count * 0.1
  s_inc  = 0.05  (passive fingerprinting is nearly silent)
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse

import aiohttp
from loguru import logger

from blackpanther.core.access import AccessPropagation
from blackpanther.core.knowledge import KnowledgeEvolution
from blackpanther.core.suspicion import SuspicionField
from blackpanther.settings import Settings, get_settings

from .base import AgentResult, BaseAgent
from .resilience import normalize_base_url


# ------------------------------------------------------------------
# Data transfer objects
# ------------------------------------------------------------------

@dataclass
class WebEndpoint:
    """A discovered web endpoint."""
    url: str
    method: str = "GET"
    status_code: int = 0
    content_type: str = ""
    content_length: int = 0
    auth_required: bool = False
    parameters: List[str] = field(default_factory=list)
    forms: List[Dict[str, Any]] = field(default_factory=list)
    interesting: bool = False
    notes: str = ""


@dataclass
class WebFingerprint:
    """Technology fingerprint of the target."""
    server: str = ""
    framework: str = ""
    technologies: List[str] = field(default_factory=list)
    cookies: Dict[str, str] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    security_headers_missing: List[str] = field(default_factory=list)
    error_handling: str = ""
    js_frameworks: List[str] = field(default_factory=list)


@dataclass
class WebReconResult:
    """Full reconnaissance output."""
    base_url: str
    fingerprint: WebFingerprint = field(default_factory=WebFingerprint)
    endpoints: List[WebEndpoint] = field(default_factory=list)
    api_endpoints: List[WebEndpoint] = field(default_factory=list)
    hidden_files: List[WebEndpoint] = field(default_factory=list)
    forms: List[Dict[str, Any]] = field(default_factory=list)
    emails: List[str] = field(default_factory=list)
    interesting_strings: List[str] = field(default_factory=list)


# ------------------------------------------------------------------
# Link extractor
# ------------------------------------------------------------------

class _LinkExtractor(HTMLParser):
    """Extract links, forms, and scripts from HTML."""

    def __init__(self, base_url: str) -> None:
        super().__init__()
        self.base_url = base_url
        self.links: Set[str] = set()
        self.forms: List[Dict[str, Any]] = []
        self.scripts: List[str] = []
        self._current_form: Optional[Dict[str, Any]] = None

    def handle_starttag(self, tag: str, attrs: list) -> None:
        attr_dict = dict(attrs)

        if tag == "a" and "href" in attr_dict:
            href = attr_dict["href"]
            if not href.startswith(("javascript:", "mailto:", "#", "data:")):
                self.links.add(urljoin(self.base_url, href))

        elif tag == "form":
            self._current_form = {
                "action": urljoin(self.base_url, attr_dict.get("action", "")),
                "method": attr_dict.get("method", "GET").upper(),
                "inputs": [],
            }

        elif tag == "input" and self._current_form is not None:
            self._current_form["inputs"].append({
                "name": attr_dict.get("name", ""),
                "type": attr_dict.get("type", "text"),
                "value": attr_dict.get("value", ""),
            })

        elif tag == "script":
            src = attr_dict.get("src", "")
            if src:
                self.scripts.append(urljoin(self.base_url, src))

        for attr in ("src", "href", "action"):
            val = attr_dict.get(attr, "")
            if val and val.startswith("/api"):
                self.links.add(urljoin(self.base_url, val))

    def handle_endtag(self, tag: str) -> None:
        if tag == "form" and self._current_form is not None:
            self.forms.append(self._current_form)
            self._current_form = None


# ------------------------------------------------------------------
# Known paths to probe (general + Juice Shop specific)
# ------------------------------------------------------------------

DISCOVERY_PATHS: List[Tuple[str, str]] = [
    # Standard recon
    ("/robots.txt", "Robots exclusion"),
    ("/.well-known/security.txt", "Security contact"),
    ("/sitemap.xml", "Sitemap"),
    ("/crossdomain.xml", "Flash crossdomain policy"),
    ("/.git/HEAD", "Git repository exposure"),
    ("/.env", "Environment file exposure"),
    ("/server-info", "Apache server info"),
    ("/server-status", "Apache server status"),
    ("/.DS_Store", "macOS directory listing"),
    ("/backup", "Backup directory"),
    ("/dump.sql", "SQL dump"),
    ("/config.json", "Configuration file"),
    ("/package.json", "Node.js package info"),
    ("/swagger.json", "Swagger/OpenAPI spec"),
    ("/api-docs", "API documentation"),
    # Juice Shop specific
    ("/ftp", "FTP directory listing"),
    ("/ftp/acquisitions.md", "Confidential document"),
    ("/ftp/coupons_2013.md.bak", "Backup coupon file"),
    ("/ftp/eastere.gg", "Easter egg file"),
    ("/ftp/encrypt.pyc", "Compiled Python file"),
    ("/ftp/incident-support.kdbx", "KeePass database"),
    ("/ftp/legal.md", "Legal information"),
    ("/ftp/package.json.bak", "Package backup"),
    ("/ftp/quarantine", "Quarantine directory"),
    ("/ftp/suspicious_errors.yml", "Error logs"),
    ("/encryptionkeys", "Encryption keys directory"),
    ("/encryptionkeys/jwt.pub", "JWT public key"),
    ("/metrics", "Prometheus metrics"),
    ("/promotion", "Promotion video"),
    ("/video", "Video directory"),
    ("/snippets", "Code snippets"),
    ("/dataerasure", "Data erasure form"),
    ("/profile", "User profile"),
    ("/accounting", "Accounting page"),
    ("/privacy-security/privacy-policy", "Privacy policy"),
    ("/privacy-security/change-password", "Password change"),
    ("/b2b/v2/orders", "B2B orders API"),
    ("/assets/public/images/padding", "Image directory"),
    # Admin and dashboard
    ("/administration", "Admin panel"),
    ("/admin", "Admin page"),
    ("/administrator", "Administrator page"),
    # API endpoints
    ("/api", "API root"),
    ("/api/Users", "Users API"),
    ("/api/Cards", "Credit cards API"),
    ("/api/Products", "Products API"),
    ("/api/Feedbacks", "Feedback API"),
    ("/api/BasketItems", "Basket items API"),
    ("/api/Complaints", "Complaints API"),
    ("/api/Recycles", "Recycles API"),
    ("/api/Challenges", "Challenges API"),
    ("/api/SecurityQuestions", "Security questions API"),
    ("/api/SecurityAnswers", "Security answers API"),
    ("/api/Deliverys", "Delivery API"),
    ("/api/Quantitys", "Quantity API"),
    ("/api/Addresss", "Address API"),
    ("/api/Wallets", "Wallet API"),
    ("/api/Memorys", "Memory API"),
    # REST endpoints
    ("/rest/user/login", "Login endpoint"),
    ("/rest/user/whoami", "Session info"),
    ("/rest/user/authentication-details", "Auth details"),
    ("/rest/user/change-password", "Password change"),
    ("/rest/products/search", "Product search"),
    ("/rest/products/reviews", "Product reviews"),
    ("/rest/basket", "Shopping basket"),
    ("/rest/saveLoginIp", "Login IP logging"),
    ("/rest/deluxe-membership", "Deluxe membership"),
    ("/rest/repeat-notification", "Notification repeat"),
    ("/rest/continue-code", "Continue code"),
    ("/rest/chatbot/status", "Chatbot status"),
    ("/rest/chatbot/respond", "Chatbot respond"),
    ("/rest/2fa/status", "2FA status"),
    ("/rest/web3/submitKey", "Web3 submit key"),
    ("/rest/web3/nftUnlocked", "NFT unlocked"),
    ("/rest/web3/nftMintListen", "NFT mint listen"),
    ("/redirect", "Open redirect"),
    # Swagger / OpenAPI
    ("/api-docs/swagger.json", "Swagger spec"),
    ("/swagger-ui.html", "Swagger UI"),
    # Score board
    ("/#/score-board", "Score board (SPA)"),
    ("/score-board", "Score board"),
]

SECURITY_HEADERS = [
    "X-Content-Type-Options",
    "X-Frame-Options",
    "X-XSS-Protection",
    "Content-Security-Policy",
    "Strict-Transport-Security",
    "Referrer-Policy",
    "Permissions-Policy",
    "X-Permitted-Cross-Domain-Policies",
]

_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_JWT_RE = re.compile(r"eyJ[a-zA-Z0-9_-]+\.eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+")
_VERSION_RE = re.compile(r"(?:version|v)[\s:\"']*(\d+\.\d+[\.\d]*)", re.I)


# ------------------------------------------------------------------
# WebReconAgent
# ------------------------------------------------------------------

class WebReconAgent(BaseAgent):
    """Web application reconnaissance via HTTP crawling and probing.

    Discovers endpoints, technologies, hidden files, API routes, forms,
    and security misconfigurations.  Designed for maximum coverage on
    OWASP Juice Shop but generalizes to any web target.

    Math deltas:
      k_gain = discovered_endpoints * 0.15  (high intel value)
      s_inc  = 0.05                          (mostly passive)
    """

    name = "web_recon"

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
        self._max_pages = cfg.web_max_pages
        self._max_depth = cfg.web_max_crawl_depth
        self._delay = cfg.web_request_delay
        self._verify_ssl = cfg.web_verify_ssl
        self._concurrency = cfg.web_concurrent_requests

    async def _execute(self, target: str, **kwargs: Any) -> AgentResult:
        base_url = normalize_base_url(target)
        recon = WebReconResult(base_url=base_url)

        connector = aiohttp.TCPConnector(
            ssl=self._verify_ssl,
            limit=self._concurrency,
        )
        async with aiohttp.ClientSession(
            timeout=self._timeout,
            connector=connector,
            headers={"User-Agent": self._user_agent},
        ) as session:
            recon.fingerprint = await self._fingerprint(session, base_url)

            discovered = await self._discover_paths(session, base_url)
            recon.hidden_files = [ep for ep in discovered if ep.interesting]
            recon.endpoints.extend(discovered)

            crawled = await self._crawl(session, base_url)
            recon.endpoints.extend(crawled)

            api_eps = await self._enumerate_api(session, base_url)
            recon.api_endpoints = api_eps
            recon.endpoints.extend(api_eps)

        all_eps = len(recon.endpoints)
        tech_count = len(recon.fingerprint.technologies)
        hidden_count = len(recon.hidden_files)

        k_gain = all_eps * 0.15 + tech_count * 0.1 + hidden_count * 0.3
        s_inc = 0.05 + (all_eps / 1000.0)

        return AgentResult(
            k_gain=min(k_gain, 15.0),
            s_inc=min(s_inc, 0.15),
            a_delta=0.1 if hidden_count > 0 else 0.0,
            raw_data={
                "base_url": base_url,
                "fingerprint": {
                    "server": recon.fingerprint.server,
                    "framework": recon.fingerprint.framework,
                    "technologies": recon.fingerprint.technologies,
                    "security_headers_missing": recon.fingerprint.security_headers_missing,
                    "js_frameworks": recon.fingerprint.js_frameworks,
                    "error_handling": recon.fingerprint.error_handling,
                },
                "endpoints": [
                    {"url": ep.url, "method": ep.method, "status": ep.status_code,
                     "content_type": ep.content_type, "interesting": ep.interesting,
                     "notes": ep.notes}
                    for ep in recon.endpoints[:100]
                ],
                "api_endpoints": [
                    {"url": ep.url, "method": ep.method, "status": ep.status_code,
                     "auth_required": ep.auth_required, "parameters": ep.parameters}
                    for ep in recon.api_endpoints
                ],
                "hidden_files": [
                    {"url": ep.url, "status": ep.status_code, "notes": ep.notes}
                    for ep in recon.hidden_files
                ],
                "forms": recon.forms[:50],
                "emails": recon.emails[:20],
                "stats": {
                    "total_endpoints": all_eps,
                    "api_endpoints": len(recon.api_endpoints),
                    "hidden_files": hidden_count,
                    "forms_found": len(recon.forms),
                    "technologies": tech_count,
                },
            },
            success=True,
        )

    # ------------------------------------------------------------------
    # Fingerprinting
    # ------------------------------------------------------------------

    async def _fingerprint(
        self, session: aiohttp.ClientSession, base_url: str,
    ) -> WebFingerprint:
        fp = WebFingerprint()
        try:
            async with session.get(base_url, allow_redirects=True) as resp:
                fp.headers = {k: v for k, v in resp.headers.items()}
                fp.server = resp.headers.get("Server", "")
                fp.cookies = {k: v.value for k, v in resp.cookies.items()}

                powered_by = resp.headers.get("X-Powered-By", "")
                if powered_by:
                    fp.framework = powered_by
                    fp.technologies.append(powered_by)

                if fp.server:
                    fp.technologies.append(fp.server)

                for hdr in SECURITY_HEADERS:
                    if hdr.lower() not in {k.lower() for k in resp.headers}:
                        fp.security_headers_missing.append(hdr)

                body = await resp.text()
                self._detect_technologies(body, fp)

        except Exception as exc:
            logger.warning("[web_recon] fingerprint failed: {}", exc)

        # Probe error handling
        try:
            async with session.get(f"{base_url}/this-does-not-exist-404-test") as resp:
                if resp.status != 404:
                    fp.error_handling = f"non-404 response for missing page: {resp.status}"
                else:
                    err_body = await resp.text()
                    if "stack" in err_body.lower() or "at " in err_body:
                        fp.error_handling = "stack_trace_exposed"
                        fp.technologies.append("verbose_errors")
        except Exception:
            pass

        logger.info(
            "[web_recon] fingerprint: server={} framework={} techs={}",
            fp.server, fp.framework, len(fp.technologies),
        )
        return fp

    @staticmethod
    def _detect_technologies(body: str, fp: WebFingerprint) -> None:
        lower = body.lower()

        js_fw_markers = {
            "angular": ["ng-app", "angular.js", "angular.min.js", "ng-controller"],
            "react": ["react.js", "react.min.js", "react-dom", "_reactRoot"],
            "vue": ["vue.js", "vue.min.js", "v-bind", "v-model"],
            "jquery": ["jquery.js", "jquery.min.js", "jquery-"],
            "express": ["express", "x-powered-by: express"],
            "node.js": ["node.js"],
        }
        for fw, markers in js_fw_markers.items():
            if any(m.lower() in lower for m in markers):
                fp.js_frameworks.append(fw)
                if fw not in fp.technologies:
                    fp.technologies.append(fw)

        if "juice shop" in lower or "owasp juice shop" in lower:
            fp.technologies.append("OWASP Juice Shop")
        if "swagger" in lower or "openapi" in lower:
            fp.technologies.append("Swagger/OpenAPI")
        if "socket.io" in lower:
            fp.technologies.append("Socket.IO")
        if "matomo" in lower or "piwik" in lower:
            fp.technologies.append("Matomo Analytics")

    # ------------------------------------------------------------------
    # Path discovery
    # ------------------------------------------------------------------

    async def _discover_paths(
        self, session: aiohttp.ClientSession, base_url: str,
    ) -> List[WebEndpoint]:
        sem = asyncio.Semaphore(self._concurrency)
        results: List[WebEndpoint] = []

        async def _probe(path: str, description: str) -> Optional[WebEndpoint]:
            async with sem:
                url = f"{base_url}{path}"
                try:
                    async with session.get(url, allow_redirects=False) as resp:
                        ct = resp.headers.get("Content-Type", "")
                        cl = int(resp.headers.get("Content-Length", 0))
                        body_preview = ""
                        if resp.status < 400:
                            body_preview = await resp.text()

                        interesting = resp.status in (200, 301, 302, 403)
                        if resp.status == 200 and cl < 50 and "not found" in body_preview.lower():
                            interesting = False

                        ep = WebEndpoint(
                            url=url,
                            method="GET",
                            status_code=resp.status,
                            content_type=ct,
                            content_length=cl,
                            interesting=interesting,
                            notes=description if interesting else "",
                        )

                        if interesting:
                            logger.info(
                                "[web_recon] found {} -> {} ({})",
                                path, resp.status, description,
                            )
                        return ep
                except Exception:
                    return None
                finally:
                    if self._delay > 0:
                        await asyncio.sleep(self._delay)

        tasks = [_probe(path, desc) for path, desc in DISCOVERY_PATHS]
        completed = await asyncio.gather(*tasks, return_exceptions=True)

        for item in completed:
            if isinstance(item, WebEndpoint):
                results.append(item)

        return results

    # ------------------------------------------------------------------
    # Crawling (BFS)
    # ------------------------------------------------------------------

    async def _crawl(
        self, session: aiohttp.ClientSession, base_url: str,
    ) -> List[WebEndpoint]:
        visited: Set[str] = set()
        queue: List[Tuple[str, int]] = [(base_url, 0)]
        results: List[WebEndpoint] = []
        parsed_base = urlparse(base_url)

        while queue and len(visited) < self._max_pages:
            url, depth = queue.pop(0)
            if url in visited or depth > self._max_depth:
                continue
            visited.add(url)

            try:
                async with session.get(url, allow_redirects=True) as resp:
                    ct = resp.headers.get("Content-Type", "")
                    if "text/html" not in ct and "application/json" not in ct:
                        continue

                    body = await resp.text()
                    ep = WebEndpoint(
                        url=url, method="GET",
                        status_code=resp.status,
                        content_type=ct,
                        content_length=len(body),
                    )

                    if "text/html" in ct:
                        extractor = _LinkExtractor(url)
                        try:
                            extractor.feed(body)
                        except Exception:
                            pass

                        ep.forms = extractor.forms
                        ep.parameters = self._extract_params(url)

                        for link in extractor.links:
                            lp = urlparse(link)
                            if lp.netloc == parsed_base.netloc or not lp.netloc:
                                if link not in visited:
                                    queue.append((link, depth + 1))

                    results.append(ep)

            except Exception:
                continue

            if self._delay > 0:
                await asyncio.sleep(self._delay)

        logger.info("[web_recon] crawled {} pages", len(results))
        return results

    # ------------------------------------------------------------------
    # API enumeration
    # ------------------------------------------------------------------

    async def _enumerate_api(
        self, session: aiohttp.ClientSession, base_url: str,
    ) -> List[WebEndpoint]:
        api_paths = [
            "/api", "/api/Users", "/api/Products", "/api/Feedbacks",
            "/api/BasketItems", "/api/Complaints", "/api/Recycles",
            "/api/Challenges", "/api/SecurityQuestions", "/api/Deliverys",
            "/api/Quantitys", "/api/Addresss", "/api/Cards", "/api/Wallets",
            "/api/Memorys",
            "/rest/products/search?q=",
            "/rest/user/whoami",
            "/rest/basket/1",
            "/rest/basket/2",
            "/rest/basket/3",
            "/rest/admin/application-version",
            "/rest/repeat-notification",
            "/rest/continue-code",
            "/rest/languages",
            "/rest/order-history",
            "/rest/wallet/balance",
            "/rest/deluxe-membership",
            "/rest/memories",
            "/rest/chatbot/status",
            "/rest/2fa/status",
            "/rest/country-mapping",
            "/b2b/v2/orders",
        ]

        results: List[WebEndpoint] = []
        sem = asyncio.Semaphore(self._concurrency)

        async def _probe_api(path: str) -> Optional[WebEndpoint]:
            async with sem:
                url = f"{base_url}{path}"
                try:
                    async with session.get(url) as resp:
                        body = await resp.text()
                        auth_required = resp.status in (401, 403)
                        ep = WebEndpoint(
                            url=url,
                            method="GET",
                            status_code=resp.status,
                            content_type=resp.headers.get("Content-Type", ""),
                            content_length=len(body),
                            auth_required=auth_required,
                            parameters=self._extract_params(url),
                            interesting=resp.status in (200, 401, 403),
                        )
                        if resp.status == 200 and "application/json" in ep.content_type:
                            ep.notes = f"Open API endpoint ({len(body)} bytes)"
                        return ep
                except Exception:
                    return None
                finally:
                    if self._delay > 0:
                        await asyncio.sleep(self._delay)

        tasks = [_probe_api(p) for p in api_paths]
        completed = await asyncio.gather(*tasks, return_exceptions=True)

        for item in completed:
            if isinstance(item, WebEndpoint) and item.interesting:
                results.append(item)

        logger.info("[web_recon] found {} interesting API endpoints", len(results))
        return results

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_params(url: str) -> List[str]:
        parsed = urlparse(url)
        if not parsed.query:
            return []
        return [p.split("=")[0] for p in parsed.query.split("&") if "=" in p]
