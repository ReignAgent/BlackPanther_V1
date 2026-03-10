"""Vulnerability Scanner Agent

Runs ``searchsploit`` against discovered services, parses CVE
identifiers, and scores severity.  Since searchsploit is a local
database lookup it generates minimal network noise.

DIP: accepts a ``VulnLookup`` backend so searchsploit can be
swapped for any CVE database without modifying this class.

Math deltas:
  k_gain = len(vulns) * 0.3   (high knowledge value)
  s_inc  = 0.1                 (local lookup, very low noise)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from loguru import logger

from blackpanther.core.access import AccessPropagation
from blackpanther.core.knowledge import KnowledgeEvolution
from blackpanther.core.suspicion import SuspicionField
from blackpanther.settings import Settings, get_settings

from .base import AgentResult, BaseAgent
from .interfaces import VulnLookup
from .resilience import is_tool_available, run_subprocess

CVE_RE = re.compile(r"CVE-\d{4}-\d{4,7}", re.IGNORECASE)

SEVERITY_BY_TYPE: Dict[str, float] = {
    "remote": 9.0, "webapps": 8.5, "local": 6.0,
    "dos": 5.0, "shellcode": 7.5,
}

K_GAIN_PER_VULN = 0.3
S_INC_LOCAL_LOOKUP = 0.1


# ------------------------------------------------------------------
# Data transfer object
# ------------------------------------------------------------------

@dataclass
class Vuln:
    """A single discovered vulnerability."""
    cve: str
    severity: float
    service: str
    version: str = ""
    port: int = 0
    exploit_path: str = ""
    title: str = ""


# ------------------------------------------------------------------
# Concrete VulnLookup: searchsploit
# ------------------------------------------------------------------

class SearchsploitLookup(VulnLookup):
    """Query the local ExploitDB via ``searchsploit --json``."""

    def __init__(self, timeout: int = 30) -> None:
        self._timeout = timeout

    async def search(self, service_str: str, port: int) -> List[Vuln]:
        returncode, stdout, stderr = await run_subprocess(
            ["searchsploit", "--json", service_str],
            timeout=self._timeout,
        )
        if returncode != 0:
            if returncode == -1:
                logger.warning("[searchsploit] binary not found")
            else:
                logger.warning("[searchsploit] exited {}: {}", returncode, stderr[:200])
            return []
        return _parse_searchsploit_json(stdout, service_str, port)


# ------------------------------------------------------------------
# Concrete VulnLookup: static fallback
# ------------------------------------------------------------------

_KNOWN_VULNS: Dict[str, List[Dict[str, Any]]] = {
    "ssh": [{"cve": "CVE-2023-38408", "severity": 7.5, "title": "OpenSSH Pre-Auth Double Free"}],
    "http": [{"cve": "CVE-2021-41773", "severity": 9.8, "title": "Apache Path Traversal"}],
    "https": [{"cve": "CVE-2014-0160", "severity": 7.5, "title": "OpenSSL Heartbleed"}],
    "smb": [{"cve": "CVE-2017-0144", "severity": 9.8, "title": "EternalBlue SMBv1 RCE"}],
    "mysql": [{"cve": "CVE-2012-2122", "severity": 7.5, "title": "MySQL Auth Bypass"}],
    "redis": [{"cve": "CVE-2022-0543", "severity": 10.0, "title": "Redis Lua Sandbox Escape"}],
    "ftp": [{"cve": "CVE-2015-3306", "severity": 9.8, "title": "ProFTPD mod_copy RCE"}],
    "rdp": [{"cve": "CVE-2019-0708", "severity": 9.8, "title": "BlueKeep RDP RCE"}],
    "postgresql": [{"cve": "CVE-2019-9193", "severity": 9.0, "title": "PostgreSQL COPY FROM PROGRAM"}],
    "mongodb": [{"cve": "CVE-2020-7921", "severity": 8.1, "title": "MongoDB Auth Bypass"}],
    "vnc": [{"cve": "CVE-2019-15681", "severity": 7.5, "title": "LibVNC Memory Leak"}],
}


class StaticVulnLookup(VulnLookup):
    """Hard-coded vulnerability DB for environments without searchsploit."""

    async def search(self, service_str: str, port: int) -> List[Vuln]:
        svc = service_str.split()[0].lower() if service_str else "unknown"
        version = " ".join(service_str.split()[1:]) if service_str else ""
        return [
            Vuln(cve=e["cve"], severity=e["severity"], service=svc,
                 version=version, port=port, title=e["title"])
            for e in _KNOWN_VULNS.get(svc, [])
        ]


# ------------------------------------------------------------------
# ScannerAgent
# ------------------------------------------------------------------

class ScannerAgent(BaseAgent):
    """CVE / exploit discovery via an injected ``VulnLookup``.

    Auto-selects searchsploit if installed, otherwise falls back
    to a static mapping.

    Args:
        k_model, s_model, a_model: Shared math-model instances.
        lookup: Explicit ``VulnLookup`` backend (DIP).
        settings: Override settings.
    """

    name = "scanner"

    def __init__(
        self,
        k_model: KnowledgeEvolution,
        s_model: SuspicionField,
        a_model: AccessPropagation,
        lookup: Optional[VulnLookup] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        super().__init__(k_model, s_model, a_model)
        cfg = settings or get_settings()

        if lookup is not None:
            self._lookup = lookup
        elif is_tool_available("searchsploit"):
            self._lookup = SearchsploitLookup(timeout=cfg.searchsploit_timeout)
        else:
            self._lookup = StaticVulnLookup()

    # ------------------------------------------------------------------
    # Core execution
    # ------------------------------------------------------------------

    async def _execute(self, target: str, **kwargs: Any) -> AgentResult:
        services: Dict[int, str] = kwargs.get("services", {})
        if not services:
            services = self._services_from_access(target)

        all_vulns: List[Vuln] = []
        for port, svc_str in services.items():
            vulns = await self._lookup.search(svc_str, port)
            all_vulns.extend(vulns)

        all_vulns.sort(key=lambda v: v.severity, reverse=True)

        return AgentResult(
            k_gain=len(all_vulns) * K_GAIN_PER_VULN,
            s_inc=S_INC_LOCAL_LOOKUP,
            a_delta=0.0,
            raw_data={
                "vulns": [
                    {"cve": v.cve, "severity": v.severity, "service": v.service,
                     "version": v.version, "port": v.port,
                     "exploit_path": v.exploit_path, "title": v.title}
                    for v in all_vulns
                ],
                "total": len(all_vulns),
                "target": target,
            },
            success=True,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _services_from_access(self, target: str) -> Dict[int, str]:
        host_data = self.a_model.hosts
        host = host_data.get(target)
        if host and host.services:
            return {i: svc for i, svc in enumerate(host.services)}
        return {}

    @staticmethod
    def vulns_from_result(result: AgentResult) -> List[Vuln]:
        """Reconstruct Vuln objects from an ``AgentResult.raw_data``."""
        return [Vuln(**v) for v in result.raw_data.get("vulns", [])]


# ------------------------------------------------------------------
# Shared parser (DRY — used by SearchsploitLookup)
# ------------------------------------------------------------------

def _parse_searchsploit_json(raw: str, service_str: str, port: int) -> List[Vuln]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("[scanner] bad JSON from searchsploit")
        return []

    vulns: List[Vuln] = []
    for entry in data.get("RESULTS_EXPLOIT", [])[:25]:
        title = entry.get("Title", "")
        path = entry.get("Path", "")
        etype = entry.get("Type", "").lower()

        cves = CVE_RE.findall(title + " " + path)
        cve = cves[0] if cves else f"EDB-{entry.get('EDB-ID', 'unknown')}"
        severity = SEVERITY_BY_TYPE.get(etype, 5.0)

        svc_parts = service_str.split()
        service = svc_parts[0] if svc_parts else "unknown"
        version = " ".join(svc_parts[1:]) if len(svc_parts) > 1 else ""

        vulns.append(Vuln(
            cve=cve, severity=severity, service=service,
            version=version, port=port, exploit_path=path, title=title,
        ))
    return vulns
