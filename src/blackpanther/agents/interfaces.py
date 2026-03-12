"""Dependency-Inversion interfaces (DIP).

Agents depend on these ABCs, never on concrete implementations.
Swapping nmap for masscan, or DeepSeek for OpenAI, requires zero
changes to the agent layer — only the provider wiring in the
coordinator factory.

NOTE: This module must NOT import from sibling modules (scanner,
recon, exploit) to avoid circular imports.  Return types use
generic containers or ``Any`` and the concrete modules handle
type narrowing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List


# ------------------------------------------------------------------
# LLM Provider  (used by ExploitAgent)
# ------------------------------------------------------------------

class LLMProvider(ABC):
    """Abstract LLM backend for exploit code generation."""

    @abstractmethod
    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Return raw text completion. Empty string on failure."""
        ...


# ------------------------------------------------------------------
# Recon Backend  (used by ReconAgent)
# ------------------------------------------------------------------

class ReconBackend(ABC):
    """Abstract network scanner for host/port discovery."""

    @abstractmethod
    async def scan(self, target: str) -> Dict[str, Dict[str, Any]]:
        """Return ``{host: {ports, services, state, hostname}}``."""
        ...


# ------------------------------------------------------------------
# Vuln Lookup  (used by ScannerAgent)
# ------------------------------------------------------------------

class VulnLookup(ABC):
    """Abstract vulnerability database query.

    Implementations return a list of dataclass-like objects with at
    least ``cve``, ``severity``, ``service``, ``port`` attributes.
    The concrete type is ``scanner.Vuln`` but we avoid importing it
    here to prevent circular imports.
    """

    @abstractmethod
    async def search(self, service_str: str, port: int) -> List[Any]:
        """Return known vulnerabilities for a service."""
        ...
