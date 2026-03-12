"""Reconnaissance Agent

Wraps python-nmap for host discovery and SYN port scanning.
Falls back to the built-in ``NetworkScanner`` (pure-socket) when nmap
is not available (e.g. outside Docker).

Every scan updates the coupled K / S / A models:
  k_gain  = new_hosts * 0.2  +  new_ports * 0.05
  s_inc   = timing_intensity * total_probes / 500

DIP: accepts an optional ``ReconBackend`` so the nmap dependency
can be swapped at construction time without modifying this class.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from loguru import logger

from blackpanther.core.access import AccessPropagation
from blackpanther.core.knowledge import KnowledgeEvolution
from blackpanther.core.scanner import NetworkScanner, ScanResult
from blackpanther.core.suspicion import SuspicionField
from blackpanther.settings import Settings, get_settings

from .base import AgentResult, BaseAgent
from .interfaces import ReconBackend
from .resilience import async_retry, is_tool_available

TIMING_INTENSITY: Dict[int, float] = {
    0: 0.05, 1: 0.1, 2: 0.2, 3: 0.4, 4: 0.7, 5: 1.0,
}

K_GAIN_PER_HOST = 0.2
K_GAIN_PER_PORT = 0.05
S_INC_PROBE_DIVISOR = 500.0


# ------------------------------------------------------------------
# Concrete ReconBackend: nmap
# ------------------------------------------------------------------

class NmapBackend(ReconBackend):
    """nmap-based scanner (runs in a thread pool)."""

    def __init__(self, timing: int = 3, extra_args: str = "-sS") -> None:
        self._timing = timing
        self._extra_args = extra_args

    @async_retry(max_attempts=2, backoff=2.0)
    async def scan(self, target: str) -> Dict[str, Dict[str, Any]]:
        import nmap

        scanner = nmap.PortScanner()
        args = f"{self._extra_args} -T{self._timing}"
        logger.info("[nmap] {} {}", args, target)

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, scanner.scan, target, None, args)

        result: Dict[str, Dict[str, Any]] = {}
        for host in scanner.all_hosts():
            ports: List[int] = []
            services: Dict[int, str] = {}
            for proto in scanner[host].all_protocols():
                for port in sorted(scanner[host][proto]):
                    info = scanner[host][proto][port]
                    if info.get("state") == "open":
                        ports.append(port)
                        svc = info.get("name", "unknown")
                        ver = info.get("version", "")
                        services[port] = f"{svc} {ver}".strip()
            result[host] = {
                "ports": ports,
                "services": services,
                "state": scanner[host].state(),
                "hostname": scanner[host].hostname(),
            }
        return result


# ------------------------------------------------------------------
# Concrete ReconBackend: socket fallback
# ------------------------------------------------------------------

class SocketBackend(ReconBackend):
    """Pure-socket fallback using the built-in ``NetworkScanner``."""

    def __init__(self, timeout: float = 1.0, max_threads: int = 50) -> None:
        self._scanner = NetworkScanner(timeout=timeout, max_threads=max_threads)

    async def scan(self, target: str) -> Dict[str, Dict[str, Any]]:
        loop = asyncio.get_running_loop()
        if "/" in target:
            alive = await loop.run_in_executor(
                None, self._scanner.discover_hosts, target,
            )
        else:
            alive = [target]

        result: Dict[str, Dict[str, Any]] = {}
        for host in alive:
            sr: ScanResult = await loop.run_in_executor(
                None, self._scanner.scan_ports, host,
            )
            result[host] = {
                "ports": [p.port for p in sr.open_ports],
                "services": {p.port: p.service for p in sr.open_ports},
                "state": "up" if sr.alive else "down",
                "hostname": host,
            }
        return result

    @property
    def inner(self) -> NetworkScanner:
        return self._scanner


# ------------------------------------------------------------------
# ReconAgent
# ------------------------------------------------------------------

class ReconAgent(BaseAgent):
    """Network reconnaissance via an injected ``ReconBackend``.

    When no backend is provided the agent auto-selects nmap (if
    installed) or falls back to a socket-based scanner.

    Args:
        k_model, s_model, a_model: Shared math-model instances.
        backend: Explicit ``ReconBackend`` (DIP).
        settings: Override settings (defaults to global singleton).
    """

    name = "recon"

    def __init__(
        self,
        k_model: KnowledgeEvolution,
        s_model: SuspicionField,
        a_model: AccessPropagation,
        backend: Optional[ReconBackend] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        super().__init__(k_model, s_model, a_model)
        cfg = settings or get_settings()
        self._timing = cfg.nmap_timing
        self._known_hosts: set[str] = set()
        self._known_ports: Dict[str, set[int]] = {}

        if backend is not None:
            self._backend = backend
        elif is_tool_available("nmap"):
            self._backend = NmapBackend(
                timing=cfg.nmap_timing,
                extra_args=cfg.nmap_extra_args,
            )
        else:
            logger.warning("nmap not found — using socket fallback")
            self._backend = SocketBackend(
                timeout=cfg.scanner_timeout,
                max_threads=cfg.scanner_max_threads,
            )

        self._socket_fallback = SocketBackend(
            timeout=cfg.scanner_timeout,
            max_threads=cfg.scanner_max_threads,
        )

    # ------------------------------------------------------------------
    # Core execution
    # ------------------------------------------------------------------

    async def _execute(self, target: str, **kwargs: Any) -> AgentResult:
        scan_data = await self._backend.scan(target)

        new_hosts = [h for h in scan_data if h not in self._known_hosts]
        new_port_count = 0
        for host, info in scan_data.items():
            prev = self._known_ports.get(host, set())
            new_ports = set(info.get("ports", [])) - prev
            new_port_count += len(new_ports)
            self._known_ports.setdefault(host, set()).update(info.get("ports", []))
        self._known_hosts.update(scan_data.keys())

        k_gain = len(new_hosts) * K_GAIN_PER_HOST + new_port_count * K_GAIN_PER_PORT
        total_probes = sum(len(v.get("ports", [])) for v in scan_data.values()) + len(scan_data)
        s_inc = TIMING_INTENSITY.get(self._timing, 0.4) * total_probes / S_INC_PROBE_DIVISOR

        self._populate_access_graph(scan_data)

        return AgentResult(
            k_gain=k_gain,
            s_inc=s_inc,
            a_delta=0.0,
            raw_data={
                "hosts": list(scan_data.keys()),
                "new_hosts": new_hosts,
                "new_port_count": new_port_count,
                "scan": {h: dict(v) for h, v in scan_data.items()},
            },
            success=bool(scan_data),
        )

    # ------------------------------------------------------------------
    # Wire discoveries into the AccessPropagation graph
    # ------------------------------------------------------------------

    def _populate_access_graph(self, scan_data: Dict[str, Dict[str, Any]]) -> None:
        for host, info in scan_data.items():
            services = list(info.get("services", {}).values())
            if host not in self.a_model.hosts:
                self.a_model.add_host(host, initial_access=0.0, services=services)

        graph = self._socket_fallback.inner.build_network_graph()
        if graph.number_of_nodes() > 0:
            self.a_model.set_network(graph)
