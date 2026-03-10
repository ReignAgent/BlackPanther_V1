"""HJB-Guided Coordinator

Orchestrates the full recon -> scan -> exploit pipeline.  After every
agent action the HJB controller re-evaluates the optimal policy so the
system balances attack intensity against detection risk in real time.

All thresholds and tunables come from ``Settings`` — nothing hardcoded.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger
from pydantic import BaseModel, Field

from blackpanther.core.access import AccessPropagation
from blackpanther.core.control import HJBController, SystemState
from blackpanther.core.knowledge import KnowledgeEvolution
from blackpanther.core.suspicion import SuspicionField
from blackpanther.settings import Settings, get_settings

from .base import AgentError, AgentResult
from .exploit import ExploitAgent
from .interfaces import LLMProvider, ReconBackend, VulnLookup
from .memory import Experience, MemoryStore
from .recon import ReconAgent
from .scanner import ScannerAgent, Vuln
from .visualizer import Visualizer


# ------------------------------------------------------------------
# Config (thin Pydantic wrapper that reads from Settings)
# ------------------------------------------------------------------

class CoordinatorConfig(BaseModel):
    """Coordinator-level configuration.

    When constructed with no arguments every field reads from the
    ``Settings`` singleton so that ``.env`` is the single source
    of truth.
    """
    suspicion_threshold: float = Field(default=None)  # type: ignore[assignment]
    attack_threshold: float = Field(default=None)  # type: ignore[assignment]
    stealth_sleep_multiplier: float = Field(default=None)  # type: ignore[assignment]
    max_exploits_per_run: int = Field(default=None)  # type: ignore[assignment]
    output_dir: str = Field(default=None)  # type: ignore[assignment]
    db_path: str = Field(default=None)  # type: ignore[assignment]
    nmap_timing: int = Field(default=None)  # type: ignore[assignment]
    deepseek_model: str = Field(default=None)  # type: ignore[assignment]

    def model_post_init(self, __context: Any) -> None:
        cfg = get_settings()
        if self.suspicion_threshold is None:
            self.suspicion_threshold = cfg.suspicion_threshold
        if self.attack_threshold is None:
            self.attack_threshold = cfg.attack_threshold
        if self.stealth_sleep_multiplier is None:
            self.stealth_sleep_multiplier = cfg.stealth_sleep_multiplier
        if self.max_exploits_per_run is None:
            self.max_exploits_per_run = cfg.max_exploits_per_run
        if self.output_dir is None:
            self.output_dir = cfg.output_dir
        if self.db_path is None:
            self.db_path = cfg.db_path
        if self.nmap_timing is None:
            self.nmap_timing = cfg.nmap_timing
        if self.deepseek_model is None:
            self.deepseek_model = cfg.deepseek_model


# ------------------------------------------------------------------
# Coordinator
# ------------------------------------------------------------------

class Coordinator:
    """HJB-guided penetration-testing orchestrator.

    Wires together recon, scanner, exploit agents plus the HJB
    controller, memory store, and visualizer.  Every action updates
    the coupled K/S/A ODE system and re-queries the HJB value
    function before deciding the next move.

    Accepts optional DIP backends so every external dependency is
    injectable — enabling full offline testing.
    """

    def __init__(
        self,
        config: CoordinatorConfig,
        k_model: KnowledgeEvolution,
        s_model: SuspicionField,
        a_model: AccessPropagation,
        hjb: HJBController,
        memory: MemoryStore,
        *,
        recon_backend: Optional[ReconBackend] = None,
        vuln_lookup: Optional[VulnLookup] = None,
        llm_provider: Optional[LLMProvider] = None,
    ) -> None:
        self.cfg = config
        self.k = k_model
        self.s = s_model
        self.a = a_model
        self.hjb = hjb
        self.memory = memory

        app_settings = get_settings()

        self.recon = ReconAgent(
            k_model, s_model, a_model,
            backend=recon_backend,
            settings=app_settings,
        )
        self.scanner = ScannerAgent(
            k_model, s_model, a_model,
            lookup=vuln_lookup,
            settings=app_settings,
        )
        self.exploit = ExploitAgent(
            k_model, s_model, a_model,
            llm=llm_provider,
            model=config.deepseek_model,
            settings=app_settings,
        )
        self.viz = Visualizer(
            k_model, s_model, a_model,
            hjb=hjb,
            output_dir=config.output_dir,
        )
        self._episode = 0
        self._start_time = 0.0

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_defaults(
        cls,
        target: str = "",
        config: Optional[CoordinatorConfig] = None,
    ) -> "Coordinator":
        """Build all models from ``Settings`` with sane defaults."""
        cfg = config or CoordinatorConfig()
        s = get_settings()

        k_model = KnowledgeEvolution(
            alpha=s.k_alpha, beta=s.k_beta,
            gamma=s.k_gamma, k_max=s.k_max,
        )
        k_model.reset(initial_knowledge=s.k_initial)

        s_model = SuspicionField(
            width=s.s_width, height=s.s_height,
            D=s.s_diffusion, r=s.s_reaction, delta=s.s_delta,
        )
        s_model.reset()

        a_model = AccessPropagation(eta=s.a_eta, mu=s.a_mu)
        a_model.reset()
        if target:
            a_model.add_host(target, initial_access=0.1)

        hjb = HJBController(
            grid_points=s.hjb_grid_points,
            gamma=s.hjb_gamma,
            dt=s.hjb_dt,
        )

        memory = MemoryStore(db_path=cfg.db_path)

        return cls(cfg, k_model, s_model, a_model, hjb, memory)

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    async def run(self, target: str) -> Dict[str, Any]:
        """Execute the full recon -> scan -> exploit pipeline."""
        self._episode += 1
        self._start_time = time.time()
        logger.info("=== Coordinator episode {} on target={} ===", self._episode, target)

        summary: Dict[str, Any] = {
            "target": target,
            "episode": self._episode,
            "hosts": [],
            "vulns": [],
            "exploits": [],
            "plots": {},
        }

        # Phase 1: Reconnaissance
        logger.info("[phase-1] Reconnaissance")
        recon_result = await self._safe_execute(self.recon, target)
        hosts = recon_result.raw_data.get("hosts", [target]) if recon_result else [target]
        summary["hosts"] = hosts
        self._record(self.recon.name, target, recon_result)
        self.viz.add_event(self._elapsed(), "recon", "recon")
        self.viz.plot_all()

        # Phase 2: Vulnerability scanning
        logger.info("[phase-2] Vulnerability scanning ({} hosts)", len(hosts))
        all_vulns: List[Vuln] = []
        for host in hosts:
            services = self._services_for_host(host, recon_result)
            scan_result = await self._safe_execute(self.scanner, host, services=services)
            if scan_result:
                all_vulns.extend(ScannerAgent.vulns_from_result(scan_result))
                self._record(self.scanner.name, host, scan_result)
        summary["vulns"] = [v.cve for v in all_vulns]
        self.viz.add_event(self._elapsed(), "scan", "scanner")
        self.viz.plot_all()

        # Phase 3: Exploitation (HJB-guided)
        logger.info("[phase-3] Exploitation ({} vulns, HJB-guided)", len(all_vulns))
        exploit_count = 0

        for vuln in sorted(all_vulns, key=lambda v: v.severity, reverse=True):
            if exploit_count >= self.cfg.max_exploits_per_run:
                logger.info("[phase-3] exploit cap reached ({})", self.cfg.max_exploits_per_run)
                break

            state = self.get_system_state()
            action = self.hjb.get_optimal_action(state.knowledge, state.suspicion, state.access)

            if state.suspicion < self.cfg.suspicion_threshold and action.attack_intensity > self.cfg.attack_threshold:
                logger.info(
                    "[phase-3] exploiting {} (sev={:.1f}, S={:.3f}, u1={:.2f})",
                    vuln.cve, vuln.severity, state.suspicion, action.attack_intensity,
                )
                exploit_result = await self._safe_execute(self.exploit, target, vuln=vuln)
                if exploit_result and exploit_result.success:
                    summary["exploits"].append(vuln.cve)
                    exploit_count += 1
                self._record(self.exploit.name, target, exploit_result)
            else:
                sleep_time = action.stealth * self.cfg.stealth_sleep_multiplier
                logger.warning(
                    "[phase-3] stealth mode — S={:.3f}, u1={:.2f}, sleeping {:.1f}s",
                    state.suspicion, action.attack_intensity, sleep_time,
                )
                await asyncio.sleep(min(sleep_time, 5.0))

            self.viz.add_event(self._elapsed(), vuln.cve, "exploit")
            self.viz.plot_all()

        summary["plots"] = {k: str(v) for k, v in self.viz.plot_all().items()}
        summary["duration"] = time.time() - self._start_time

        logger.info(
            "=== Episode {} complete in {:.1f}s  hosts={} vulns={} exploits={} ===",
            self._episode, summary["duration"],
            len(summary["hosts"]), len(summary["vulns"]), len(summary["exploits"]),
        )
        return summary

    # ------------------------------------------------------------------
    # State query
    # ------------------------------------------------------------------

    def get_system_state(self) -> SystemState:
        return SystemState(
            knowledge=self.k.knowledge,
            suspicion=float(np.mean(self.s.field)),
            access=self.a.global_access,
            time=self._elapsed(),
            episode=self._episode,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _safe_execute(self, agent: Any, target: str, **kwargs: Any) -> Optional[AgentResult]:
        try:
            return await agent.execute(target, **kwargs)
        except AgentError as exc:
            logger.error("[coordinator] {} failed: {}", agent.name, exc)
            return None

    def _record(self, agent_name: str, target: str, result: Optional[AgentResult]) -> None:
        if result is None:
            return
        state = self.get_system_state()
        self.memory.add(Experience(
            agent_name=agent_name,
            target=target,
            k_gain=result.k_gain,
            s_inc=result.s_inc,
            a_delta=result.a_delta,
            success=result.success,
            knowledge=state.knowledge,
            suspicion_mean=state.suspicion,
            access_global=state.access,
            episode=self._episode,
            timestamp=result.timestamp,
            duration=result.duration,
            raw_data=result.raw_data,
        ))

    def _elapsed(self) -> float:
        return time.time() - self._start_time if self._start_time else 0.0

    @staticmethod
    def _services_for_host(host: str, recon_result: Optional[AgentResult]) -> Dict[int, str]:
        if recon_result is None:
            return {}
        scan = recon_result.raw_data.get("scan", {})
        return scan.get(host, {}).get("services", {})


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

async def _main() -> None:
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    coordinator = Coordinator.from_defaults(target)
    summary = await coordinator.run(target)

    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title=f"BlackPanther Run Summary — {target}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Hosts discovered", str(len(summary["hosts"])))
    table.add_row("Vulnerabilities", str(len(summary["vulns"])))
    table.add_row("Exploits generated", str(len(summary["exploits"])))
    table.add_row("Duration", f"{summary['duration']:.1f}s")
    table.add_row("Plots", ", ".join(summary["plots"].keys()))
    console.print(table)


if __name__ == "__main__":
    asyncio.run(_main())
