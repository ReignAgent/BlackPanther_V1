"""HJB-Guided Coordinator

Orchestrates the full recon -> scan -> exploit pipeline.  After every
agent action the HJB controller re-evaluates the optimal policy so the
system balances attack intensity against detection risk in real time.

All thresholds and tunables come from ``Settings`` — nothing hardcoded.

Progress events are emitted to:
  - ProgressConsole (Rich-based CLI display)
  - WebSocket (for React Native frontend)
  - Optional callbacks for custom integrations
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from loguru import logger
from pydantic import BaseModel, Field

from blackpanther.core.access import AccessPropagation
from blackpanther.core.control import HJBController, SystemState
from blackpanther.core.knowledge import KnowledgeEvolution
from blackpanther.core.suspicion import SuspicionField
from blackpanther.settings import Settings, get_settings

from .base import AgentError, AgentResult
from .console import ProgressConsole, ScanProgressTracker, TaskStatus
from .exploit import ExploitAgent
from .interfaces import LLMProvider, ReconBackend, VulnLookup
from .memory import Experience, MemoryStore
from .recon import ReconAgent
from .scanner import ScannerAgent, Vuln
from .visualizer import Visualizer


ProgressCallback = Callable[[str, float, str], None]


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
        
        self._progress_callbacks: List[ProgressCallback] = []
        self._progress_tracker: Optional[ScanProgressTracker] = None
        self._use_rich_console = True

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

    def add_progress_callback(self, callback: ProgressCallback) -> None:
        """Add a callback for progress events.
        
        Callback signature: (phase: str, progress: float, message: str) -> None
        """
        self._progress_callbacks.append(callback)

    def set_progress_tracker(self, tracker: ScanProgressTracker) -> None:
        """Set the progress tracker for Rich console display."""
        self._progress_tracker = tracker

    def disable_rich_console(self) -> None:
        """Disable Rich console output (for API/background usage)."""
        self._use_rich_console = False

    def _emit_progress(self, phase: str, progress: float, message: str) -> None:
        """Emit progress event to all registered callbacks."""
        for callback in self._progress_callbacks:
            try:
                callback(phase, progress, message)
            except Exception as e:
                logger.warning(f"[coordinator] Progress callback error: {e}")

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    async def run(self, target: str) -> Dict[str, Any]:
        """Execute the full recon -> scan -> exploit pipeline."""
        self._episode += 1
        self._start_time = time.time()
        logger.info("=== Coordinator episode {} on target={} ===", self._episode, target)

        tracker = self._progress_tracker

        summary: Dict[str, Any] = {
            "target": target,
            "episode": self._episode,
            "hosts": [],
            "vulns": [],
            "exploits": [],
            "plots": {},
        }

        self._emit_progress("initialization", 5.0, "Initializing mathematical models...")
        if tracker:
            tracker.start_initialization()

        await asyncio.sleep(0.1)
        self._emit_progress("initialization", 10.0, "Models initialized")
        if tracker:
            tracker.complete_initialization()

        self._emit_progress("recon", 15.0, f"Running nmap scan on {target}...")
        if tracker:
            tracker.start_recon(target)
        logger.info("[phase-1] Reconnaissance")
        
        recon_result = await self._safe_execute(self.recon, target)
        hosts = recon_result.raw_data.get("hosts", [target]) if recon_result else [target]
        summary["hosts"] = hosts
        self._record(self.recon.name, target, recon_result)
        self.viz.add_event(self._elapsed(), "recon", "recon")
        self.viz.plot_all()
        
        self._emit_progress("recon", 25.0, f"Found {len(hosts)} hosts")
        if tracker:
            tracker.complete_recon(len(hosts))

        self._emit_progress("scanning", 30.0, f"Scanning {len(hosts)} hosts for vulnerabilities...")
        if tracker:
            tracker.start_scanning(len(hosts))
        logger.info("[phase-2] Vulnerability scanning ({} hosts)", len(hosts))
        
        all_vulns: List[Vuln] = []
        for i, host in enumerate(hosts):
            progress = 30.0 + (i / max(len(hosts), 1)) * 20.0
            self._emit_progress("scanning", progress, f"Scanning host {host}...")
            
            services = self._services_for_host(host, recon_result)
            scan_result = await self._safe_execute(self.scanner, host, services=services)
            if scan_result:
                all_vulns.extend(ScannerAgent.vulns_from_result(scan_result))
                self._record(self.scanner.name, host, scan_result)
        
        summary["vulns"] = [v.cve for v in all_vulns]
        self.viz.add_event(self._elapsed(), "scan", "scanner")
        self.viz.plot_all()
        
        self._emit_progress("scanning", 50.0, f"Found {len(all_vulns)} vulnerabilities")
        if tracker:
            tracker.complete_scanning(len(all_vulns))

        self._emit_progress("exploitation", 55.0, "Generating exploits with LLM...")
        logger.info("[phase-3] Exploitation ({} vulns, HJB-guided)", len(all_vulns))
        exploit_count = 0
        
        settings = get_settings()
        current_llm = settings.llm_provider

        vulns_to_exploit = sorted(all_vulns, key=lambda v: v.severity, reverse=True)
        max_exploits = min(len(vulns_to_exploit), self.cfg.max_exploits_per_run)

        for i, vuln in enumerate(vulns_to_exploit):
            if exploit_count >= self.cfg.max_exploits_per_run:
                logger.info("[phase-3] exploit cap reached ({})", self.cfg.max_exploits_per_run)
                self._emit_progress("exploitation", 85.0, f"Exploit cap reached ({self.cfg.max_exploits_per_run})")
                break

            progress = 55.0 + (i / max(max_exploits, 1)) * 30.0
            self._emit_progress("exploitation", progress, f"Generating exploit for {vuln.cve}...")
            if tracker:
                tracker.start_exploit_generation(current_llm, vuln.cve)

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
                    if tracker:
                        tracker.complete_exploit_generation(vuln.cve, True)
                else:
                    if tracker:
                        tracker.complete_exploit_generation(vuln.cve, False)
                        
                self._record(self.exploit.name, target, exploit_result)
            else:
                sleep_time = action.stealth * self.cfg.stealth_sleep_multiplier
                logger.warning(
                    "[phase-3] stealth mode — S={:.3f}, u1={:.2f}, sleeping {:.1f}s",
                    state.suspicion, action.attack_intensity, sleep_time,
                )
                self._emit_progress("exploitation", progress, f"Stealth mode - suspicion too high ({state.suspicion:.3f})")
                if tracker:
                    tracker.warning(f"Stealth mode activated for {vuln.cve}")
                await asyncio.sleep(min(sleep_time, 5.0))

            self.viz.add_event(self._elapsed(), vuln.cve, "exploit")
            self.viz.plot_all()

        self._emit_progress("hjb_evaluation", 90.0, "Evaluating HJB optimal policy...")
        if tracker:
            tracker.start_hjb_evaluation()
        
        final_state = self.get_system_state()
        summary["knowledge_final"] = final_state.knowledge
        summary["suspicion_mean"] = final_state.suspicion
        summary["access_global"] = final_state.access
        
        if tracker:
            tracker.complete_hjb_evaluation()

        summary["plots"] = {k: str(v) for k, v in self.viz.plot_all().items()}
        summary["duration"] = time.time() - self._start_time

        self._emit_progress("complete", 100.0, "Scan completed successfully")
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
    tracker = ScanProgressTracker()
    coordinator.set_progress_tracker(tracker)

    with tracker.live():
        summary = await coordinator.run(target)

    tracker.console.print_summary(summary)


async def run_with_progress(target: str, progress_callback: Optional[ProgressCallback] = None) -> Dict[str, Any]:
    """Run scan with optional progress callback (for API integration)."""
    coordinator = Coordinator.from_defaults(target)
    coordinator.disable_rich_console()
    
    if progress_callback:
        coordinator.add_progress_callback(progress_callback)
    
    return await coordinator.run(target)


if __name__ == "__main__":
    asyncio.run(_main())
