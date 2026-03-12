"""Tests for the BlackPanther agent layer.

Covers:
  - BaseAgent math-model integration + validation + error hierarchy
  - ModelUpdater isolation
  - MemoryStore CRUD + analytics
  - ReconAgent with mocked nmap backend (DIP)
  - ScannerAgent with mocked VulnLookup (DIP)
  - ExploitAgent with mocked LLMProvider (DIP)
  - Visualizer PNG generation
  - Coordinator end-to-end pipeline
  - Settings singleton
  - Resilience utilities (retry, validation)
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import pytest_asyncio

from blackpanther.core.access import AccessPropagation
from blackpanther.core.control import HJBController
from blackpanther.core.knowledge import KnowledgeEvolution
from blackpanther.core.suspicion import SuspicionField
from blackpanther.settings import Settings

from blackpanther.agents.base import (
    AgentError,
    AgentExecutionError,
    AgentResult,
    AgentValidationError,
    BaseAgent,
    ModelUpdater,
)
from blackpanther.agents.memory import Experience, MemoryStore
from blackpanther.agents.recon import ReconAgent, NmapBackend, SocketBackend
from blackpanther.agents.scanner import ScannerAgent, StaticVulnLookup, Vuln
from blackpanther.agents.exploit import ExploitAgent, StubLLMProvider
from blackpanther.agents.visualizer import Visualizer
from blackpanther.agents.coordinator import Coordinator, CoordinatorConfig
from blackpanther.agents.interfaces import LLMProvider, ReconBackend, VulnLookup
from blackpanther.agents.resilience import async_retry, validate_target


# ===================================================================
# Test-only Settings (no .env needed)
# ===================================================================

def _test_settings(**overrides: Any) -> Settings:
    defaults = dict(
        deepseek_api_key="",
        db_path=":memory:",
        output_dir="/tmp/bp-test-proofs",
        exploit_sandbox_dir="/tmp/bp-test-exploits",
        log_level="WARNING",
    )
    defaults.update(overrides)
    return Settings(**defaults)


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def k_model():
    m = KnowledgeEvolution(alpha=0.1, beta=0.01, gamma=0.05)
    m.reset(initial_knowledge=1.0)
    return m


@pytest.fixture
def s_model():
    m = SuspicionField(width=20, height=20, D=0.1, r=0.05, delta=0.01)
    m.reset()
    return m


@pytest.fixture
def a_model():
    m = AccessPropagation(eta=0.2, mu=0.01)
    m.reset()
    m.add_host("192.168.1.1", initial_access=0.1)
    m.add_host("192.168.1.2", initial_access=0.0)
    return m


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def settings():
    return _test_settings()


# ===================================================================
# Resilience utilities
# ===================================================================

class TestValidateTarget:
    def test_valid_ip(self):
        assert validate_target("192.168.1.1") == "192.168.1.1"

    def test_valid_cidr(self):
        assert validate_target("10.0.0.0/24") == "10.0.0.0/24"

    def test_valid_hostname(self):
        assert validate_target("web-server.local") == "web-server.local"

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            validate_target("")

    def test_injection_raises(self):
        with pytest.raises(ValueError):
            validate_target("192.168.1.1; rm -rf /")

    def test_strips_whitespace(self):
        assert validate_target("  10.0.0.1  ") == "10.0.0.1"


class TestAsyncRetry:
    @pytest.mark.asyncio
    async def test_succeeds_on_first_try(self):
        call_count = 0

        @async_retry(max_attempts=3, backoff=0.01)
        async def ok():
            nonlocal call_count
            call_count += 1
            return "done"

        assert await ok() == "done"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_failure(self):
        call_count = 0

        @async_retry(max_attempts=3, backoff=0.01)
        async def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("transient")
            return "recovered"

        assert await flaky() == "recovered"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_raises_after_exhaustion(self):
        @async_retry(max_attempts=2, backoff=0.01)
        async def always_fail():
            raise ValueError("permanent")

        with pytest.raises(ValueError, match="permanent"):
            await always_fail()


# ===================================================================
# BaseAgent
# ===================================================================

class _StubAgent(BaseAgent):
    name = "stub"

    def __init__(self, *args: Any, result: AgentResult | None = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._result = result or AgentResult(k_gain=1.0, s_inc=0.5, a_delta=0.2)

    async def _execute(self, target: str, **kwargs: Any) -> AgentResult:
        return self._result


class TestBaseAgent:
    @pytest.mark.asyncio
    async def test_execute_updates_models(self, k_model, s_model, a_model):
        agent = _StubAgent(k_model, s_model, a_model)
        k_before = k_model.knowledge
        result = await agent.execute("192.168.1.1")
        assert result.success is True
        assert result.duration > 0
        assert k_model.knowledge != k_before
        assert len(k_model.history) >= 2

    @pytest.mark.asyncio
    async def test_execution_error_hierarchy(self, k_model, s_model, a_model):
        class _Failing(BaseAgent):
            name = "fail"
            async def _execute(self, target, **kw):
                raise RuntimeError("boom")

        agent = _Failing(k_model, s_model, a_model)
        with pytest.raises(AgentExecutionError, match="boom"):
            await agent.execute("192.168.1.1")

    @pytest.mark.asyncio
    async def test_validation_error_on_bad_target(self, k_model, s_model, a_model):
        agent = _StubAgent(k_model, s_model, a_model)
        with pytest.raises(AgentValidationError):
            await agent.execute("")

    @pytest.mark.asyncio
    async def test_state_snapshot(self, k_model, s_model, a_model):
        agent = _StubAgent(k_model, s_model, a_model)
        snap = agent.get_state_snapshot()
        assert "knowledge" in snap
        assert "suspicion_mean" in snap
        assert "access_global" in snap

    @pytest.mark.asyncio
    async def test_action_count_increments(self, k_model, s_model, a_model):
        agent = _StubAgent(k_model, s_model, a_model)
        await agent.execute("192.168.1.1")
        await agent.execute("192.168.1.2")
        assert agent.action_count == 2


class TestModelUpdater:
    def test_update_survives_model_failure(self, k_model, s_model, a_model):
        updater = ModelUpdater(k_model, s_model, a_model)
        result = AgentResult(k_gain=1.0, s_inc=0.5, a_delta=0.2)
        with patch.object(k_model, "step", side_effect=RuntimeError("K exploded")):
            updater.update("192.168.1.1", result)
        assert len(s_model.history) >= 1


# ===================================================================
# MemoryStore
# ===================================================================

class TestMemoryStore:
    def test_add_and_retrieve(self):
        store = MemoryStore(db_path=":memory:")
        exp = Experience(agent_name="recon", target="10.0.0.1", k_gain=1.5)
        row_id = store.add(exp)
        assert row_id >= 1
        assert store.count == 1

    def test_get_recent(self):
        store = MemoryStore(db_path=":memory:")
        for i in range(5):
            store.add(Experience(agent_name="a", target=f"t{i}", k_gain=float(i)))
        recent = store.get_recent(3)
        assert len(recent) == 3
        assert recent[-1].k_gain == 4.0

    def test_get_by_target(self):
        store = MemoryStore(db_path=":memory:")
        store.add(Experience(agent_name="a", target="host-a"))
        store.add(Experience(agent_name="b", target="host-b"))
        store.add(Experience(agent_name="c", target="host-a"))
        assert len(store.get_by_target("host-a")) == 2

    def test_query_filters(self):
        store = MemoryStore(db_path=":memory:")
        store.add(Experience(agent_name="recon", target="t", k_gain=0.1, success=True))
        store.add(Experience(agent_name="recon", target="t", k_gain=2.0, success=True))
        store.add(Experience(agent_name="exploit", target="t", k_gain=3.0, success=False))
        results = store.query(agent_name="recon", min_k_gain=1.0)
        assert len(results) == 1
        assert results[0].k_gain == 2.0

    def test_success_rate(self):
        store = MemoryStore(db_path=":memory:")
        store.add(Experience(agent_name="a", target="t", success=True))
        store.add(Experience(agent_name="a", target="t", success=True))
        store.add(Experience(agent_name="a", target="t", success=False))
        assert abs(store.success_rate() - 2 / 3) < 0.01

    def test_to_dataframe(self):
        store = MemoryStore(db_path=":memory:")
        store.add(Experience(agent_name="a", target="t"))
        df = store.to_dataframe()
        assert len(df) == 1
        assert "agent_name" in df.columns

    def test_suspicion_trend(self):
        store = MemoryStore(db_path=":memory:")
        for i in range(10):
            store.add(Experience(agent_name="a", target="t", suspicion_mean=i * 0.1))
        trend = store.suspicion_trend(5)
        assert len(trend) == 5
        assert trend[-1] == pytest.approx(0.9)


# ===================================================================
# ReconAgent (DIP: inject mock ReconBackend)
# ===================================================================

class _MockReconBackend(ReconBackend):
    async def scan(self, target: str) -> Dict[str, Dict[str, Any]]:
        return {
            "192.168.1.1": {
                "ports": [22, 80],
                "services": {22: "ssh 8.2", 80: "http 2.4.41"},
                "state": "up",
                "hostname": "192.168.1.1",
            },
            "192.168.1.2": {
                "ports": [443],
                "services": {443: "https"},
                "state": "up",
                "hostname": "192.168.1.2",
            },
        }


class TestReconAgent:
    @pytest.mark.asyncio
    async def test_scan_via_injected_backend(self, k_model, s_model, a_model, settings):
        agent = ReconAgent(
            k_model, s_model, a_model,
            backend=_MockReconBackend(),
            settings=settings,
        )
        result = await agent.execute("192.168.1.0/24")
        assert result.success
        assert result.k_gain > 0
        assert "192.168.1.1" in result.raw_data["hosts"]

    @pytest.mark.asyncio
    async def test_incremental_discovery(self, k_model, s_model, a_model, settings):
        agent = ReconAgent(
            k_model, s_model, a_model,
            backend=_MockReconBackend(),
            settings=settings,
        )
        r1 = await agent.execute("192.168.1.0/24")
        r2 = await agent.execute("192.168.1.0/24")
        assert r2.k_gain < r1.k_gain


# ===================================================================
# ScannerAgent (DIP: inject mock VulnLookup)
# ===================================================================

class _MockVulnLookup(VulnLookup):
    async def search(self, service_str: str, port: int) -> List[Vuln]:
        return [Vuln(cve="CVE-2024-0001", severity=9.0, service=service_str.split()[0], port=port, title="Mock Vuln")]


class TestScannerAgent:
    @pytest.mark.asyncio
    async def test_lookup_via_injected_backend(self, k_model, s_model, a_model, settings):
        agent = ScannerAgent(
            k_model, s_model, a_model,
            lookup=_MockVulnLookup(),
            settings=settings,
        )
        result = await agent.execute("192.168.1.1", services={80: "apache 2.4.49"})
        assert result.success
        vulns = ScannerAgent.vulns_from_result(result)
        assert len(vulns) == 1
        assert vulns[0].severity == 9.0

    @pytest.mark.asyncio
    async def test_static_fallback(self, k_model, s_model, a_model, settings):
        agent = ScannerAgent(
            k_model, s_model, a_model,
            lookup=StaticVulnLookup(),
            settings=settings,
        )
        result = await agent.execute("10.0.0.1", services={22: "ssh 8.2", 80: "http"})
        vulns = ScannerAgent.vulns_from_result(result)
        assert len(vulns) >= 2
        assert any(v.service == "ssh" for v in vulns)


# ===================================================================
# ExploitAgent (DIP: inject mock LLMProvider)
# ===================================================================

class _MockLLM(LLMProvider):
    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        return "import socket\nprint('exploit generated')"


class TestExploitAgent:
    @pytest.mark.asyncio
    async def test_with_injected_llm(self, k_model, s_model, a_model, tmp_dir, settings):
        agent = ExploitAgent(
            k_model, s_model, a_model,
            llm=_MockLLM(),
            sandbox_dir=tmp_dir,
            settings=settings,
        )
        vuln = Vuln(cve="CVE-2021-44228", severity=10.0, service="log4j", version="2.14.1", port=8080, title="Log4Shell")
        result = await agent.execute("10.0.0.1", vuln=vuln)
        assert result.success
        assert result.k_gain == 0.5
        assert Path(result.raw_data["script_path"]).exists()

    @pytest.mark.asyncio
    async def test_stub_provider(self, k_model, s_model, a_model, tmp_dir, settings):
        agent = ExploitAgent(
            k_model, s_model, a_model,
            llm=StubLLMProvider(),
            sandbox_dir=tmp_dir,
            settings=settings,
        )
        vuln = Vuln(cve="CVE-2023-0001", severity=8.0, service="nginx", port=443)
        result = await agent.execute("10.0.0.1", vuln=vuln)
        assert result.success
        assert "socket" in Path(result.raw_data["script_path"]).read_text()

    @pytest.mark.asyncio
    async def test_missing_vuln_kwarg(self, k_model, s_model, a_model, tmp_dir, settings):
        agent = ExploitAgent(k_model, s_model, a_model, sandbox_dir=tmp_dir, settings=settings)
        with pytest.raises(AgentError, match="vuln"):
            await agent.execute("10.0.0.1")

    def test_aggressiveness_scoring(self):
        from blackpanther.agents.exploit import _estimate_aggressiveness
        assert _estimate_aggressiveness("import os; os.system('ls')") > 0
        assert _estimate_aggressiveness("reverse_shell()") > 0.3
        assert _estimate_aggressiveness("print('hello')") == 0.0

    def test_strip_fences(self):
        from blackpanther.agents.exploit import _strip_fences
        assert _strip_fences("```python\nprint('hi')\n```") == "print('hi')"


# ===================================================================
# Visualizer
# ===================================================================

class TestVisualizer:
    def test_plot_knowledge_curve(self, k_model, s_model, a_model, tmp_dir):
        for _ in range(20):
            k_model.step(suspicion=0.1, learning_action=0.5)
        viz = Visualizer(k_model, s_model, a_model, output_dir=tmp_dir)
        assert viz.plot_knowledge_curve().exists()

    def test_plot_suspicion_heatmap(self, k_model, s_model, a_model, tmp_dir):
        for _ in range(5):
            s_model.step([(0.5, 0.5, 0.8)], knowledge=10, access=0.3)
        viz = Visualizer(k_model, s_model, a_model, output_dir=tmp_dir)
        path = viz.plot_suspicion_heatmap()
        assert path.exists()
        assert path.stat().st_size > 1000

    def test_plot_access_bars(self, k_model, s_model, a_model, tmp_dir):
        viz = Visualizer(k_model, s_model, a_model, output_dir=tmp_dir)
        assert viz.plot_access_bars().exists()

    def test_plot_hjb_policy_analytical(self, k_model, s_model, a_model, tmp_dir):
        viz = Visualizer(k_model, s_model, a_model, hjb=None, output_dir=tmp_dir)
        assert viz.plot_hjb_policy().exists()

    def test_plot_all(self, k_model, s_model, a_model, tmp_dir):
        for _ in range(10):
            k_model.step(suspicion=0.1)
        viz = Visualizer(k_model, s_model, a_model, output_dir=tmp_dir)
        paths = viz.plot_all()
        assert len(paths) == 4
        for p in paths.values():
            assert p.exists()


# ===================================================================
# Coordinator
# ===================================================================

class TestCoordinator:
    @pytest.mark.asyncio
    async def test_full_pipeline(self, tmp_dir):
        cfg = CoordinatorConfig(
            output_dir=os.path.join(tmp_dir, "proofs"),
            db_path=":memory:",
            max_exploits_per_run=2,
        )
        coord = Coordinator.from_defaults("127.0.0.1", config=cfg)

        recon_result = AgentResult(
            k_gain=0.4, s_inc=0.1,
            raw_data={
                "hosts": ["127.0.0.1"],
                "new_hosts": ["127.0.0.1"],
                "new_port_count": 2,
                "scan": {"127.0.0.1": {"ports": [22, 80], "services": {22: "ssh 8.2", 80: "http"}, "state": "up", "hostname": "127.0.0.1"}},
            },
            success=True,
        )
        scan_result = AgentResult(
            k_gain=0.6, s_inc=0.1,
            raw_data={
                "vulns": [{"cve": "CVE-2023-0001", "severity": 8.0, "service": "ssh", "version": "8.2", "port": 22, "exploit_path": "", "title": "Test"}],
                "total": 1, "target": "127.0.0.1",
            },
            success=True,
        )
        exploit_result = AgentResult(
            k_gain=0.5, s_inc=0.3, a_delta=0.3,
            raw_data={"vuln_cve": "CVE-2023-0001", "script_path": "/tmp/test.py"},
            success=True,
        )

        with patch.object(coord.recon, "execute", new_callable=AsyncMock, return_value=recon_result), \
             patch.object(coord.scanner, "execute", new_callable=AsyncMock, return_value=scan_result), \
             patch.object(coord.exploit, "execute", new_callable=AsyncMock, return_value=exploit_result):
            summary = await coord.run("127.0.0.1")

        assert summary["episode"] == 1
        assert "127.0.0.1" in summary["hosts"]
        assert len(summary["vulns"]) >= 1

    @pytest.mark.asyncio
    async def test_stealth_mode(self, tmp_dir):
        cfg = CoordinatorConfig(
            output_dir=os.path.join(tmp_dir, "proofs"),
            db_path=":memory:",
            suspicion_threshold=0.0,
            stealth_sleep_multiplier=0.01,
        )
        coord = Coordinator.from_defaults("127.0.0.1", config=cfg)

        recon_result = AgentResult(
            k_gain=0.1, s_inc=0.1,
            raw_data={"hosts": ["127.0.0.1"], "new_hosts": [], "new_port_count": 0, "scan": {}},
            success=True,
        )
        scan_result = AgentResult(
            k_gain=0.3, s_inc=0.1,
            raw_data={"vulns": [{"cve": "CVE-X", "severity": 9.0, "service": "http", "version": "", "port": 80, "exploit_path": "", "title": "X"}], "total": 1, "target": "127.0.0.1"},
            success=True,
        )

        with patch.object(coord.recon, "execute", new_callable=AsyncMock, return_value=recon_result), \
             patch.object(coord.scanner, "execute", new_callable=AsyncMock, return_value=scan_result), \
             patch.object(coord.exploit, "execute", new_callable=AsyncMock):
            summary = await coord.run("127.0.0.1")

        assert len(summary["exploits"]) == 0

    def test_get_system_state(self, tmp_dir):
        cfg = CoordinatorConfig(output_dir=tmp_dir, db_path=":memory:")
        coord = Coordinator.from_defaults("127.0.0.1", config=cfg)
        state = coord.get_system_state()
        assert state.knowledge > 0
        assert 0 <= state.suspicion <= 1


# ===================================================================
# Settings
# ===================================================================

class TestSettings:
    def test_defaults_load(self):
        s = _test_settings()
        assert s.nmap_timing == 3
        assert s.max_retries == 3
        assert s.deepseek_api_key == ""

    def test_invalid_log_level(self):
        with pytest.raises(Exception):
            _test_settings(log_level="INVALID")

    def test_override(self):
        s = _test_settings(nmap_timing=5, max_retries=1)
        assert s.nmap_timing == 5
        assert s.max_retries == 1
