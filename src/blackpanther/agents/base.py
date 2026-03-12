"""Base Agent with Mathematical Model Integration

Every agent action feeds deltas into the coupled ODE/PDE system:
  K  (KnowledgeEvolution)  -- dK/dt = αK(1-K/K_max) - βK + γS + σξ
  S  (SuspicionField)      -- ∂S/∂t = D∇²S + rS(1-S) - δKA + σξ
  A  (AccessPropagation)   -- dA/dt = ηKA(1-A) - μA + Σ w_ji A_j(1-A_i)

Design principles:
  SRP  — execution, model updates, and state queries are distinct methods
  OCP  — new agents subclass without touching this file
  DIP  — agents receive model instances via constructor (injectable)
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
from loguru import logger

from blackpanther.core.knowledge import KnowledgeEvolution
from blackpanther.core.suspicion import SuspicionField
from blackpanther.core.access import AccessPropagation

from .resilience import validate_target


# ------------------------------------------------------------------
# Structured error hierarchy
# ------------------------------------------------------------------

class AgentError(Exception):
    """Base for all agent-layer errors."""


class AgentExecutionError(AgentError):
    """The agent's ``_execute`` raised an unrecoverable error."""


class AgentValidationError(AgentError):
    """Input failed validation before execution started."""


class ModelUpdateError(AgentError):
    """A math-model step failed after execution completed."""


# ------------------------------------------------------------------
# Result value object
# ------------------------------------------------------------------

@dataclass
class AgentResult:
    """Outcome of a single agent action.

    Attributes:
        k_gain:  Knowledge delta fed into KnowledgeEvolution.
        s_inc:   Suspicion increment injected into the 2-D field.
        a_delta: Access change (informational; actual update goes
                 through AccessPropagation.step).
        raw_data: Arbitrary payload returned by the concrete agent.
        success: Whether the action completed without error.
        timestamp: Wall-clock time when the result was produced.
        duration: Seconds elapsed during execution.
    """
    k_gain: float = 0.0
    s_inc: float = 0.0
    a_delta: float = 0.0
    raw_data: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    timestamp: float = field(default_factory=time.time)
    duration: float = 0.0


# ------------------------------------------------------------------
# SRP: model-update logic extracted into its own class
# ------------------------------------------------------------------

class ModelUpdater:
    """Single-responsibility class that pushes deltas into K, S, A.

    Isolated so the update logic can be tested, replaced, or
    extended (e.g. with event sourcing) without touching BaseAgent.
    """

    def __init__(
        self,
        k_model: KnowledgeEvolution,
        s_model: SuspicionField,
        a_model: AccessPropagation,
    ) -> None:
        self._k = k_model
        self._s = s_model
        self._a = a_model

    def update(self, target: str, result: AgentResult) -> None:
        """Push *result* deltas into the three coupled models.

        Wrapped in a try/except so a model failure never crashes
        the pipeline — the error is logged and the agent continues.
        """
        try:
            self._k.step(
                suspicion=float(np.mean(self._s.field)),
                learning_action=result.k_gain / 10.0,
            )
        except Exception as exc:
            logger.error("K-model update failed: {}", exc)

        try:
            x_norm = (hash(target) % self._s.width) / self._s.width
            self._s.step(
                attack_positions=[(x_norm, 0.5, result.s_inc)],
                knowledge=self._k.knowledge,
                access=self._a.global_access,
            )
        except Exception as exc:
            logger.error("S-model update failed: {}", exc)

        try:
            target_host: Optional[str] = None
            if target in self._a.hosts:
                target_host = target
            self._a.step(
                knowledge=self._k.knowledge,
                target_host=target_host,
            )
        except Exception as exc:
            logger.error("A-model update failed: {}", exc)


# ------------------------------------------------------------------
# Base agent
# ------------------------------------------------------------------

class BaseAgent(ABC):
    """Abstract async agent wired to the three math models.

    Subclasses implement ``_execute`` with domain logic.  The public
    ``execute`` wrapper validates input, calls ``_execute``, then
    delegates model updates to :class:`ModelUpdater`.
    """

    name: str = "base"

    def __init__(
        self,
        k_model: KnowledgeEvolution,
        s_model: SuspicionField,
        a_model: AccessPropagation,
    ) -> None:
        self.k_model = k_model
        self.s_model = s_model
        self.a_model = a_model
        self._updater = ModelUpdater(k_model, s_model, a_model)
        self._action_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def execute(self, target: str, **kwargs: Any) -> AgentResult:
        """Run the agent and update all math models.

        Args:
            target: IP, CIDR, hostname, or vulnerability identifier.
            **kwargs: Forwarded to ``_execute``.

        Returns:
            ``AgentResult`` with deltas already applied to models.

        Raises:
            AgentValidationError: If *target* is invalid.
            AgentExecutionError: If ``_execute`` fails and cannot recover.
        """
        try:
            target = validate_target(target)
        except ValueError as exc:
            raise AgentValidationError(str(exc)) from exc

        t0 = time.time()
        self._action_count += 1
        logger.info("[{}] action #{} on target={}", self.name, self._action_count, target)

        try:
            result = await self._execute(target, **kwargs)
        except AgentError:
            raise
        except Exception as exc:
            logger.error("[{}] _execute failed: {}", self.name, exc)
            raise AgentExecutionError(f"{self.name} failed on {target}: {exc}") from exc

        result.duration = time.time() - t0
        self._updater.update(target, result)

        logger.info(
            "[{}] done in {:.2f}s  k_gain={:.3f} s_inc={:.3f} a_delta={:.3f}",
            self.name,
            result.duration,
            result.k_gain,
            result.s_inc,
            result.a_delta,
        )
        return result

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def get_state_snapshot(self) -> Dict[str, float]:
        """Return a flat dict of the current K / S / A values."""
        return {
            "knowledge": self.k_model.knowledge,
            "suspicion_mean": float(np.mean(self.s_model.field)),
            "suspicion_max": float(np.max(self.s_model.field)),
            "access_global": self.a_model.global_access,
        }

    @property
    def action_count(self) -> int:
        return self._action_count

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    async def _execute(self, target: str, **kwargs: Any) -> AgentResult:
        """Domain-specific logic implemented by each concrete agent."""
        ...
