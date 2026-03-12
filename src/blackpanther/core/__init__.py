"""BlackPanther Core Mathematical Models

Coupled ODE/PDE system governing the autonomous penetration tester:
  K  -- Knowledge Evolution   (logistic ODE)
  S  -- Suspicion Field       (reaction-diffusion PDE)
  A  -- Access Propagation    (networked epidemic ODE)
  HJB -- Optimal Controller   (Hamilton-Jacobi-Bellman)
"""

from .knowledge import KnowledgeEvolution, KnowledgeState
from .suspicion import SuspicionField, SuspicionState
from .access import AccessPropagation, AccessState, HostAccess
from .control import HJBController, Control, SystemState
from .scanner import NetworkScanner, ScanResult, PortResult
from .base import DifferentialEquation, EquationState

__all__ = [
    "DifferentialEquation",
    "EquationState",
    "KnowledgeEvolution",
    "KnowledgeState",
    "SuspicionField",
    "SuspicionState",
    "AccessPropagation",
    "AccessState",
    "HostAccess",
    "HJBController",
    "Control",
    "SystemState",
    "NetworkScanner",
    "ScanResult",
    "PortResult",
]
