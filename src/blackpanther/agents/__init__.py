"""BlackPanther Agent Layer

Autonomous agents backed by differential-equation models for
knowledge evolution, suspicion diffusion, and access propagation.
"""

from .base import BaseAgent, AgentResult, AgentError, AgentExecutionError, AgentValidationError, ModelUpdateError
from .interfaces import LLMProvider, ReconBackend, VulnLookup
from .memory import MemoryStore, Experience
from .recon import ReconAgent, NmapBackend, SocketBackend
from .scanner import ScannerAgent, Vuln, SearchsploitLookup, StaticVulnLookup
from .exploit import ExploitAgent, DeepSeekProvider, StubLLMProvider
from .coordinator import Coordinator, CoordinatorConfig
from .visualizer import Visualizer

__all__ = [
    # Base
    "BaseAgent", "AgentResult", "AgentError",
    "AgentExecutionError", "AgentValidationError", "ModelUpdateError",
    # DIP interfaces
    "LLMProvider", "ReconBackend", "VulnLookup",
    # Concrete providers
    "NmapBackend", "SocketBackend",
    "SearchsploitLookup", "StaticVulnLookup",
    "DeepSeekProvider", "StubLLMProvider",
    # Agents
    "ReconAgent", "ScannerAgent", "ExploitAgent",
    # Infrastructure
    "MemoryStore", "Experience", "Vuln",
    "Coordinator", "CoordinatorConfig", "Visualizer",
]
