"""BlackPanther Agent Layer

Autonomous agents backed by differential-equation models for
knowledge evolution, suspicion diffusion, and access propagation.
"""

from .base import BaseAgent, AgentResult, AgentError, AgentExecutionError, AgentValidationError, ModelUpdateError
from .interfaces import LLMProvider, ReconBackend, VulnLookup
from .memory import MemoryStore, Experience
from .recon import ReconAgent, NmapBackend, SocketBackend
from .scanner import ScannerAgent, Vuln, SearchsploitLookup, StaticVulnLookup
from .exploit import ExploitAgent, DeepSeekProvider, MistralProvider, StubLLMProvider
from .coordinator import Coordinator, CoordinatorConfig, run_with_progress
from .visualizer import Visualizer
from .console import ProgressConsole, ScanProgressTracker, TaskStatus
from .report_generator import ReportGenerator, StubReportGenerator

__all__ = [
    # Base
    "BaseAgent", "AgentResult", "AgentError",
    "AgentExecutionError", "AgentValidationError", "ModelUpdateError",
    # DIP interfaces
    "LLMProvider", "ReconBackend", "VulnLookup",
    # Concrete providers
    "NmapBackend", "SocketBackend",
    "SearchsploitLookup", "StaticVulnLookup",
    "DeepSeekProvider", "MistralProvider", "StubLLMProvider",
    # Agents
    "ReconAgent", "ScannerAgent", "ExploitAgent",
    # Infrastructure
    "MemoryStore", "Experience", "Vuln",
    "Coordinator", "CoordinatorConfig", "Visualizer",
    "run_with_progress",
    # Console
    "ProgressConsole", "ScanProgressTracker", "TaskStatus",
    # Report Generation
    "ReportGenerator", "StubReportGenerator",
]
