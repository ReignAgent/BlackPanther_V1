"""BlackPanther Agent Layer

Autonomous agents backed by differential-equation models for
knowledge evolution, suspicion diffusion, and access propagation.

Includes both network-level and web-application attack capabilities:
  - ReconAgent       : nmap / socket port scanning
  - ScannerAgent     : CVE lookup via searchsploit or static DB
  - ExploitAgent     : LLM-powered exploit generation
  - WebReconAgent    : Web app crawling, fingerprinting, endpoint discovery
  - WebAttackAgent   : Modular web vuln scanner (SQLi, XSS, JWT, IDOR, ...)
"""

from .base import BaseAgent, AgentResult, AgentError, AgentExecutionError, AgentValidationError, ModelUpdateError
from .interfaces import LLMProvider, ReconBackend, VulnLookup
from .memory import MemoryStore, Experience
from .recon import ReconAgent, NmapBackend, SocketBackend
from .scanner import ScannerAgent, Vuln, SearchsploitLookup, StaticVulnLookup
from .exploit import ExploitAgent, DeepSeekProvider, MistralProvider, StubLLMProvider
from .web_recon import WebReconAgent, WebEndpoint, WebFingerprint, WebReconResult
from .web_attack import (
    WebAttackAgent, WebVuln, AttackModule,
    SQLiAttack, XSSAttack, AuthAttack, JWTAttack, IDORAttack,
    TraversalAttack, DisclosureAttack, NoSQLiAttack, SSRFAttack,
    APIAttack, MisconfigAttack,
)
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
    # Network agents
    "ReconAgent", "ScannerAgent", "ExploitAgent",
    # Web agents
    "WebReconAgent", "WebAttackAgent",
    "WebEndpoint", "WebFingerprint", "WebReconResult", "WebVuln",
    # Attack modules
    "AttackModule", "SQLiAttack", "XSSAttack", "AuthAttack",
    "JWTAttack", "IDORAttack", "TraversalAttack", "DisclosureAttack",
    "NoSQLiAttack", "SSRFAttack", "APIAttack", "MisconfigAttack",
    # Infrastructure
    "MemoryStore", "Experience", "Vuln",
    "Coordinator", "CoordinatorConfig", "Visualizer",
    "run_with_progress",
    # Console
    "ProgressConsole", "ScanProgressTracker", "TaskStatus",
    # Report Generation
    "ReportGenerator", "StubReportGenerator",
]
