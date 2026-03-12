"""Rich-based interactive console for progress display.

Provides beautiful, real-time progress tracking with:
  - Animated spinners and checkmarks
  - Task status updates
  - LLM provider switching notifications
  - Color-coded status messages
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Generator, List, Optional

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text


class TaskStatus(Enum):
    """Status of a tracked task."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass
class TaskInfo:
    """Information about a tracked task."""
    task_id: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    message: str = ""
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ProgressConsole:
    """Interactive console with real-time task tracking.
    
    Usage:
        console = ProgressConsole()
        with console.live_display():
            console.add_task("init", "Initializing models")
            console.update_task("init", TaskStatus.RUNNING)
            # ... do work ...
            console.update_task("init", TaskStatus.SUCCESS, "Models initialized")
    """

    STATUS_ICONS = {
        TaskStatus.PENDING: ("[ ]", "dim"),
        TaskStatus.RUNNING: ("[●]", "cyan"),
        TaskStatus.SUCCESS: ("[✓]", "green"),
        TaskStatus.WARNING: ("[!]", "yellow"),
        TaskStatus.ERROR: ("[✗]", "red"),
        TaskStatus.SKIPPED: ("[-]", "dim"),
    }

    def __init__(self, title: str = "BlackPanther v2.0") -> None:
        self.console = Console()
        self.title = title
        self.tasks: Dict[str, TaskInfo] = {}
        self.task_order: List[str] = []
        self.logs: List[str] = []
        self.max_logs = 10
        self._live: Optional[Live] = None
        self._start_time: Optional[float] = None
        self._callbacks: List[Callable[[str, TaskStatus, str], None]] = []

    def add_callback(self, callback: Callable[[str, TaskStatus, str], None]) -> None:
        """Add a callback to be called on task updates."""
        self._callbacks.append(callback)

    def add_task(
        self,
        task_id: str,
        description: str,
        status: TaskStatus = TaskStatus.PENDING,
    ) -> None:
        """Add a new task to track."""
        self.tasks[task_id] = TaskInfo(
            task_id=task_id,
            description=description,
            status=status,
        )
        if task_id not in self.task_order:
            self.task_order.append(task_id)
        self._refresh()

    def update_task(
        self,
        task_id: str,
        status: TaskStatus,
        message: str = "",
        **metadata: Any,
    ) -> None:
        """Update a task's status."""
        if task_id not in self.tasks:
            self.add_task(task_id, message or task_id)

        task = self.tasks[task_id]
        
        if status == TaskStatus.RUNNING and task.started_at is None:
            task.started_at = time.time()
        
        if status in (TaskStatus.SUCCESS, TaskStatus.ERROR, TaskStatus.WARNING):
            task.completed_at = time.time()

        task.status = status
        if message:
            task.message = message
        task.metadata.update(metadata)

        for callback in self._callbacks:
            try:
                callback(task_id, status, message)
            except Exception:
                pass

        self._refresh()

    def log(self, message: str, level: str = "info") -> None:
        """Add a log message."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        level_colors = {
            "info": "white",
            "success": "green",
            "warning": "yellow",
            "error": "red",
            "debug": "dim",
        }
        color = level_colors.get(level, "white")
        formatted = f"[dim]{timestamp}[/dim] [{color}]{message}[/{color}]"
        self.logs.append(formatted)
        if len(self.logs) > self.max_logs:
            self.logs = self.logs[-self.max_logs:]
        self._refresh()

    def llm_switch_notification(self, from_provider: str, to_provider: str, reason: str = "") -> None:
        """Display LLM provider switch notification."""
        self.log(
            f"[yellow]LLM Switch:[/yellow] {from_provider} → {to_provider}"
            + (f" ({reason})" if reason else ""),
            "warning"
        )
        self.add_task(
            "llm_switch",
            f"Switched to {to_provider}",
            TaskStatus.WARNING,
        )
        self.update_task("llm_switch", TaskStatus.WARNING, reason)

    def _build_display(self) -> Panel:
        """Build the display panel."""
        elements = []

        task_table = Table(show_header=False, box=None, padding=(0, 1))
        task_table.add_column("Status", width=4)
        task_table.add_column("Description", ratio=3)
        task_table.add_column("Message", ratio=2)
        task_table.add_column("Time", width=10, justify="right")

        for task_id in self.task_order:
            task = self.tasks[task_id]
            icon, style = self.STATUS_ICONS[task.status]
            
            duration = ""
            if task.started_at:
                end = task.completed_at or time.time()
                elapsed = end - task.started_at
                if elapsed < 60:
                    duration = f"{elapsed:.1f}s"
                else:
                    duration = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"

            task_table.add_row(
                Text(icon, style=style),
                Text(task.description, style=style if task.status == TaskStatus.RUNNING else ""),
                Text(task.message, style="dim"),
                Text(duration, style="dim"),
            )

        elements.append(task_table)

        if self.logs:
            elements.append(Text(""))
            elements.append(Text("─" * 50, style="dim"))
            for log_line in self.logs[-5:]:
                elements.append(Text.from_markup(log_line))

        if self._start_time:
            total_elapsed = time.time() - self._start_time
            elements.append(Text(""))
            elements.append(Text(f"Total elapsed: {total_elapsed:.1f}s", style="dim"))

        return Panel(
            Group(*elements),
            title=f"[bold cyan]{self.title}[/bold cyan]",
            subtitle="[dim]Autonomous Penetration Testing[/dim]",
            border_style="cyan",
            padding=(1, 2),
        )

    def _refresh(self) -> None:
        """Refresh the live display."""
        if self._live:
            self._live.update(self._build_display())

    @contextmanager
    def live_display(self) -> Generator[ProgressConsole, None, None]:
        """Context manager for live display."""
        self._start_time = time.time()
        with Live(self._build_display(), console=self.console, refresh_per_second=10) as live:
            self._live = live
            try:
                yield self
            finally:
                self._live = None

    def print_summary(self, results: Dict[str, Any]) -> None:
        """Print a final summary table."""
        is_web = results.get("mode") == "web"
        title_suffix = "Web App Attack" if is_web else "Run"
        table = Table(title=f"{self.title} - {title_suffix} Summary", border_style="cyan")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Mode", "[bold magenta]WEB APP[/bold magenta]" if is_web else "NETWORK")
        table.add_row("Target", results.get("target", "unknown"))

        if "hosts" in results:
            table.add_row("Hosts Discovered", str(len(results["hosts"])))
        if "vulns" in results:
            table.add_row("Network CVEs", str(len(results["vulns"])))

        web_vulns = results.get("web_vulns", [])
        if web_vulns:
            total = len(web_vulns)
            crit = sum(1 for v in web_vulns if v.get("severity", 0) >= 9.0)
            high = sum(1 for v in web_vulns if 7.0 <= v.get("severity", 0) < 9.0)
            med = sum(1 for v in web_vulns if 4.0 <= v.get("severity", 0) < 7.0)
            low = sum(1 for v in web_vulns if v.get("severity", 0) < 4.0)
            cats = list({v.get("category", "") for v in web_vulns})

            table.add_row(
                "Web Vulnerabilities",
                f"[bold red]{total}[/bold red]"
            )
            table.add_row(
                "  Severity",
                f"[red]{crit} CRIT[/red] / [yellow]{high} HIGH[/yellow] / "
                f"[blue]{med} MED[/blue] / [dim]{low} LOW[/dim]"
            )
            table.add_row("  Categories", ", ".join(cats))

        if "exploits" in results:
            table.add_row("Exploits Generated", str(len(results["exploits"])))

        if "web_recon" in results:
            recon = results["web_recon"]
            table.add_row("Endpoints Mapped", str(recon.get("total_endpoints", 0)))
            table.add_row("Hidden Files", str(recon.get("hidden_files", 0)))
            table.add_row("API Endpoints", str(recon.get("api_endpoints", 0)))

        if "knowledge_final" in results:
            table.add_row("Final Knowledge", f"{results['knowledge_final']:.2f}")
        if "suspicion_mean" in results:
            table.add_row("Mean Suspicion", f"{results['suspicion_mean']:.4f}")
        if "access_global" in results:
            table.add_row("Global Access", f"{results['access_global']:.3f}")
        if "duration" in results:
            table.add_row("Duration", f"{results['duration']:.1f}s")
        if "web_report" in results:
            table.add_row("Report", str(results["web_report"]))

        task_summary = {
            "success": sum(1 for t in self.tasks.values() if t.status == TaskStatus.SUCCESS),
            "warning": sum(1 for t in self.tasks.values() if t.status == TaskStatus.WARNING),
            "error": sum(1 for t in self.tasks.values() if t.status == TaskStatus.ERROR),
        }
        table.add_row(
            "Task Status",
            f"[green]{task_summary['success']} ok[/green] / "
            f"[yellow]{task_summary['warning']} warn[/yellow] / "
            f"[red]{task_summary['error']} err[/red]"
        )

        self.console.print()
        self.console.print(table)

        if web_vulns:
            self._print_vuln_details(web_vulns)

    def _print_vuln_details(self, web_vulns: list) -> None:
        """Print detailed vulnerability findings."""
        vuln_table = Table(
            title="Vulnerability Findings",
            border_style="red",
            show_lines=True,
        )
        vuln_table.add_column("#", style="dim", width=3)
        vuln_table.add_column("Severity", width=8)
        vuln_table.add_column("Category", width=12, style="cyan")
        vuln_table.add_column("Title", ratio=2)
        vuln_table.add_column("OWASP", width=20, style="dim")
        vuln_table.add_column("Evidence", ratio=2, style="dim")

        sorted_vulns = sorted(web_vulns, key=lambda v: v.get("severity", 0), reverse=True)

        for i, v in enumerate(sorted_vulns[:30], 1):
            sev = v.get("severity", 0)
            if sev >= 9.0:
                sev_str = f"[bold red]{sev:.1f}[/bold red]"
            elif sev >= 7.0:
                sev_str = f"[red]{sev:.1f}[/red]"
            elif sev >= 4.0:
                sev_str = f"[yellow]{sev:.1f}[/yellow]"
            else:
                sev_str = f"[dim]{sev:.1f}[/dim]"

            vuln_table.add_row(
                str(i),
                sev_str,
                v.get("category", ""),
                v.get("title", ""),
                v.get("owasp", ""),
                v.get("evidence", "")[:80],
            )

        self.console.print()
        self.console.print(vuln_table)


class ScanProgressTracker:
    """High-level progress tracker for scan operations.
    
    Provides semantic methods for common scan phases.
    """

    def __init__(self, console: Optional[ProgressConsole] = None) -> None:
        self.console = console or ProgressConsole()
        self._current_llm: Optional[str] = None

    def start_initialization(self) -> None:
        """Mark initialization as started."""
        self.console.add_task("init", "Initializing mathematical models...", TaskStatus.RUNNING)
        self.console.log("Loading K, S, A models and HJB controller", "info")

    def complete_initialization(self) -> None:
        """Mark initialization as complete."""
        self.console.update_task("init", TaskStatus.SUCCESS, "Models initialized")

    def start_recon(self, target: str) -> None:
        """Start reconnaissance phase."""
        self.console.add_task("recon", f"Running nmap on {target}...", TaskStatus.RUNNING)
        self.console.log(f"Starting nmap scan with timing template", "info")

    def complete_recon(self, host_count: int) -> None:
        """Complete reconnaissance phase."""
        self.console.update_task("recon", TaskStatus.SUCCESS, f"Found {host_count} hosts")

    def start_scanning(self, host_count: int) -> None:
        """Start vulnerability scanning phase."""
        self.console.add_task("scan", f"Scanning {host_count} hosts for vulnerabilities...", TaskStatus.RUNNING)

    def complete_scanning(self, vuln_count: int) -> None:
        """Complete vulnerability scanning phase."""
        self.console.update_task("scan", TaskStatus.SUCCESS, f"Found {vuln_count} vulnerabilities")

    def start_exploit_generation(self, llm_provider: str, cve: str) -> None:
        """Start exploit generation for a CVE."""
        self._current_llm = llm_provider
        self.console.add_task(
            f"exploit_{cve}",
            f"Generating exploit from {llm_provider.upper()}...",
            TaskStatus.RUNNING,
        )

    def complete_exploit_generation(self, cve: str, success: bool) -> None:
        """Complete exploit generation."""
        status = TaskStatus.SUCCESS if success else TaskStatus.WARNING
        message = "Generated" if success else "Generation failed"
        self.console.update_task(f"exploit_{cve}", status, message)

    def llm_switch(self, from_provider: str, to_provider: str, reason: str) -> None:
        """Handle LLM provider switch."""
        self.console.llm_switch_notification(from_provider, to_provider, reason)
        self._current_llm = to_provider

    def start_web_recon(self, target: str) -> None:
        """Start web application reconnaissance phase."""
        self.console.add_task("web_recon", f"Deep web recon on {target}...", TaskStatus.RUNNING)
        self.console.log("Crawling endpoints, fingerprinting tech stack, discovering hidden files", "info")

    def complete_web_recon(self, endpoint_count: int, hidden_count: int) -> None:
        """Complete web reconnaissance phase."""
        self.console.update_task(
            "web_recon", TaskStatus.SUCCESS,
            f"Mapped {endpoint_count} endpoints, {hidden_count} hidden files",
        )

    def start_web_attack(self) -> None:
        """Start web attack modules."""
        self.console.add_task("web_attack", "Running 11 web attack modules...", TaskStatus.RUNNING)
        self.console.log(
            "SQLi | XSS | Auth | JWT | IDOR | Traversal | Disclosure | NoSQLi | SSRF | API | Misconfig",
            "info",
        )

    def complete_web_attack(
        self, total: int, critical: int, high: int, categories: list,
    ) -> None:
        """Complete web attack phase."""
        if total == 0:
            self.console.update_task("web_attack", TaskStatus.WARNING, "No vulnerabilities found")
        else:
            self.console.update_task(
                "web_attack", TaskStatus.SUCCESS,
                f"{total} vulns ({critical} crit, {high} high) across {len(categories)} categories",
            )
            if critical > 0:
                self.console.log(
                    f"[red bold]CRITICAL:[/red bold] {critical} critical vulnerabilities found!",
                    "error",
                )

    def start_hjb_evaluation(self) -> None:
        """Start HJB policy evaluation."""
        self.console.add_task("hjb", "Evaluating HJB optimal policy...", TaskStatus.RUNNING)

    def complete_hjb_evaluation(self) -> None:
        """Complete HJB policy evaluation."""
        self.console.update_task("hjb", TaskStatus.SUCCESS, "Policy computed")

    def start_report_generation(self) -> None:
        """Start report generation."""
        self.console.add_task("report", "Generating interactive report...", TaskStatus.RUNNING)
        self.console.log("Using GPT-3.5-turbo for report generation", "info")

    def complete_report_generation(self) -> None:
        """Complete report generation."""
        self.console.update_task("report", TaskStatus.SUCCESS, "Report generated")

    def error(self, task_id: str, message: str) -> None:
        """Mark a task as errored."""
        self.console.update_task(task_id, TaskStatus.ERROR, message)
        self.console.log(f"Error in {task_id}: {message}", "error")

    def warning(self, message: str) -> None:
        """Log a warning."""
        self.console.log(message, "warning")

    @contextmanager
    def live(self) -> Generator[ScanProgressTracker, None, None]:
        """Context manager for live display."""
        with self.console.live_display():
            yield self


def create_default_progress_console() -> ProgressConsole:
    """Create a default progress console with standard tasks."""
    console = ProgressConsole("BlackPanther v2.0 - Autonomous Penetration Testing")
    
    console.add_task("init", "Initializing mathematical models...")
    console.add_task("recon", "Running nmap scan...")
    console.add_task("scan", "Scanning for vulnerabilities...")
    console.add_task("exploit", "Generating exploits...")
    console.add_task("hjb", "Evaluating HJB optimal policy...")
    console.add_task("report", "Generating interactive report...")
    
    return console
