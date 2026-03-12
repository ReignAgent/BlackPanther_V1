"""Celery tasks for async scan execution.

Runs the BlackPanther scan pipeline in a background worker,
broadcasting progress updates via WebSocket.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from celery import Celery
from loguru import logger

app = Celery("blackpanther")
app.config_from_object("blackpanther.api.django_settings", namespace="CELERY")


def _get_event_loop() -> asyncio.AbstractEventLoop:
    """Get or create an event loop for async operations."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("Loop is closed")
        return loop
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop


@app.task(bind=True, max_retries=3)
def run_scan_task(self, task_id: str) -> Dict[str, Any]:
    """Run a BlackPanther scan as a Celery task.
    
    Updates progress in the scan store and broadcasts via WebSocket.
    """
    from .views import get_scan_store

    store = get_scan_store()
    if task_id not in store:
        logger.error(f"[task] Scan {task_id} not found in store")
        return {"error": "Scan not found"}

    scan = store[task_id]
    scan["status"] = "running"
    scan["started_at"] = datetime.utcnow().isoformat()
    scan["message"] = "Initializing scan..."

    loop = _get_event_loop()

    try:
        result = loop.run_until_complete(_run_scan_async(task_id, scan))
        return result
    except Exception as e:
        logger.error(f"[task] Scan failed: {e}")
        scan["status"] = "failed"
        scan["message"] = str(e)
        scan["completed_at"] = datetime.utcnow().isoformat()
        return {"error": str(e)}


async def _run_scan_async(task_id: str, scan: Dict[str, Any]) -> Dict[str, Any]:
    """Run the scan pipeline asynchronously."""
    from blackpanther.agents.coordinator import Coordinator, CoordinatorConfig
    from blackpanther.settings import Settings, get_settings

    from .consumers import broadcast_log, broadcast_progress

    target = scan["target"]
    llm_provider = scan["llm_provider"]

    settings = get_settings()
    settings_override = settings.model_copy(update={"llm_provider": llm_provider})

    get_settings.cache_clear()

    scan["current_phase"] = "initialization"
    scan["progress"] = 5.0
    scan["message"] = "Initializing mathematical models..."
    await broadcast_progress(task_id, "initialization", 5.0, "Initializing mathematical models...")
    await broadcast_log(task_id, "info", "Initializing mathematical models...")

    config = CoordinatorConfig(
        max_exploits_per_run=scan.get("max_exploits", 20),
        nmap_timing=scan.get("nmap_timing", 3),
    )
    coordinator = Coordinator.from_defaults(target, config)

    from .interactive_viz import InteractiveVisualizer
    interactive_viz = InteractiveVisualizer(
        coordinator.k, coordinator.s, coordinator.a, coordinator.hjb
    )

    scan["current_phase"] = "recon"
    scan["progress"] = 10.0
    scan["message"] = f"Running nmap scan on {target}..."
    await broadcast_progress(task_id, "recon", 10.0, f"Running nmap scan on {target}...")
    await broadcast_log(task_id, "info", f"[✓] Running nmap scan on {target}...")

    try:
        recon_result = await coordinator._safe_execute(coordinator.recon, target)
        hosts = recon_result.raw_data.get("hosts", [target]) if recon_result else [target]
    except Exception as e:
        logger.warning(f"[task] Recon failed: {e}, using target as single host")
        hosts = [target]

    scan["progress"] = 25.0
    scan["message"] = f"Found {len(hosts)} hosts"
    await broadcast_log(task_id, "success", f"[✓] Found {len(hosts)} hosts")

    scan["current_phase"] = "scanning"
    scan["progress"] = 30.0
    scan["message"] = "Scanning for vulnerabilities..."
    await broadcast_progress(task_id, "scanning", 30.0, "Scanning for vulnerabilities...")

    all_vulns: List[Any] = []
    for i, host in enumerate(hosts):
        progress = 30.0 + (i / max(len(hosts), 1)) * 20.0
        scan["progress"] = progress
        await broadcast_log(task_id, "info", f"[●] Scanning host {host}...")

        try:
            from blackpanther.agents.scanner import ScannerAgent
            scan_result = await coordinator._safe_execute(
                coordinator.scanner, host, services={}
            )
            if scan_result:
                all_vulns.extend(ScannerAgent.vulns_from_result(scan_result))
        except Exception as e:
            logger.warning(f"[task] Scan of {host} failed: {e}")

    scan["progress"] = 50.0
    scan["message"] = f"Found {len(all_vulns)} vulnerabilities"
    await broadcast_log(task_id, "success", f"[✓] Found {len(all_vulns)} vulnerabilities")

    viz_data = interactive_viz.generate_all_charts()
    scan["visualizations"] = viz_data
    from .consumers import broadcast_visualization
    for chart_type, chart_data in viz_data.items():
        await broadcast_visualization(
            task_id, chart_type, chart_data.get("data", []), chart_data.get("layout", {})
        )

    scan["current_phase"] = "exploitation"
    scan["progress"] = 55.0
    scan["message"] = "Generating exploits with LLM..."
    await broadcast_progress(task_id, "exploitation", 55.0, "Generating exploits with LLM...")
    await broadcast_log(task_id, "info", f"[●] Generating exploits from {llm_provider.upper()}...")

    generated_exploits: List[Dict[str, Any]] = []
    max_exploits = scan.get("max_exploits", 20)

    for i, vuln in enumerate(sorted(all_vulns, key=lambda v: v.severity, reverse=True)[:max_exploits]):
        progress = 55.0 + (i / max(len(all_vulns[:max_exploits]), 1)) * 30.0
        scan["progress"] = progress
        scan["message"] = f"Generating exploit for {vuln.cve}..."
        await broadcast_log(task_id, "info", f"[●] Generating exploit for {vuln.cve}...")

        try:
            exploit_result = await coordinator._safe_execute(
                coordinator.exploit, target, vuln=vuln
            )
            if exploit_result and exploit_result.success:
                generated_exploits.append({
                    "cve": vuln.cve,
                    "script_path": exploit_result.raw_data.get("script_path", ""),
                    "aggressiveness": exploit_result.raw_data.get("aggressiveness", 0.0),
                    "code_lines": exploit_result.raw_data.get("code_lines", 0),
                    "llm_provider": llm_provider,
                })
                await broadcast_log(task_id, "success", f"[✓] Generated exploit for {vuln.cve}")
        except Exception as e:
            logger.warning(f"[task] Exploit generation failed for {vuln.cve}: {e}")
            await broadcast_log(task_id, "warning", f"[!] Failed to generate exploit for {vuln.cve}: {e}")

    viz_data = interactive_viz.generate_all_charts()
    scan["visualizations"] = viz_data
    for chart_type, chart_data in viz_data.items():
        await broadcast_visualization(
            task_id, chart_type, chart_data.get("data", []), chart_data.get("layout", {})
        )

    scan["current_phase"] = "finalizing"
    scan["progress"] = 90.0
    scan["message"] = "Evaluating HJB optimal policy..."
    await broadcast_progress(task_id, "finalizing", 90.0, "Evaluating HJB optimal policy...")
    await broadcast_log(task_id, "info", "[●] Evaluating HJB optimal policy...")

    state = coordinator.get_system_state()

    results = {
        "hosts": hosts,
        "vulns": [
            {
                "cve": v.cve,
                "severity": v.severity,
                "service": v.service,
                "port": v.port,
                "title": v.title,
                "version": v.version,
            }
            for v in all_vulns
        ],
        "exploits": generated_exploits,
        "knowledge_final": state.knowledge,
        "suspicion_mean": state.suspicion,
        "access_global": state.access,
        "duration": (datetime.utcnow() - datetime.fromisoformat(scan["started_at"])).total_seconds(),
    }

    scan["results"] = results
    scan["status"] = "completed"
    scan["progress"] = 100.0
    scan["message"] = "Scan completed successfully"
    scan["completed_at"] = datetime.utcnow().isoformat()

    await broadcast_progress(task_id, "complete", 100.0, "Scan completed successfully")
    await broadcast_log(task_id, "success", "[✓] Scan completed successfully")

    return results


@app.task(bind=True)
def generate_report_task(self, task_id: str) -> Dict[str, Any]:
    """Generate GPT-powered report for a completed scan."""
    from .views import get_scan_store

    store = get_scan_store()
    if task_id not in store:
        return {"error": "Scan not found"}

    scan = store[task_id]
    if scan["status"] not in ("completed", "failed"):
        return {"error": "Scan not complete"}

    loop = _get_event_loop()
    try:
        report = loop.run_until_complete(_generate_report_async(task_id, scan))
        scan["report"] = report
        return report
    except Exception as e:
        logger.error(f"[task] Report generation failed: {e}")
        return {"error": str(e)}


async def _generate_report_async(task_id: str, scan: Dict[str, Any]) -> Dict[str, Any]:
    """Generate report using GPT-3.5-turbo."""
    from blackpanther.agents.report_generator import ReportGenerator
    from blackpanther.settings import get_settings

    settings = get_settings()
    results = scan.get("results", {})

    generator = ReportGenerator(settings)

    executive_summary = await generator.generate_executive_summary(results)
    technical_details = await generator.generate_technical_details(results.get("vulns", []))
    risk_assessment = await generator.generate_risk_assessment(results)
    recommendations = await generator.generate_recommendations(results.get("exploits", []))

    return {
        "executive_summary": executive_summary,
        "technical_details": technical_details,
        "risk_assessment": risk_assessment,
        "recommendations": recommendations,
        "generated_at": datetime.utcnow().isoformat(),
        "model_used": settings.report_model,
    }


def query_report_sync(task_id: str, question: str, scan: Dict[str, Any]) -> Tuple[str, List[str]]:
    """Synchronous wrapper for interactive report query."""
    loop = _get_event_loop()
    return loop.run_until_complete(_query_report_async(task_id, question, scan))


async def _query_report_async(task_id: str, question: str, scan: Dict[str, Any]) -> Tuple[str, List[str]]:
    """Handle interactive Q&A on the report."""
    from blackpanther.agents.report_generator import ReportGenerator
    from blackpanther.settings import get_settings

    settings = get_settings()
    results = scan.get("results", {})
    report = scan.get("report", {})

    generator = ReportGenerator(settings)
    
    context = {
        "results": results,
        "report": report,
    }

    answer = await generator.interactive_query(question, context)
    context_used = ["scan_results", "generated_report"]

    return answer, context_used
