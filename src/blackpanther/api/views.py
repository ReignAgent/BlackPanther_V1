"""REST API views for BlackPanther.

Provides endpoints for scan management, visualization retrieval,
and report generation.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict

from django.http import JsonResponse
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from blackpanther.settings import get_settings

from .serializers import (
    ErrorSerializer,
    LLMConfigSerializer,
    ReportQuerySerializer,
    ReportQueryResponseSerializer,
    ReportSerializer,
    ScanResultsSerializer,
    ScanStartSerializer,
    ScanStatusSerializer,
    VisualizationsSerializer,
)

_scan_store: Dict[str, Dict[str, Any]] = {}


@api_view(["GET"])
def health_check(request: Request) -> Response:
    """Health check endpoint."""
    return Response({
        "status": "healthy",
        "service": "blackpanther-api",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
    })


class ScanStartView(APIView):
    """Start a new penetration testing scan."""

    def post(self, request: Request) -> Response:
        serializer = ScanStartSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(
                ErrorSerializer({"error": "Invalid request", "detail": str(serializer.errors)}).initial_data,
                status=status.HTTP_400_BAD_REQUEST
            )

        data = serializer.validated_data
        task_id = str(uuid.uuid4())

        _scan_store[task_id] = {
            "task_id": task_id,
            "target": data["target"],
            "llm_provider": data["llm_provider"],
            "max_exploits": data["max_exploits"],
            "nmap_timing": data["nmap_timing"],
            "status": "pending",
            "progress": 0.0,
            "current_phase": None,
            "message": "Scan queued",
            "started_at": None,
            "completed_at": None,
            "results": None,
            "visualizations": None,
            "report": None,
        }

        from .tasks import run_scan_task
        run_scan_task.delay(task_id)

        return Response({
            "task_id": task_id,
            "status": "pending",
            "message": f"Scan started for target: {data['target']}",
        }, status=status.HTTP_202_ACCEPTED)


class ScanStatusView(APIView):
    """Get status of a running or completed scan."""

    def get(self, request: Request, task_id: str) -> Response:
        if task_id not in _scan_store:
            return Response(
                {"error": "Scan not found", "detail": f"No scan with task_id: {task_id}"},
                status=status.HTTP_404_NOT_FOUND
            )

        scan = _scan_store[task_id]
        duration = None
        if scan["started_at"]:
            end_time = scan["completed_at"] or datetime.utcnow()
            if isinstance(scan["started_at"], str):
                start = datetime.fromisoformat(scan["started_at"])
            else:
                start = scan["started_at"]
            if isinstance(end_time, str):
                end_time = datetime.fromisoformat(end_time)
            duration = (end_time - start).total_seconds()

        return Response(ScanStatusSerializer({
            "task_id": scan["task_id"],
            "status": scan["status"],
            "progress": scan["progress"],
            "current_phase": scan["current_phase"],
            "message": scan["message"],
            "started_at": scan["started_at"],
            "completed_at": scan["completed_at"],
            "duration_seconds": duration,
        }).initial_data)


class ScanResultsView(APIView):
    """Get results of a completed scan."""

    def get(self, request: Request, task_id: str) -> Response:
        if task_id not in _scan_store:
            return Response(
                {"error": "Scan not found"},
                status=status.HTTP_404_NOT_FOUND
            )

        scan = _scan_store[task_id]
        if scan["status"] not in ("completed", "failed"):
            return Response(
                {"error": "Scan not complete", "status": scan["status"]},
                status=status.HTTP_400_BAD_REQUEST
            )

        results = scan.get("results", {})
        return Response(ScanResultsSerializer({
            "task_id": task_id,
            "target": scan["target"],
            "status": scan["status"],
            "hosts": results.get("hosts", []),
            "vulnerabilities": results.get("vulns", []),
            "exploits": results.get("exploits", []),
            "knowledge_final": results.get("knowledge_final", 0.0),
            "suspicion_mean": results.get("suspicion_mean", 0.0),
            "access_global": results.get("access_global", 0.0),
            "duration_seconds": results.get("duration", 0.0),
        }).initial_data)


class ScanStopView(APIView):
    """Stop a running scan."""

    def post(self, request: Request, task_id: str) -> Response:
        if task_id not in _scan_store:
            return Response(
                {"error": "Scan not found"},
                status=status.HTTP_404_NOT_FOUND
            )

        scan = _scan_store[task_id]
        if scan["status"] != "running":
            return Response(
                {"error": "Scan is not running", "status": scan["status"]},
                status=status.HTTP_400_BAD_REQUEST
            )

        scan["status"] = "stopped"
        scan["message"] = "Scan stopped by user"
        scan["completed_at"] = datetime.utcnow().isoformat()

        return Response({"task_id": task_id, "status": "stopped"})


class VisualizationsView(APIView):
    """Get all visualizations for a scan."""

    def get(self, request: Request, task_id: str) -> Response:
        if task_id not in _scan_store:
            return Response(
                {"error": "Scan not found"},
                status=status.HTTP_404_NOT_FOUND
            )

        scan = _scan_store[task_id]
        viz_data = scan.get("visualizations", {})

        charts = []
        for chart_type, chart_data in viz_data.items():
            charts.append({
                "chart_type": chart_type,
                "data": chart_data.get("data", []),
                "layout": chart_data.get("layout", {}),
                "config": chart_data.get("config", {}),
            })

        return Response(VisualizationsSerializer({
            "task_id": task_id,
            "charts": charts,
            "timestamp": datetime.utcnow(),
        }).initial_data)


class SingleVisualizationView(APIView):
    """Get a single visualization by type."""

    def get(self, request: Request, task_id: str, chart_type: str) -> Response:
        if task_id not in _scan_store:
            return Response(
                {"error": "Scan not found"},
                status=status.HTTP_404_NOT_FOUND
            )

        scan = _scan_store[task_id]
        viz_data = scan.get("visualizations", {})

        if chart_type not in viz_data:
            available = list(viz_data.keys())
            return Response(
                {"error": f"Chart type '{chart_type}' not found", "available": available},
                status=status.HTTP_404_NOT_FOUND
            )

        chart_data = viz_data[chart_type]
        return Response({
            "chart_type": chart_type,
            "data": chart_data.get("data", []),
            "layout": chart_data.get("layout", {}),
            "config": chart_data.get("config", {}),
            "timestamp": datetime.utcnow().isoformat(),
        })


class ReportView(APIView):
    """Get or generate GPT-powered report."""

    def get(self, request: Request, task_id: str) -> Response:
        if task_id not in _scan_store:
            return Response(
                {"error": "Scan not found"},
                status=status.HTTP_404_NOT_FOUND
            )

        scan = _scan_store[task_id]
        if scan["status"] not in ("completed", "failed"):
            return Response(
                {"error": "Scan not complete"},
                status=status.HTTP_400_BAD_REQUEST
            )

        report = scan.get("report")
        if not report:
            from .tasks import generate_report_task
            generate_report_task.delay(task_id)
            return Response({
                "task_id": task_id,
                "status": "generating",
                "message": "Report generation started",
            }, status=status.HTTP_202_ACCEPTED)

        return Response(ReportSerializer({
            "task_id": task_id,
            "executive_summary": report.get("executive_summary", ""),
            "technical_details": report.get("technical_details", ""),
            "risk_assessment": report.get("risk_assessment", ""),
            "recommendations": report.get("recommendations", ""),
            "generated_at": report.get("generated_at"),
            "model_used": report.get("model_used", "gpt-3.5-turbo"),
        }).initial_data)


class ReportQueryView(APIView):
    """Interactive Q&A on the report."""

    def post(self, request: Request, task_id: str) -> Response:
        if task_id not in _scan_store:
            return Response(
                {"error": "Scan not found"},
                status=status.HTTP_404_NOT_FOUND
            )

        serializer = ReportQuerySerializer(data=request.data)
        if not serializer.is_valid():
            return Response(
                {"error": "Invalid request", "detail": str(serializer.errors)},
                status=status.HTTP_400_BAD_REQUEST
            )

        scan = _scan_store[task_id]
        question = serializer.validated_data["question"]

        from .tasks import query_report_sync
        answer, context = query_report_sync(task_id, question, scan)

        return Response(ReportQueryResponseSerializer({
            "question": question,
            "answer": answer,
            "context_used": context,
        }).initial_data)


class LLMConfigView(APIView):
    """Get current LLM configuration status."""

    def get(self, request: Request) -> Response:
        settings = get_settings()
        
        available = []
        if settings.deepseek_api_key:
            available.append("deepseek")
        if settings.mistral_api_key:
            available.append("mistral")
        if not available:
            available.append("stub")

        return Response(LLMConfigSerializer({
            "available_providers": available,
            "current_provider": settings.llm_provider,
            "deepseek_configured": bool(settings.deepseek_api_key),
            "mistral_configured": bool(settings.mistral_api_key),
            "openai_configured": bool(settings.openai_api_key),
        }).initial_data)

    def put(self, request: Request) -> Response:
        provider = request.data.get("provider")
        if provider not in ("deepseek", "mistral"):
            return Response(
                {"error": "Invalid provider", "valid": ["deepseek", "mistral"]},
                status=status.HTTP_400_BAD_REQUEST
            )

        return Response({
            "message": f"LLM provider set to {provider}",
            "provider": provider,
            "note": "This change is session-only. Set LLM_PROVIDER env var for persistence.",
        })


def get_scan_store() -> Dict[str, Dict[str, Any]]:
    """Get the scan store (for use by tasks module)."""
    return _scan_store
