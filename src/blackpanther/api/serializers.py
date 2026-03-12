"""Serializers for BlackPanther API.

Uses Django REST Framework serializers for request/response validation.
"""

from rest_framework import serializers


class ScanStartSerializer(serializers.Serializer):
    """Request body for starting a new scan."""
    target = serializers.CharField(
        max_length=255,
        help_text="Target IP, hostname, or CIDR range"
    )
    llm_provider = serializers.ChoiceField(
        choices=["deepseek", "mistral"],
        default="deepseek",
        help_text="LLM provider for exploit generation"
    )
    max_exploits = serializers.IntegerField(
        min_value=1,
        max_value=100,
        default=20,
        help_text="Maximum exploits to generate per run"
    )
    nmap_timing = serializers.IntegerField(
        min_value=0,
        max_value=5,
        default=3,
        help_text="Nmap timing template (0-5)"
    )


class ScanStatusSerializer(serializers.Serializer):
    """Response for scan status endpoint."""
    task_id = serializers.CharField()
    status = serializers.ChoiceField(
        choices=["pending", "running", "completed", "failed", "stopped"]
    )
    progress = serializers.FloatField(min_value=0, max_value=100)
    current_phase = serializers.CharField(allow_null=True)
    message = serializers.CharField(allow_blank=True)
    started_at = serializers.DateTimeField(allow_null=True)
    completed_at = serializers.DateTimeField(allow_null=True)
    duration_seconds = serializers.FloatField(allow_null=True)


class VulnerabilitySerializer(serializers.Serializer):
    """Serializer for vulnerability data."""
    cve = serializers.CharField()
    severity = serializers.FloatField()
    service = serializers.CharField()
    port = serializers.IntegerField()
    title = serializers.CharField()
    version = serializers.CharField(allow_blank=True)


class ExploitSerializer(serializers.Serializer):
    """Serializer for generated exploit data."""
    cve = serializers.CharField()
    script_path = serializers.CharField()
    aggressiveness = serializers.FloatField()
    code_lines = serializers.IntegerField()
    llm_provider = serializers.CharField()


class ScanResultsSerializer(serializers.Serializer):
    """Response for scan results endpoint."""
    task_id = serializers.CharField()
    target = serializers.CharField()
    status = serializers.CharField()
    hosts = serializers.ListField(child=serializers.CharField())
    vulnerabilities = VulnerabilitySerializer(many=True)
    exploits = ExploitSerializer(many=True)
    knowledge_final = serializers.FloatField()
    suspicion_mean = serializers.FloatField()
    access_global = serializers.FloatField()
    duration_seconds = serializers.FloatField()


class PlotlyChartSerializer(serializers.Serializer):
    """Serializer for Plotly chart JSON."""
    chart_type = serializers.CharField()
    data = serializers.JSONField()
    layout = serializers.JSONField()
    config = serializers.JSONField(required=False, default=dict)


class VisualizationsSerializer(serializers.Serializer):
    """Response for all visualizations."""
    task_id = serializers.CharField()
    charts = PlotlyChartSerializer(many=True)
    timestamp = serializers.DateTimeField()


class ReportSerializer(serializers.Serializer):
    """Response for GPT-generated report."""
    task_id = serializers.CharField()
    executive_summary = serializers.CharField()
    technical_details = serializers.CharField()
    risk_assessment = serializers.CharField()
    recommendations = serializers.CharField()
    generated_at = serializers.DateTimeField()
    model_used = serializers.CharField()


class ReportQuerySerializer(serializers.Serializer):
    """Request for interactive report query."""
    question = serializers.CharField(max_length=1000)


class ReportQueryResponseSerializer(serializers.Serializer):
    """Response for interactive report query."""
    question = serializers.CharField()
    answer = serializers.CharField()
    context_used = serializers.ListField(child=serializers.CharField())


class LLMConfigSerializer(serializers.Serializer):
    """LLM configuration status."""
    available_providers = serializers.ListField(child=serializers.CharField())
    current_provider = serializers.CharField()
    deepseek_configured = serializers.BooleanField()
    mistral_configured = serializers.BooleanField()
    openai_configured = serializers.BooleanField()


class ErrorSerializer(serializers.Serializer):
    """Standard error response."""
    error = serializers.CharField()
    detail = serializers.CharField(required=False)
    code = serializers.CharField(required=False)
