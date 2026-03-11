"""URL routing for BlackPanther API."""

from django.urls import path

from . import views

app_name = "blackpanther_api"

urlpatterns = [
    # Health check
    path("api/v1/health", views.health_check, name="health"),
    
    # Scan management
    path("api/v1/scan/start", views.ScanStartView.as_view(), name="scan_start"),
    path("api/v1/scan/<str:task_id>/status", views.ScanStatusView.as_view(), name="scan_status"),
    path("api/v1/scan/<str:task_id>/results", views.ScanResultsView.as_view(), name="scan_results"),
    path("api/v1/scan/<str:task_id>/stop", views.ScanStopView.as_view(), name="scan_stop"),
    
    # Visualizations
    path("api/v1/visualizations/<str:task_id>", views.VisualizationsView.as_view(), name="visualizations"),
    path("api/v1/visualizations/<str:task_id>/<str:chart_type>", views.SingleVisualizationView.as_view(), name="single_visualization"),
    
    # Reports
    path("api/v1/report/<str:task_id>", views.ReportView.as_view(), name="report"),
    path("api/v1/report/<str:task_id>/query", views.ReportQueryView.as_view(), name="report_query"),
    
    # Configuration
    path("api/v1/config/llm", views.LLMConfigView.as_view(), name="llm_config"),
]
