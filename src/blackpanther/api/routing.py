"""WebSocket URL routing for BlackPanther API."""

from django.urls import re_path

from . import consumers

websocket_urlpatterns = [
    re_path(r"ws/scan/(?P<task_id>[a-f0-9\-]+)$", consumers.ScanProgressConsumer.as_asgi()),
]
