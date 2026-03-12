"""WebSocket consumers for real-time scan progress streaming.

Provides bidirectional communication between the React Native frontend
and the BlackPanther scanning engine.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Optional

from channels.generic.websocket import AsyncWebsocketConsumer
from loguru import logger


class ScanProgressConsumer(AsyncWebsocketConsumer):
    """WebSocket consumer for real-time scan progress updates.
    
    Clients connect to /ws/scan/{task_id} to receive:
    - Progress updates (phase, percentage, message)
    - Real-time visualization data (Plotly JSON)
    - Status changes (running, completed, failed)
    - Log messages from the scan process
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.task_id: Optional[str] = None
        self.room_group_name: Optional[str] = None
        self._polling_task: Optional[asyncio.Task] = None

    async def connect(self) -> None:
        """Handle WebSocket connection."""
        self.task_id = self.scope["url_route"]["kwargs"]["task_id"]
        self.room_group_name = f"scan_{self.task_id}"

        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )

        await self.accept()
        logger.info(f"[ws] Client connected to scan {self.task_id}")

        await self.send_json({
            "type": "connection_established",
            "task_id": self.task_id,
            "message": "Connected to scan progress stream",
        })

        self._polling_task = asyncio.create_task(self._poll_scan_status())

    async def disconnect(self, close_code: int) -> None:
        """Handle WebSocket disconnection."""
        if self._polling_task:
            self._polling_task.cancel()
            try:
                await self._polling_task
            except asyncio.CancelledError:
                pass

        if self.room_group_name:
            await self.channel_layer.group_discard(
                self.room_group_name,
                self.channel_name
            )
        logger.info(f"[ws] Client disconnected from scan {self.task_id}")

    async def receive(self, text_data: str) -> None:
        """Handle incoming WebSocket messages."""
        try:
            data = json.loads(text_data)
            message_type = data.get("type")

            if message_type == "ping":
                await self.send_json({"type": "pong"})

            elif message_type == "request_status":
                await self._send_current_status()

            elif message_type == "request_visualization":
                chart_type = data.get("chart_type", "all")
                await self._send_visualization(chart_type)

            elif message_type == "stop_scan":
                await self._handle_stop_scan()

            else:
                await self.send_json({
                    "type": "error",
                    "message": f"Unknown message type: {message_type}",
                })

        except json.JSONDecodeError:
            await self.send_json({
                "type": "error",
                "message": "Invalid JSON",
            })

    async def send_json(self, content: Dict[str, Any]) -> None:
        """Send JSON data to the WebSocket."""
        await self.send(text_data=json.dumps(content))

    async def scan_progress(self, event: Dict[str, Any]) -> None:
        """Handle scan progress events from the channel layer."""
        await self.send_json({
            "type": "progress",
            "task_id": self.task_id,
            "phase": event.get("phase"),
            "progress": event.get("progress"),
            "message": event.get("message"),
            "timestamp": event.get("timestamp"),
        })

    async def scan_visualization(self, event: Dict[str, Any]) -> None:
        """Handle visualization update events."""
        await self.send_json({
            "type": "visualization",
            "task_id": self.task_id,
            "chart_type": event.get("chart_type"),
            "data": event.get("data"),
            "layout": event.get("layout"),
            "timestamp": event.get("timestamp"),
        })

    async def scan_log(self, event: Dict[str, Any]) -> None:
        """Handle log message events."""
        await self.send_json({
            "type": "log",
            "task_id": self.task_id,
            "level": event.get("level", "info"),
            "message": event.get("message"),
            "timestamp": event.get("timestamp"),
        })

    async def scan_complete(self, event: Dict[str, Any]) -> None:
        """Handle scan completion events."""
        await self.send_json({
            "type": "complete",
            "task_id": self.task_id,
            "status": event.get("status"),
            "summary": event.get("summary"),
            "timestamp": event.get("timestamp"),
        })

    async def _poll_scan_status(self) -> None:
        """Poll scan status and send updates."""
        from .views import get_scan_store
        
        last_progress = -1
        last_phase = None

        while True:
            try:
                await asyncio.sleep(0.5)

                store = get_scan_store()
                if self.task_id not in store:
                    continue

                scan = store[self.task_id]
                current_progress = scan.get("progress", 0)
                current_phase = scan.get("current_phase")
                status = scan.get("status")

                if current_progress != last_progress or current_phase != last_phase:
                    await self.send_json({
                        "type": "progress",
                        "task_id": self.task_id,
                        "status": status,
                        "phase": current_phase,
                        "progress": current_progress,
                        "message": scan.get("message", ""),
                    })
                    last_progress = current_progress
                    last_phase = current_phase

                if status in ("completed", "failed", "stopped"):
                    await self.send_json({
                        "type": "complete",
                        "task_id": self.task_id,
                        "status": status,
                        "message": scan.get("message", ""),
                    })
                    break

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[ws] Polling error: {e}")
                await asyncio.sleep(1)

    async def _send_current_status(self) -> None:
        """Send current scan status."""
        from .views import get_scan_store

        store = get_scan_store()
        if self.task_id not in store:
            await self.send_json({
                "type": "error",
                "message": "Scan not found",
            })
            return

        scan = store[self.task_id]
        await self.send_json({
            "type": "status",
            "task_id": self.task_id,
            "status": scan.get("status"),
            "phase": scan.get("current_phase"),
            "progress": scan.get("progress"),
            "message": scan.get("message"),
        })

    async def _send_visualization(self, chart_type: str) -> None:
        """Send visualization data."""
        from .views import get_scan_store

        store = get_scan_store()
        if self.task_id not in store:
            return

        scan = store[self.task_id]
        viz_data = scan.get("visualizations", {})

        if chart_type == "all":
            for ctype, cdata in viz_data.items():
                await self.send_json({
                    "type": "visualization",
                    "task_id": self.task_id,
                    "chart_type": ctype,
                    "data": cdata.get("data", []),
                    "layout": cdata.get("layout", {}),
                })
        elif chart_type in viz_data:
            cdata = viz_data[chart_type]
            await self.send_json({
                "type": "visualization",
                "task_id": self.task_id,
                "chart_type": chart_type,
                "data": cdata.get("data", []),
                "layout": cdata.get("layout", {}),
            })

    async def _handle_stop_scan(self) -> None:
        """Handle stop scan request."""
        from .views import get_scan_store

        store = get_scan_store()
        if self.task_id in store:
            store[self.task_id]["status"] = "stopped"
            store[self.task_id]["message"] = "Scan stopped by user"
            await self.send_json({
                "type": "stopped",
                "task_id": self.task_id,
                "message": "Scan stopped",
            })


async def broadcast_progress(task_id: str, phase: str, progress: float, message: str) -> None:
    """Broadcast progress update to all connected clients for a task.
    
    Call this from the Celery task to push updates.
    """
    from channels.layers import get_channel_layer
    from datetime import datetime

    channel_layer = get_channel_layer()
    if channel_layer is None:
        return

    await channel_layer.group_send(
        f"scan_{task_id}",
        {
            "type": "scan_progress",
            "phase": phase,
            "progress": progress,
            "message": message,
            "timestamp": datetime.utcnow().isoformat(),
        }
    )


async def broadcast_visualization(task_id: str, chart_type: str, data: Any, layout: Dict) -> None:
    """Broadcast visualization update to all connected clients."""
    from channels.layers import get_channel_layer
    from datetime import datetime

    channel_layer = get_channel_layer()
    if channel_layer is None:
        return

    await channel_layer.group_send(
        f"scan_{task_id}",
        {
            "type": "scan_visualization",
            "chart_type": chart_type,
            "data": data,
            "layout": layout,
            "timestamp": datetime.utcnow().isoformat(),
        }
    )


async def broadcast_log(task_id: str, level: str, message: str) -> None:
    """Broadcast log message to connected clients."""
    from channels.layers import get_channel_layer
    from datetime import datetime

    channel_layer = get_channel_layer()
    if channel_layer is None:
        return

    await channel_layer.group_send(
        f"scan_{task_id}",
        {
            "type": "scan_log",
            "level": level,
            "message": message,
            "timestamp": datetime.utcnow().isoformat(),
        }
    )
