"""Production resilience primitives.

Shared utilities that eliminate duplicated error-handling patterns
across agents and harden the system against transient failures.
"""

from __future__ import annotations

import asyncio
import ipaddress
import re
import shutil
from functools import wraps
from typing import Any, Awaitable, Callable, Optional, Tuple, TypeVar

from loguru import logger

T = TypeVar("T")


# ------------------------------------------------------------------
# Retry with exponential backoff
# ------------------------------------------------------------------

def async_retry(
    max_attempts: int = 3,
    backoff: float = 1.0,
    exceptions: Tuple[type, ...] = (Exception,),
    on_retry: Optional[Callable[[int, Exception], None]] = None,
) -> Callable:
    """Decorator that retries an async function on transient failures.

    Args:
        max_attempts: Total tries (1 = no retry).
        backoff:      Seconds multiplied by attempt number between retries.
        exceptions:   Exception types considered retryable.
        on_retry:     Optional callback ``(attempt, exc)`` for logging.
    """
    def decorator(fn: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exc: Optional[Exception] = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return await fn(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt < max_attempts:
                        wait = backoff * attempt
                        if on_retry:
                            on_retry(attempt, exc)
                        else:
                            logger.warning(
                                "retry {}/{} for {} after {:.1f}s — {}",
                                attempt, max_attempts, fn.__qualname__, wait, exc,
                            )
                        await asyncio.sleep(wait)
            raise last_exc  # type: ignore[misc]
        return wrapper
    return decorator


# ------------------------------------------------------------------
# Async subprocess runner (DRY)
# ------------------------------------------------------------------

async def run_subprocess(
    cmd: list[str],
    timeout: int = 30,
) -> Tuple[int, str, str]:
    """Run a command asynchronously and return ``(returncode, stdout, stderr)``.

    Raises ``asyncio.TimeoutError`` if the process exceeds *timeout*.
    Returns ``(-1, "", error_message)`` if the binary is not found.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(), timeout=timeout,
        )
        return (
            proc.returncode or 0,
            stdout_bytes.decode(errors="replace"),
            stderr_bytes.decode(errors="replace"),
        )
    except FileNotFoundError:
        return -1, "", f"binary not found: {cmd[0]}"
    except asyncio.TimeoutError:
        logger.warning("subprocess timed out after {}s: {}", timeout, " ".join(cmd))
        raise


# ------------------------------------------------------------------
# Tool availability check (DRY)
# ------------------------------------------------------------------

_tool_cache: dict[str, bool] = {}


def is_tool_available(name: str) -> bool:
    """Check whether a CLI tool is on ``$PATH`` (cached)."""
    if name not in _tool_cache:
        _tool_cache[name] = shutil.which(name) is not None
    return _tool_cache[name]


def clear_tool_cache() -> None:
    """Reset the tool cache (useful in tests)."""
    _tool_cache.clear()


# ------------------------------------------------------------------
# Input validation
# ------------------------------------------------------------------

_CIDR_RE = re.compile(
    r"^(\d{1,3}\.){3}\d{1,3}/\d{1,2}$"
)
_HOST_RE = re.compile(
    r"^[a-zA-Z0-9._:-]+$"
)


def validate_target(target: str) -> str:
    """Sanitize and validate a scan target string.

    Accepts:
      - IPv4 address (``192.168.1.1``)
      - CIDR notation (``10.0.0.0/24``)
      - Hostname (``web-server.local``)

    Raises:
        ValueError: If the target looks malicious or unparseable.
    """
    target = target.strip()
    if not target:
        raise ValueError("target must not be empty")

    if _CIDR_RE.match(target):
        ipaddress.ip_network(target, strict=False)
        return target

    if not _HOST_RE.match(target):
        raise ValueError(f"invalid target: {target!r}")

    try:
        ipaddress.ip_address(target)
        return target
    except ValueError:
        pass

    if len(target) > 253:
        raise ValueError(f"hostname too long ({len(target)} chars)")
    return target
