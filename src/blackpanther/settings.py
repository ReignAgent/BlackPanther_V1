"""Centralized, validated configuration.

All runtime parameters are loaded from environment variables (with
``.env`` file support).  No credentials or tunables are hardcoded
anywhere else in the codebase — every module imports ``get_settings()``.

Uses ``pydantic-settings`` so every value is type-checked at startup
and missing required secrets surface immediately as clear errors.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Single source of truth for all BlackPanther configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- LLM / AI ------------------------------------------------
    # Provider selection: deepseek, mistral, or auto (tries deepseek first)
    llm_provider: str = Field("deepseek", description="LLM provider choice: deepseek|mistral")

    # DeepSeek configuration
    deepseek_api_key: str = Field("", description="DeepSeek API key (empty = stub mode)")
    deepseek_base_url: str = Field("https://api.deepseek.com")
    deepseek_model: str = Field("deepseek-coder")
    deepseek_max_tokens: int = Field(2048, ge=64, le=16384)
    deepseek_temperature: float = Field(0.2, ge=0.0, le=2.0)

    # Mistral configuration (free tier alternative)
    mistral_api_key: str = Field("", description="Mistral API key (free tier)")
    mistral_base_url: str = Field("https://api.mistral.ai/v1")
    mistral_model: str = Field("mistral-small-latest")
    mistral_max_tokens: int = Field(2048, ge=64, le=16384)
    mistral_temperature: float = Field(0.2, ge=0.0, le=2.0)

    # OpenAI configuration (for GPT-3.5-turbo reports)
    openai_api_key: str = Field("", description="OpenAI API key for report generation")
    openai_base_url: str = Field("https://api.openai.com/v1")
    report_model: str = Field("gpt-3.5-turbo")
    report_max_tokens: int = Field(4096, ge=64, le=16384)
    report_temperature: float = Field(0.7, ge=0.0, le=2.0)

    # ---- Scanning -------------------------------------------------
    nmap_timing: int = Field(3, ge=0, le=5)
    nmap_extra_args: str = Field("-sS")
    scanner_timeout: float = Field(1.0, gt=0)
    scanner_max_threads: int = Field(50, ge=1, le=500)
    searchsploit_timeout: int = Field(30, ge=1)

    # ---- Pipeline -------------------------------------------------
    suspicion_threshold: float = Field(0.7, ge=0.0, le=1.0)
    attack_threshold: float = Field(0.3, ge=0.0, le=1.0)
    stealth_sleep_multiplier: float = Field(10.0, ge=0.0)
    max_exploits_per_run: int = Field(20, ge=1)

    # ---- Math model defaults --------------------------------------
    k_alpha: float = Field(0.1, ge=0.0, le=2.0)
    k_beta: float = Field(0.01, ge=0.0, le=1.0)
    k_gamma: float = Field(0.05, ge=0.0, le=1.0)
    k_max: float = Field(100.0, gt=0)
    k_initial: float = Field(1.0, ge=0.0)
    s_width: int = Field(50, ge=5)
    s_height: int = Field(50, ge=5)
    s_diffusion: float = Field(0.1, ge=0.0, le=1.0)
    s_reaction: float = Field(0.05, ge=0.0, le=1.0)
    s_delta: float = Field(0.01, ge=0.0, le=1.0)
    a_eta: float = Field(0.2, ge=0.0, le=1.0)
    a_mu: float = Field(0.01, ge=0.0, le=1.0)
    hjb_grid_points: int = Field(15, ge=5, le=100)
    hjb_gamma: float = Field(0.95, ge=0.0, le=1.0)
    hjb_dt: float = Field(0.1, gt=0)

    # ---- Web Attack -------------------------------------------------
    web_concurrent_requests: int = Field(10, ge=1, le=100)
    web_request_timeout: float = Field(10.0, gt=0)
    web_request_delay: float = Field(0.1, ge=0.0, description="Delay between requests (stealth)")
    web_user_agent: str = Field(
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    web_max_crawl_depth: int = Field(3, ge=1, le=10)
    web_max_pages: int = Field(200, ge=10, le=5000)
    web_attack_threads: int = Field(5, ge=1, le=50)
    web_follow_redirects: bool = Field(True)
    web_verify_ssl: bool = Field(False, description="Disable for self-signed certs")

    # ---- Visualization ---------------------------------------------
    realtime_plots: bool = Field(True, description="Live-updating matplotlib windows (disable for headless/API)")

    # ---- Storage --------------------------------------------------
    db_path: str = Field("data/memory.db")
    output_dir: str = Field("output/proofs")
    exploit_sandbox_dir: str = Field("output/exploits")
    web_report_dir: str = Field("output/web_reports")
    log_level: str = Field("INFO")

    # ---- Retry / Resilience ---------------------------------------
    max_retries: int = Field(3, ge=0, le=10)
    retry_backoff: float = Field(1.0, ge=0.0)
    agent_timeout: int = Field(120, ge=1)

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        allowed = {"TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}, got '{v}'")
        return upper

    @field_validator("llm_provider")
    @classmethod
    def _validate_llm_provider(cls, v: str) -> str:
        allowed = {"deepseek", "mistral"}
        lower = v.lower()
        if lower not in allowed:
            raise ValueError(f"llm_provider must be one of {allowed}, got '{v}'")
        return lower


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton settings instance.

    Cached so the ``.env`` file is read only once.  Call
    ``get_settings.cache_clear()`` in tests to reload.
    """
    return Settings()
