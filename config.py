"""
Environment configuration and startup validation

- Validation happens eagerly via `validate_config()`, which the app calls at startup. Fail fast beats a confusing runtime crash mid-party.
- Numeric tuning knobs (auction duration, snipe window, starting balance) are configurable via env vars with sane defaults, since these are the values a host will actually want to tweak per-event without touching code.
- `use_llm_fallback` is intentionally separate from `OPENAI_API_KEY` being set: the LLM is only a *fallback* parser (see nodes.py). You can run this bot with zero OpenAI spend by leaving USE_LLM_FALLBACK=false and relying on the regex parser, which handles standard messages.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


@dataclass
class Settings:
    spotify_client_id: str = field(default_factory=lambda: os.getenv("SPOTIFY_CLIENT_ID", ""))
    spotify_client_secret: str = field(default_factory=lambda: os.getenv("SPOTIFY_CLIENT_SECRET", ""))
    spotify_redirect_uri: str = field(default_factory=lambda: os.getenv("SPOTIFY_REDIRECT_URI", ""))
    spotify_scope: str = field(
        default_factory=lambda: os.getenv(
            "SPOTIFY_SCOPE", "user-modify-playback-state user-read-playback-state"
        )
    )
    spotify_token_cache_path: str = field(
        default_factory=lambda: os.getenv("SPOTIFY_TOKEN_CACHE_PATH", ".spotify_host_token_cache")
    )

    # --- LLM (fallback parser only) ---
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    use_llm_fallback: bool = field(
        default_factory=lambda: os.getenv("USE_LLM_FALLBACK", "true").lower() == "true"
    )
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o-mini"))

    # --- Telegram Bot ---
    telegram_bot_token: str = field(default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN", ""))

    # --- Auction tuning ---
    starting_token_balance: int = field(
        default_factory=lambda: int(os.getenv("STARTING_TOKEN_BALANCE", "100"))
    )
    auction_duration_seconds: float = field(
        default_factory=lambda: float(os.getenv("AUCTION_DURATION_SECONDS", "20"))
    )
    snipe_window_seconds: float = field(
        default_factory=lambda: float(os.getenv("SNIPE_WINDOW_SECONDS", "10"))
    )
    snipe_extension_seconds: float = field(
        default_factory=lambda: float(os.getenv("SNIPE_EXTENSION_SECONDS", "30"))
    )
    # Prevents an unbounded bidding war from holding the aux cord hostage forever.
    max_auction_extensions: int = field(
        default_factory=lambda: int(os.getenv("MAX_AUCTION_EXTENSIONS", "5"))
    )
    tick_interval_seconds: float = field(
        default_factory=lambda: float(os.getenv("TICK_INTERVAL_SECONDS", "1.0"))
    )

    def validate(self, require_spotify: bool = True, require_llm: bool = False) -> List[str]:
        """Returns a list of human-readable problems; empty list = valid."""
        problems: List[str] = []

        if require_spotify:
            if not self.spotify_client_id:
                problems.append("SPOTIFY_CLIENT_ID is not set")
            if not self.spotify_client_secret:
                problems.append("SPOTIFY_CLIENT_SECRET is not set")
            if not self.spotify_redirect_uri:
                problems.append("SPOTIFY_REDIRECT_URI is not set")

        if self.use_llm_fallback and require_llm and not self.openai_api_key:
            problems.append(
                "USE_LLM_FALLBACK is true but OPENAI_API_KEY is not set "
                "(set USE_LLM_FALLBACK=false to run regex-only)"
            )

        if self.snipe_extension_seconds <= 0:
            problems.append("SNIPE_EXTENSION_SECONDS must be positive")

        if self.auction_duration_seconds <= 0:
            problems.append("AUCTION_DURATION_SECONDS must be positive")

        return problems


def validate_config(settings: "Settings", strict: bool = True) -> None:
    """Call at startup. Raises ConfigError if strict and problems exist."""
    problems = settings.validate(require_spotify=strict, require_llm=False)
    if problems:
        message = "Configuration problems found:\n  - " + "\n  - ".join(problems)
        if strict:
            raise ConfigError(message)
        else:
            import logging
            logging.getLogger("aux_cord_bot.config").warning(message)


settings = Settings()
