"""Client configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class ClientConfig:
    """Configuration for SmgClient."""

    base_url: str = "http://localhost:30000"
    api_key: str | None = None
    timeout: float = 60.0
    max_retries: int = 2
    default_headers: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        if self.api_key is None:
            self.api_key = os.environ.get("SMG_API_KEY")
