"""Environment-backed application configuration."""
from dataclasses import dataclass
import os

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Runtime settings. Secrets are read from the environment, never source code."""

    mongodb_uri: str | None = None
    mongodb_database: str = "image_verification"
    mongodb_collection: str = "analyses"
    mongodb_demo_collection: str = "demo_analyses"
    mongodb_connect_timeout_ms: int = 3000

    @classmethod
    def from_env(cls):
        uri = os.getenv("MONGODB_URI", "").strip() or None
        return cls(
            mongodb_uri=uri,
            mongodb_database=os.getenv("MONGODB_DATABASE", "image_verification").strip() or "image_verification",
            mongodb_collection=os.getenv("MONGODB_COLLECTION", "analyses").strip() or "analyses",
            mongodb_demo_collection=os.getenv("MONGODB_DEMO_COLLECTION", "demo_analyses").strip() or "demo_analyses",
            mongodb_connect_timeout_ms=_positive_int("MONGODB_CONNECT_TIMEOUT_MS", 3000),
        )


def _positive_int(name, default):
    value = os.getenv(name, str(default))
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return parsed
