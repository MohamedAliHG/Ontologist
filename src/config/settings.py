"""Default settings and config loading for Pass 1 schema candidate generation."""

from pathlib import Path
from typing import Any
import tomllib

DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_SIMILARITY_THRESHOLD = 0.82
DEFAULT_EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
DEFAULT_LLM_TEMPERATURE = 0.0

DEFAULT_VALIDATION_RETRIES = 3
DEFAULT_API_RETRIES = 3


def load_pass1_config(config_path: str | None) -> dict[str, Any]:
    """Load a Pass 1 TOML config file, returning an empty dict when omitted."""

    if config_path is None:
        return {}

    path = Path(config_path)
    with path.open("rb") as handle:
        return tomllib.load(handle)


def config_get(
    config: dict[str, Any],
    section: str,
    key: str,
    default: Any = None,
) -> Any:
    """Read a nested config value with a default."""

    section_values = config.get(section, {})
    if not isinstance(section_values, dict):
        return default
    value = section_values.get(key, default)
    if value == "":
        return default
    return value
