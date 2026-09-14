"""Configuration loading and path resolution.

This module is the only entry point for reading the YAML configuration files
under ``configs/``. It resolves project-root-relative paths and exposes shallow,
typed accessors so the rest of the project never hard-codes settings.

Usage::

    from utils import config

    cfg = config.load_config("pipeline")
    img_size = config.get(cfg, "data.image_size")
    out_dir = config.resolve_path(config.get(cfg, "output.analysis_dir"))

"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml


class ConfigError(RuntimeError):
    """Raised when a configuration file cannot be located or parsed."""


ROOT_DIR = Path(__file__).resolve().parent.parent

CONFIG_DIR = ROOT_DIR / "configs"

_CACHE: dict[str, dict[str, Any]] = {}


def config_path(name: str) -> Path:
    """Return the filesystem path of a config by name (with or without .yaml)."""
    if not name.endswith(".yaml"):
        name = f"{name}.yaml"
    return CONFIG_DIR / name


def load_config(
    name: str,
    overrides: dict[str, Any] | None = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Load a YAML config file from ``configs/`` as a nested dictionary.

    Args:
        name: Config name, e.g. ``"unet"`` or ``"unet.yaml"``.
        overrides: Optional nested dictionary deep-merged on top of the file.
        use_cache: Reuse the previously loaded value when no overrides are given.

    Returns:
        A fresh copy of the configuration dictionary.

    Raises:
        ConfigError: If the file is missing, unreadable, or not a mapping.
    """
    if use_cache and overrides is None and name in _CACHE:
        return copy.deepcopy(_CACHE[name])

    data = _read_yaml(config_path(name))

    if overrides:
        data = _deep_merge(data, overrides)
    elif use_cache:
        _CACHE[name] = copy.deepcopy(data)

    return data


def clear_config_cache() -> None:
    """Drop all cached configurations (mainly useful in tests)."""
    _CACHE.clear()


def get(cfg: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    """Fetch a value using dot notation, e.g. ``"train.epochs"``.

    Returns ``default`` when the key is missing or is not a mapping along the way.
    """
    value: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def resolve_path(path: str | os.PathLike[str]) -> Path:
    """Resolve a path against the project root.

    Absolute paths are returned unchanged; ``~`` is expanded; relative paths are
    resolved relative to ``ROOT_DIR``.
    """
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return (ROOT_DIR / p).resolve()


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"Configuration file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Failed to parse configuration {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigError(f"Configuration root must be a mapping, got: {type(data).__name__}")

    return data


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if (
            isinstance(value, dict)
            and isinstance(merged.get(key), dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged