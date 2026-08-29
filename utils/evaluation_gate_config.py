"""Load versioned quality-gate profiles from one auditable policy file."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

import yaml


DEFAULT_GATE_CONFIG = Path("config/ci_quality_gates.yml")


def load_gate_profile(
    path: str | Path,
    gate_name: str,
    profile_name: str,
) -> Dict[str, Any]:
    config_path = Path(path)
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except OSError as exc:
        raise ValueError(f"gate config cannot be read: {config_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid gate config YAML: {config_path}: {exc}") from exc

    if payload.get("schema_version") != 1:
        raise ValueError("gate config schema_version must be 1")
    gates = payload.get("gates")
    if not isinstance(gates, Mapping) or gate_name not in gates:
        raise ValueError(f"gate config has no {gate_name!r} section")
    gate = gates[gate_name]
    profiles = gate.get("profiles") if isinstance(gate, Mapping) else None
    if not isinstance(profiles, Mapping) or profile_name not in profiles:
        raise ValueError(
            f"gate config {gate_name!r} has no profile {profile_name!r}"
        )
    profile = profiles[profile_name]
    if not isinstance(profile, Mapping):
        raise ValueError(f"gate profile {gate_name}.{profile_name} must be a mapping")

    return {
        "policy_version": str(payload.get("policy_version") or "unknown"),
        "gate": gate_name,
        "profile": profile_name,
        **dict(profile),
    }


def policy_value(
    policy: Mapping[str, Any],
    dotted_path: str,
    fallback: Any = None,
) -> Any:
    value: Any = policy
    for part in dotted_path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return fallback
        value = value[part]
    return value
