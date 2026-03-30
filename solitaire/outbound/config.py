"""Configuration for the outbound writing quality gate.

Loads per-persona thresholds from the persona YAML file.
Falls back to sensible defaults when no config section is present.
"""

from dataclasses import dataclass, field
from typing import Optional
import os

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore


@dataclass
class SurfaceConfig:
    enabled: bool = True
    em_dash_severity: str = "warning"
    cursed_cluster_threshold: int = 3
    cursed_cluster_window: int = 100


@dataclass
class StructuralConfig:
    enabled: bool = True
    paragraph_cv_threshold: float = 0.15
    paragraph_cv_warning: float = 0.10
    min_paragraphs: int = 3


@dataclass
class WritingGateConfig:
    """Full configuration for the outbound writing gate."""
    enabled: bool = True
    surface: SurfaceConfig = field(default_factory=SurfaceConfig)
    structural: StructuralConfig = field(default_factory=StructuralConfig)
    min_response_length: int = 50
    exclude_code_blocks: bool = True
    exclude_quoted_text: bool = True


def load_config(persona_key: Optional[str] = None, workspace: Optional[str] = None) -> WritingGateConfig:
    """Load writing gate config from persona YAML.

    Looks for a `writing_gate` section in the persona's YAML file.
    Falls back to defaults if not found or if YAML parsing fails.
    """
    config = WritingGateConfig()

    if not persona_key or not yaml:
        return config

    # Resolve persona YAML path
    ws = workspace or os.getcwd()
    yaml_path = os.path.join(ws, "librarian", "personas", persona_key, f"{persona_key}.yaml")
    if not os.path.isfile(yaml_path):
        return config

    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return config

    wg = data.get("writing_gate")
    if not wg or not isinstance(wg, dict):
        return config

    config.enabled = wg.get("enabled", True)
    config.min_response_length = wg.get("min_response_length", 50)
    config.exclude_code_blocks = wg.get("exclude_code_blocks", True)
    config.exclude_quoted_text = wg.get("exclude_quoted_text", True)

    layers = wg.get("layers", {})

    surface = layers.get("surface", {})
    if isinstance(surface, dict):
        config.surface.enabled = surface.get("enabled", True)
        config.surface.em_dash_severity = surface.get("em_dash_severity", "warning")
        config.surface.cursed_cluster_threshold = surface.get("cursed_cluster_threshold", 3)
        config.surface.cursed_cluster_window = surface.get("cursed_cluster_window", 100)

    structural = layers.get("structural", {})
    if isinstance(structural, dict):
        config.structural.enabled = structural.get("enabled", True)
        config.structural.paragraph_cv_threshold = structural.get("paragraph_cv_threshold", 0.15)
        config.structural.min_paragraphs = structural.get("min_paragraphs", 3)

    return config
