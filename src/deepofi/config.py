from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_CONFIG = PROJECT_ROOT / "configs" / "pipeline.yaml"


def load_config(path=PIPELINE_CONFIG):
    return yaml.safe_load(Path(path).read_text())


def session_kwargs(config):
    """Arguments of clean.session_bounds from the `session` section."""
    session = config["session"]
    return {"timezone": session["timezone"], "open_time": session["open"], "close_time": session["close"], "trim_minutes": session["trim_minutes"]}
