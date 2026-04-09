import tomllib
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def load_local_config() -> dict:
    config_path = Path(__file__).resolve().parent.parent / "config.toml"
    if not config_path.exists():
        return {}

    with config_path.open("rb") as f:
        return tomllib.load(f)
