from pathlib import Path

import pytest
import yaml


pytestmark = pytest.mark.unit


def test_example_config_has_required_sections():
    config = yaml.safe_load(Path("pfit.example.yaml").read_text())

    assert config["llm"]["model"]
    assert config["llm"]["base_url"] == "http://localhost:11434"
    assert config["workflow"]["max_repair_attempts"] >= 1
