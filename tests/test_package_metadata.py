import tomllib
from pathlib import Path

import pytest

from local_agent import __version__


pytestmark = pytest.mark.unit


def test_pyproject_exposes_pfit_cli_script():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())

    assert pyproject["project"]["scripts"]["pfit"] == "local_agent.cli.main:main"
    assert pyproject["project"]["dynamic"] == ["version"]


def test_package_version_is_defined():
    assert __version__
