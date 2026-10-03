#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""LIBERO must be importable from non-interactive runs (no ``input()`` prompt on first use)."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest
import yaml

from lerobot.envs.configs import LiberoEnv, _ensure_libero_config

# Same first-import behaviour as libero/libero/__init__.py: prompt when the config is missing.
_FAKE_LIBERO_INIT = """
import os
config_file = os.path.join(
    os.environ.get("LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero")), "config.yaml"
)
if not os.path.exists(config_file):
    input("Do you want to specify a custom path for the dataset folder? (Y/N): ")
"""


def _no_input(*args, **kwargs):
    raise EOFError("input() called in a non-interactive run")


@pytest.fixture
def fake_libero(tmp_path, monkeypatch):
    """Install a minimal ``libero.libero`` package that prompts like the real one."""
    pkg_root = tmp_path / "site"
    inner = pkg_root / "libero" / "libero"
    inner.mkdir(parents=True)
    (pkg_root / "libero" / "__init__.py").write_text("")
    (inner / "__init__.py").write_text(_FAKE_LIBERO_INIT)

    monkeypatch.syspath_prepend(str(pkg_root))
    for name in ("libero", "libero.libero"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    importlib.invalidate_caches()

    config_dir = tmp_path / "libero_config"
    monkeypatch.setenv("LIBERO_CONFIG_PATH", str(config_dir))
    monkeypatch.setattr("builtins.input", _no_input)
    yield inner, config_dir
    for name in ("libero", "libero.libero"):
        sys.modules.pop(name, None)


def test_fake_libero_prompts_without_config(fake_libero):
    """Guard for the fixture itself: without a config, importing LIBERO asks for input."""
    with pytest.raises(EOFError):
        importlib.import_module("libero.libero")


def test_writes_default_config_and_import_does_not_prompt(fake_libero):
    inner, config_dir = fake_libero

    _ensure_libero_config()

    config = yaml.safe_load((config_dir / "config.yaml").read_text())
    root = str(inner)
    assert config == {
        "benchmark_root": root,
        "bddl_files": f"{root}/./bddl_files",
        "init_states": f"{root}/./init_files",
        "datasets": f"{root}/../datasets",
        "assets": f"{root}/./assets",
    }
    importlib.import_module("libero.libero")  # would raise EOFError if it prompted


def test_existing_config_is_left_untouched(fake_libero):
    _, config_dir = fake_libero
    config_dir.mkdir()
    config_file = config_dir / "config.yaml"
    config_file.write_text("datasets: /my/custom/datasets\n")

    _ensure_libero_config()

    assert config_file.read_text() == "datasets: /my/custom/datasets\n"


def test_no_config_written_when_libero_is_missing(tmp_path, monkeypatch):
    for name in ("libero", "libero.libero"):
        monkeypatch.setitem(sys.modules, name, None)  # makes find_spec raise ModuleNotFoundError
    monkeypatch.setenv("LIBERO_CONFIG_PATH", str(tmp_path / "libero_config"))

    _ensure_libero_config()

    assert not (tmp_path / "libero_config").exists()


def test_create_envs_does_not_prompt(fake_libero, monkeypatch):
    """create_envs() must import LIBERO without hitting the interactive prompt."""
    sentinel = object()

    def fake_create_libero_envs(**kwargs):
        importlib.import_module("libero.libero")  # the real module imports libero.libero at top level
        return sentinel

    fake_module = types.ModuleType("lerobot.envs.libero")
    fake_module.create_libero_envs = fake_create_libero_envs
    monkeypatch.setitem(sys.modules, "lerobot.envs.libero", fake_module)

    assert LiberoEnv(task="libero_spatial").create_envs(n_envs=1) is sentinel
    _, config_dir = fake_libero
    assert Path(config_dir / "config.yaml").is_file()
