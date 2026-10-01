import json
from argparse import Namespace
from pathlib import Path

import pytest

from sentinel_evc import cli, installation


def test_native_app_cache_is_bound_to_workspace_and_runtime(tmp_path, monkeypatch):
    built, opened = [], []

    def build(app, python, data_dir, *, source_dir):
        built.append(app)
        resources = app / "Contents/Resources"
        resources.mkdir(parents=True)
        (resources / "app-config.json").write_text(json.dumps({
            "schema_version": "native-app-v1", "python": str(Path(python).absolute()),
            "data_dir": str(data_dir), "source_dir": str(source_dir) if source_dir else "",
            "bind": "127.0.0.1",
        }))

    monkeypatch.setattr(cli.sys, "platform", "darwin")
    monkeypatch.setattr(installation, "build_macos_app", build)
    monkeypatch.setattr("subprocess.run", lambda command, **kwargs: opened.append(command[-1]))
    first = Namespace(browser=False, port=0, data_dir=str(tmp_path / "first"))
    second = Namespace(browser=False, port=0, data_dir=str(tmp_path / "second"))
    assert cli.cmd_app(first) == cli.cmd_app(first) == cli.cmd_app(second) == 0
    assert len(built) == 2 and opened[0] == opened[1] != opened[2]
    monkeypatch.setattr(cli.sys, "executable", str(tmp_path / "other-python"))
    assert cli.cmd_app(first) == 0
    assert len(built) == 3 and opened[-1] != opened[0]
    config = built[-1] / "Contents/Resources/app-config.json"
    config.write_text('{}')
    with pytest.raises(ValueError, match="配置"):
        cli.cmd_app(first)
    assert len(opened) == 4
