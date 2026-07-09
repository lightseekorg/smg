"""Unit tests for the smg CLI parser."""

import sys
import types

import pytest


@pytest.fixture
def cli_module(monkeypatch):
    stub = types.ModuleType("smg.smg_rs")
    stub.get_version_string = lambda: "smg test"
    stub.get_verbose_version_string = lambda: "smg test"
    monkeypatch.setitem(sys.modules, "smg.smg_rs", stub)
    sys.modules.pop("smg.cli", None)

    import smg.cli as cli

    return cli


def test_cli_parser_exposes_launch_and_serve(cli_module):
    parser = cli_module.create_parser()
    help_text = parser.format_help()

    assert "launch" in help_text
    assert "serve" in help_text
    assert "server" not in help_text


def test_cli_rejects_removed_server_subcommand(cli_module, capsys):
    with pytest.raises(SystemExit) as exc_info:
        cli_module.main(["server", "--help"])

    assert exc_info.value.code == 1
    assert "server" not in capsys.readouterr().out
