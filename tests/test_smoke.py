import pytest

import flightrec
from flightrec.cli import build_parser, main


def test_package_imports():
    assert flightrec.__version__ == "0.1.0"


def test_cli_no_args_prints_help_and_succeeds(capsys):
    assert main([]) == 0
    assert "flight recorder" in capsys.readouterr().out


def test_cli_version_flag():
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["--version"])
    assert exc.value.code == 0
