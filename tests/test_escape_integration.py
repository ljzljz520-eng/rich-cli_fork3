"""End-to-end tests: untrusted content through every rendering entry point."""

import io
import json
import re

import pytest
from click.testing import CliRunner
from rich.console import Console

from rich_cli.__main__ import main, render_ipynb
from rich_cli.terminal_safety import POLICY_RAW, POLICY_STRICT

_SGR_RE = re.compile(r"\x1b\[[0-9:;]*m")
_EXECUTABLE_C1 = ("\x9b", "\x9d", "\x90", "\x9f", "\x98", "\x9e", "\x9c", "\x07")


def has_executable_controls(text):
    """True if anything besides allow-listed canonical SGR survives."""
    rest = _SGR_RE.sub("", text)
    return "\x1b" in rest or any(c in rest for c in _EXECUTABLE_C1)


# Clipboard write, screen erase, cursor home, title spoof, alt screen,
# followed by legitimate green text -- all in one "file".
ATTACK = (
    "normal "
    "\x1b]52;c;U0VDTE9BRA==\x07"
    "\x1b[2J\x1b[H"
    "\x1b]0;spoofed-title\x07"
    "\x1b[?1049h"
    " end "
    "\x1b[32mgreen\x1b[0m"
)


def assert_attacks_neutralised(output, *, sgr_survives=True):
    assert not has_executable_controls(output)
    assert "]52;c" not in output  # clipboard payload dropped entirely
    assert "?1049" not in output
    assert "normal" in output and "end" in output and "green" in output
    if sgr_survives:
        # Legitimately coloured log content keeps its colour through the
        # verbatim-text pipeline.
        assert "\x1b[32m" in output


@pytest.fixture()
def runner():
    return CliRunner()


@pytest.fixture()
def attack_file(tmp_path):
    path = tmp_path / "evil.txt"
    path.write_text(ATTACK, encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# Syntax / default path
# ---------------------------------------------------------------------------


def test_syntax_path_neutralises_attacks(runner, attack_file):
    result = runner.invoke(main, [attack_file, "--syntax"])
    assert result.exit_code == 0, result.output
    assert_attacks_neutralised(result.output)


def test_syntax_stdin_neutralises_attacks(runner):
    result = runner.invoke(main, ["-", "--syntax"], input=ATTACK)
    assert result.exit_code == 0, result.output
    assert_attacks_neutralised(result.output)


def test_forensic_mode_exposes_markers(runner, attack_file):
    result = runner.invoke(main, [attack_file, "--syntax", "--escape-forensics"])
    assert result.exit_code == 0, result.output
    assert "␛[2J" in result.output
    assert "␛]52;c;U0VDTE9BRA==" in result.output
    assert "␛]0;spoofed-title" in result.output
    # Markers are inert text: no executable control byte remains
    assert not has_executable_controls(result.output)


def test_strict_policy_drops_colors(runner, attack_file):
    result = runner.invoke(
        main, [attack_file, "--syntax", "--escape-policy", POLICY_STRICT]
    )
    assert result.exit_code == 0, result.output
    assert not has_executable_controls(result.output)
    assert "\x1b[32m" not in result.output
    assert "green" in result.output


def test_code_lexer_drops_embedded_sgr(runner, tmp_path):
    # A Python file containing ANSI colour bytes in its source: Pygments would
    # tokenize the sequence apart, so SGR must be removed outright.
    path = tmp_path / "evil.py"
    path.write_text(
        "print('x')\x1b[32mgreen\x1b[0m\n# \x1b[2J\x1b]0;t\x07\n", encoding="utf8"
    )
    result = runner.invoke(main, [str(path), "--syntax"])
    assert result.exit_code == 0, result.output
    assert "\x1b" not in result.output
    assert "green" in result.output


def test_code_lexer_sgr_dropped_in_html_export(runner, tmp_path):
    path = tmp_path / "evil.py"
    path.write_text("print('x')\x1b[32mgreen\x1b[0m\n", encoding="utf8")
    out_html = tmp_path / "p.html"
    result = runner.invoke(
        main, [str(path), "--syntax", "--export-html", str(out_html)]
    )
    assert result.exit_code == 0, result.output
    assert "\x1b" not in out_html.read_text(encoding="utf8")


def test_colored_log_keeps_color(runner):
    # Legitimate coloured log piped as plain text still renders coloured.
    result = runner.invoke(
        main,
        ["-", "--syntax", "--lexer", "text"],
        input="\x1b[32mOK\x1b[0m started\n",
    )
    assert result.exit_code == 0, result.output
    assert "\x1b[32m" in result.output


def test_raw_policy_is_explicit_passthrough(runner, attack_file):
    result = runner.invoke(
        main, [attack_file, "--syntax", "--escape-policy", POLICY_RAW]
    )
    assert result.exit_code == 0, result.output
    assert "]52;c" in result.output  # labelled escape hatch works as advertised


# ---------------------------------------------------------------------------
# Export backends consume the same sanitized renderable
# ---------------------------------------------------------------------------


def test_html_export_is_control_free(runner, attack_file, tmp_path):
    out_html = tmp_path / "out.html"
    result = runner.invoke(
        main, [attack_file, "--syntax", "--export-html", str(out_html)]
    )
    assert result.exit_code == 0, result.output
    html = out_html.read_text(encoding="utf8")
    assert not has_executable_controls(html)
    assert "]52;c" not in html
    assert "normal" in html


def test_svg_export_is_control_free(runner, attack_file, tmp_path):
    out_svg = tmp_path / "out.svg"
    result = runner.invoke(
        main, [attack_file, "--syntax", "--export-svg", str(out_svg)]
    )
    assert result.exit_code == 0, result.output
    svg = out_svg.read_text(encoding="utf8")
    assert not has_executable_controls(svg)
    assert "]52;c" not in svg


# ---------------------------------------------------------------------------
# Markup / rule path
# ---------------------------------------------------------------------------


def test_print_path_strips_attacks(runner):
    result = runner.invoke(
        main, ["-", "--print"], input="[bold]hi[/bold]\x1b[2J\x1b]52;c;A\x07"
    )
    assert result.exit_code == 0, result.output
    assert "\x1b[2J" not in result.output
    assert "]52;c" not in result.output
    assert "hi" in result.output


def test_rule_path_strips_attacks(runner):
    result = runner.invoke(
        main,
        ["\x1b]0;x\x07 title", "--rule"],
    )
    assert result.exit_code == 0, result.output
    assert "]0;x" not in result.output
    assert "title" in result.output


# ---------------------------------------------------------------------------
# JSON path -- "\u001b" only becomes ESC after decoding
# ---------------------------------------------------------------------------


def test_json_decoded_escapes_are_neutralised(runner, tmp_path):
    path = tmp_path / "evil.json"
    path.write_text(
        json.dumps(
            {
                "clip": "\x1b]52;c;SECRET\x07visible",
                "erase": "\x1b[2Jhi\x1b[H",
                "c1": "x\x9b2Jy",
            }
        ),
        encoding="utf8",
    )
    result = runner.invoke(main, [str(path), "--json"])
    assert result.exit_code == 0, result.output
    assert "\x1b" not in result.output
    assert "SECRET" not in result.output
    assert "visible" in result.output and "hi" in result.output


def test_json_forensic_markers_survive(runner, tmp_path):
    path = tmp_path / "evil.json"
    path.write_text(json.dumps({"v": "\x1b[2Jx"}), encoding="utf8")
    result = runner.invoke(main, [str(path), "--json", "--escape-forensics"])
    assert result.exit_code == 0, result.output
    assert "␛[2J" in result.output


# ---------------------------------------------------------------------------
# CSV path
# ---------------------------------------------------------------------------


def test_csv_cells_are_neutralised(runner, tmp_path):
    path = tmp_path / "evil.csv"
    path.write_text(f"name,note\nalice,{ATTACK}\n", encoding="utf8")
    result = runner.invoke(main, [str(path), "--csv"])
    assert result.exit_code == 0, result.output
    assert not has_executable_controls(result.output)
    assert "]52;c" not in result.output
    assert "alice" in result.output


# ---------------------------------------------------------------------------
# Markdown path
# ---------------------------------------------------------------------------


def test_markdown_content_is_neutralised(runner, tmp_path):
    path = tmp_path / "evil.md"
    path.write_text(f"# Hi\n\n{ATTACK}\n", encoding="utf8")
    result = runner.invoke(main, [str(path), "--markdown"])
    assert result.exit_code == 0, result.output
    assert not has_executable_controls(result.output)
    assert "]52;c" not in result.output


# ---------------------------------------------------------------------------
# Notebook path: stream / error / execute_result + split chunk lists
# ---------------------------------------------------------------------------


def _malicious_notebook(tmp_path):
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {"kernelspec": {"language": "python"}},
        "cells": [
            {
                "cell_type": "code",
                "execution_count": 1,
                "metadata": {},
                "source": ["print('hi')\x1b[32m"],
                "outputs": [
                    {
                        "output_type": "stream",
                        "name": "stdout",
                        # Clipboard sequence deliberately split across chunks.
                        "text": ["stdout \x1b]52;c;SE", "CRET\x07 tail"],
                    },
                    {
                        "output_type": "error",
                        "ename": "E",
                        "evalue": "v",
                        "traceback": [
                            "frame1",
                            "\x1b]8;;http://evil\x1b\\click\x1b]8;;\x1b\\",
                        ],
                    },
                    {
                        "output_type": "execute_result",
                        "execution_count": 2,
                        "data": {"text/plain": ["res", "ult \x9b2J\x1b[?1049h end"]},
                        "metadata": {},
                    },
                ],
            },
            {"cell_type": "raw", "metadata": {}, "source": ["raw \x1b]2;x\x07 z"]},
        ],
    }
    path = tmp_path / "evil.ipynb"
    path.write_text(json.dumps(notebook), encoding="utf8")  # \u001b on disk
    return str(path)


def test_notebook_direct_rendering_is_clean(tmp_path):
    path = _malicious_notebook(tmp_path)
    renderable = render_ipynb(
        path, "ansi_dark", False, "", None, None, False, False, True
    )
    buf = io.StringIO()
    console = Console(
        file=buf, force_terminal=True, color_system="truecolor", width=120
    )
    console.print(renderable)
    output = buf.getvalue()
    assert not has_executable_controls(output)
    assert "\x9b" not in output and "\x9d" not in output
    assert "SECRET" not in output  # clipboard payload gone
    assert "http://evil" not in output  # OSC 8 target gone
    assert "stdout" in output and "tail" in output
    assert "result" in output and "end" in output


def test_notebook_cli_is_neutralised(runner, tmp_path):
    path = _malicious_notebook(tmp_path)
    result = runner.invoke(main, [str(path), "--ipynb"])
    assert result.exit_code == 0, result.output
    assert not has_executable_controls(result.output)
    assert "\x9b" not in result.output
    assert "SECRET" not in result.output
    assert "http://evil" not in result.output


def test_notebook_forensic_markers(runner, tmp_path):
    path = _malicious_notebook(tmp_path)
    result = runner.invoke(main, [str(path), "--ipynb", "--escape-forensics"])
    assert result.exit_code == 0, result.output
    assert "␛]52;c;SECRET" in result.output
    assert "␛[?1049h" in result.output
    assert not has_executable_controls(result.output)


def test_notebook_html_export_clean(runner, tmp_path):
    path = _malicious_notebook(tmp_path)
    out_html = tmp_path / "nb.html"
    result = runner.invoke(main, [str(path), "--ipynb", "--export-html", str(out_html)])
    assert result.exit_code == 0, result.output
    html = out_html.read_text(encoding="utf8")
    assert not has_executable_controls(html)
    assert "SECRET" not in html
