"""Unit tests for the terminal-control sanitizer (parser -> IR -> policy)."""

import re

import pytest

from rich_cli.terminal_safety import (
    POLICY_RAW,
    POLICY_SAFE,
    POLICY_STRICT,
    CsiOp,
    OscOp,
    StringOp,
    EscapeOp,
    ControlOp,
    TerminalSanitizer,
    decide,
    DROP,
    KEEP,
    parse_terminal,
    sanitize_string,
    sanitize_to_text,
    visualize,
)


# ---------------------------------------------------------------------------
# Legitimate colour / formatting must survive (no over-filtering)
# ---------------------------------------------------------------------------


def test_plain_text_untouched():
    assert sanitize_string("just text") == "just text"
    assert sanitize_to_text("just text").plain == "just text"


@pytest.mark.parametrize(
    "sequence",
    [
        "\x1b[31m",
        "\x1b[1;31m",
        "\x1b[0m",
        "\x1b[38;5;200m",
        "\x1b[48;2;10;20;30m",
        "\x1b[38:2::10:20:30m",
        "\x1b[m",
    ],
)
def test_sgr_rendition_is_allow_listed(sequence):
    out = sanitize_string(f"a{sequence}b")
    assert sequence in out
    assert "\x1b" in out  # canonical SGR is intentionally retained


def test_sgr_becomes_style_spans_in_text():
    text = sanitize_to_text("\x1b[1;31mRED\x1b[0m")
    colors = {str(span.style) for span in text.spans if str(span.style)}
    assert any("color(1)" in style for style in colors)
    assert any("bold" in style for style in colors)
    assert text.plain == "RED"
    # OSC 8 style links must never appear
    assert_no_links(text)


def test_256_and_truecolor_supported():
    assert "color(200)" in {
        str(s.style) for s in sanitize_to_text("\x1b[38;5;200mX").spans if str(s.style)
    }
    rgb = {
        str(s.style)
        for s in sanitize_to_text("\x1b[38;2;1;2;3mX").spans
        if str(s.style)
    }
    assert "#010203" in rgb


def test_safe_whitespace_kept():
    assert sanitize_string("a\tb\nc\vd\fe") == "a\tb\nc\nd\ne"
    assert sanitize_string("one\r\ntwo") == "one\ntwo"


# ---------------------------------------------------------------------------
# Attack corpus: every dangerous primitive must be stripped
# ---------------------------------------------------------------------------


ATTACKS = [
    pytest.param("\x1b]8;;http://evil.example\x1b\\CLICK\x1b]8;;\x1b\\", id="osc8-st"),
    pytest.param("\x1b]8;;http://evil.example\x07CLICK\x1b]8;;\x07", id="osc8-bel"),
    pytest.param("\x1b]52;c;QUJDRA\x07", id="osc52-bel"),
    pytest.param("\x1b]52;c;QUJDRA\x1b\\", id="osc52-st"),
    pytest.param("\x1b]0;fake title\x07", id="osc0-title-bel"),
    pytest.param("\x1b]2;fake title\x1b\\", id="osc2-title-st"),
    pytest.param("\x1b[2J\x1b[3J\x1b[H", id="erase-screen"),
    pytest.param("\x1b[10;10H\x1b[5A\x1b[2B", id="cursor-move"),
    pytest.param("\x1b[?1049h\x1b[?1049l", id="alt-screen"),
    pytest.param("\x1b[?25l\x1b[?2004h", id="cursor-bracketed"),
    pytest.param("\x1b[6n\x1b[c\x1b[>c", id="device-query"),
    pytest.param("\x1b[s\x1b[u\x1b[2K", id="cursor-save-erase"),
    pytest.param("\x1bP0+rKEYMACRO\x1b\\", id="dcs-7bit"),
    pytest.param("\x1b_GRAPHICS\x1b\\", id="apc-7bit"),
    pytest.param("\x1b^PMDATA\x1b\\", id="pm-7bit"),
    pytest.param("\x1bXSOSDATA\x1b\\", id="sos-7bit"),
    pytest.param("\x9b2J", id="csi-8bit"),
    pytest.param("\x9d52;c;QUJDRA\x07", id="osc52-8bit"),
    pytest.param("\x90DCSDATA\x9c", id="dcs-8bit"),
    pytest.param("\x9fAPCDATA\x9c", id="apc-8bit"),
    pytest.param("\x1bc", id="ris-reset"),
    pytest.param("\x1b7\x1b8\x1b=\x1b>", id="esc-cursor-keypad"),
    pytest.param("\x1b(B\x1b)B", id="charset-designation"),
    pytest.param("\x07\x08\x00\x7f", id="bel-bs-nul-del"),
    pytest.param("hidden\r   xx", id="cr-rewrite"),
    pytest.param("\x1b[?m", id="private-marker-not-sgr"),
    pytest.param("\x1b[31m\x1b[2J\x1b[0m", id="sgr-then-erase"),
]

FORBIDDEN_C1 = ("\x9b", "\x9d", "\x90", "\x9f", "\x98", "\x9e", "\x9c", "\x07")

# String output may retain canonical SGR, but nothing else introduced by ESC.
_SGR_SEQUENCE = re.compile(r"\x1b\[[0-9:;]*m")


def assert_only_sgr_escapes(text):
    """Every ESC in *text* must start a canonical SGR rendition sequence."""
    stripped = _SGR_SEQUENCE.sub("", text)
    assert "\x1b" not in stripped


def assert_no_control_bytes(text):
    assert "\x1b" not in text
    for forbidden in FORBIDDEN_C1:
        assert forbidden not in text


def assert_no_links(text):
    assert not any(getattr(span.style, "link", None) for span in text.spans)


@pytest.mark.parametrize("attack", ATTACKS)
def test_attack_stripped_safe_string(attack):
    assert_only_sgr_escapes(sanitize_string(attack))


@pytest.mark.parametrize("attack", ATTACKS)
def test_attack_stripped_safe_text(attack):
    text = sanitize_to_text(attack)
    assert_no_control_bytes(text.plain)
    assert_no_links(text)


def test_osc8_hyperlink_does_not_become_link():
    text = sanitize_to_text("\x1b]8;;http://evil\x1b\\spam\x1b]8;;\x1b\\")
    assert text.plain == "spam"
    assert_no_links(text)


def test_osc52_clipboard_payload_not_in_text_output():
    text = sanitize_to_text("\x1b]52;c;SECRET\x07visible")
    assert "SECRET" not in text.plain
    assert text.plain == "visible"


def test_crlf_collapses_and_lone_cr_dropped():
    assert sanitize_string("a\r\nb") == "a\nb"
    # CR rewrite cannot hide preceding text
    assert sanitize_string("VISIBLE\r  ") == "VISIBLE  "


def test_nel_variants_become_newline():
    assert sanitize_string("a\x1bEb") == "a\nb"
    assert sanitize_string("a\x85b") == "a\nb"


# ---------------------------------------------------------------------------
# Truncated / malformed sequences
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "truncated",
    [
        "\x1b",
        "\x1b[",
        "\x1b[2",
        "\x1b[31",
        "\x1b]",
        "\x1b]52;c;ab",
        "\x1bP0+r",
        "\x1b_",
        "\x1b(",
        "\x1b[31m\x1b",
    ],
)
def test_truncated_sequences_never_leak_esc(truncated):
    assert_only_sgr_escapes(sanitize_string(truncated))
    assert_no_control_bytes(sanitize_to_text(truncated).plain)


def test_truncated_forensic_markers():
    assert sanitize_string("\x1b[2", forensics=True).endswith("␛[2…")
    assert sanitize_string("\x1b", forensics=True).endswith("␛…")
    assert sanitize_string("\x1b]52;c;ab", forensics=True).endswith("␛]52;c;ab…")


def test_doubled_and_interrupted_escapes():
    # ESC ESC [ 31 m -- the abandoned ESC must not confuse the parser
    out = sanitize_string("\x1b\x1b[31mX\x1b[0m")
    assert "X" in out
    assert "\x1b[31m" in out
    # CSI aborted by ESC, followed by an erase
    assert "\x1b" not in sanitize_string("\x1b[31\x1b[2Jx")


def test_osc_body_cannot_embed_executed_controls():
    # ESC inside an OSC body cannot reopen an executable CSI: the string is
    # aborted/stripped and no raw control byte reaches the output.
    attack = "\x1b]0;\x1b[2Jtitle\x07after"
    out = sanitize_string(attack)
    assert_only_sgr_escapes(out)
    assert out.endswith("after")
    assert "[2J" in out  # payload may remain as *visible* text, never executed
    marker = sanitize_string(attack, forensics=True)
    assert "␛]0;…" in marker  # the aborted OSC is shown incomplete
    assert "␇" in marker  # the dangling BEL too


# ---------------------------------------------------------------------------
# Cross-chunk splicing resistance
# ---------------------------------------------------------------------------

SPLIT_SAMPLES = [
    "x\x1b]52;c;AB\x07y",
    "\x1b[31mred\x1b[0m tail",
    "\x9b2Jz",
    "abc\r\ndef",
    "p\x1b]8;;http://evil\x1b\\q\x1b[?1049hr",
    "\x1b" + "]52;spli" + "t\x07!",
]


@pytest.mark.parametrize("sample", SPLIT_SAMPLES)
def test_chunk_split_cannot_smuggle_sequences(sample):
    whole_string = sanitize_string(sample)
    whole_text = sanitize_to_text(sample).plain
    for index in range(len(sample) + 1):
        stream = TerminalSanitizer()
        stream.feed(sample[:index])
        stream.feed(sample[index:])
        stream.finish()
        assert stream.to_string() == whole_string
        text_stream = TerminalSanitizer()
        text_stream.feed(sample[:index])
        text_stream.feed(sample[index:])
        text_stream.finish()
        assert text_stream.to_text().plain == whole_text


def test_many_small_chunks_equivalent():
    sample = "\x1b[31mA\x1b[0m" "\x1b]52;c;X\x07" "\x9b2J" "plain" "\x1b]0;t\x1b\\"
    whole = sanitize_string(sample)
    stream = TerminalSanitizer()
    for char in sample:
        stream.feed(char)
    stream.finish()
    assert stream.to_string() == whole
    # SGR kept, clipboard/title/erase content gone
    assert "\x1b[31m" in whole
    assert "]52" not in whole
    assert "]0;t" not in whole
    assert "\x9b" not in whole


# ---------------------------------------------------------------------------
# Forensic mode
# ---------------------------------------------------------------------------


def test_forensic_markers_are_inert_and_visible():
    shown = sanitize_string("\x1b[2J\x1b]52;c;A\x07", forensics=True)
    assert "\x1b" not in shown  # no real ESC byte
    assert "␛[2J" in shown
    assert "␛]52;c;A␇" in shown


def test_forensic_text_markers_are_styled():
    text = sanitize_to_text("\x1b]52;c;A\x07", forensics=True)
    styled_markers = [
        span for span in text.spans if str(span.style) in ("bold red", "yellow", "dim")
    ]
    assert styled_markers


def test_forensic_marker_for_each_op_kind():
    from rich_cli.terminal_safety import PrintOp

    ops = parse_terminal("\x1b[2J\x1b]0;x\x07\x1bPab\x1b\\\x1bc\x07")
    markers = [visualize(op) for op in ops if not isinstance(op, PrintOp)]
    assert any(m.startswith("␛[2J") for m in markers)
    assert any("␛]0;x" in m for m in markers)
    assert any(m.startswith("␛Pab") for m in markers)
    assert "␛c" in markers
    assert "␇" in markers


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------


def test_strict_policy_strips_sgr():
    sample = "\x1b[31mRED\x1b[0m"
    assert "\x1b[31m" not in sanitize_string(sample, POLICY_STRICT)
    text = sanitize_to_text(sample, POLICY_STRICT)
    meaningful = [span for span in text.spans if str(span.style)]
    assert meaningful == []
    assert text.plain == "RED"


def test_strict_keeps_plain_text_and_newlines():
    assert sanitize_string("a\tb\nc", POLICY_STRICT) == "a\tb\nc"


def test_allow_sgr_false_strips_sgr_under_safe_policy():
    sample = "\x1b[31mRED\x1b[0m \x1b[2Jgone"
    out = sanitize_string(sample, allow_sgr=False)
    assert "\x1b" not in out
    assert "RED" in out and "gone" in out
    text = sanitize_to_text(sample, allow_sgr=False)
    meaningful = [span for span in text.spans if str(span.style)]
    assert meaningful == []
    assert text.plain == "RED gone"


def test_allow_sgr_false_keeps_whitespace():
    assert sanitize_string("a\tb\nc", allow_sgr=False) == "a\tb\nc"


def test_allow_sgr_false_ignored_by_raw_policy():
    sample = "\x1b[31mRED\x1b[0m"
    assert sanitize_string(sample, POLICY_RAW, allow_sgr=False) == sample


def test_raw_policy_is_byte_faithful_passthrough():
    sample = "\x1b]52;c;A\x07\x1b[31mx\x1b[0m"
    assert sanitize_string(sample, POLICY_RAW) == sample
    # Legacy Text behaviour under raw policy
    assert sanitize_to_text(sample, POLICY_RAW).__class__.__name__ == "Text"


def test_unknown_policy_rejected():
    with pytest.raises(ValueError):
        sanitize_string("x", "bogus")
    with pytest.raises(ValueError):
        TerminalSanitizer(policy="bogus")


# ---------------------------------------------------------------------------
# IR typing
# ---------------------------------------------------------------------------


def test_parser_emits_typed_ops():
    ops = parse_terminal("\x1b[31m\x1b[2J\x1b]0;t\x07\x1bPab\x1b\\\x1bc\x07ab")
    kinds = [op.__class__ for op in ops]
    assert CsiOp in kinds
    assert OscOp in kinds
    assert StringOp in kinds
    assert EscapeOp in kinds
    assert ControlOp in kinds
    sgr = next(op for op in ops if isinstance(op, CsiOp) and op.final == "m")
    erase = next(op for op in ops if isinstance(op, CsiOp) and op.final == "J")
    assert sgr.params == "31"
    assert erase.params == "2"
    osc = next(op for op in ops if isinstance(op, OscOp))
    assert osc.code == 0
    assert osc.payload == "0;t"
    dcs = next(op for op in ops if isinstance(op, StringOp))
    assert dcs.kind == "dcs"
    assert dcs.body == "ab"


def test_policy_default_deny_for_unknown_csi():
    op = CsiOp("", "999", "", "z")
    assert decide(POLICY_SAFE, op).action == DROP
    assert decide(POLICY_STRICT, op).action == DROP


def test_decisions_are_default_deny():
    # Every non-print, non-whitespace op in a corpus is dropped by safe unless SGR
    sample = "\x1b[?9999h\x1b[z\x1bQ\x1b]1337;x\x07\x9b?q"
    for op in parse_terminal(sample):
        decision = decide(POLICY_SAFE, op)
        if isinstance(op, CsiOp) and op.final == "m":
            assert decision.action == KEEP
        else:
            assert decision.action == DROP


# ---------------------------------------------------------------------------
# Resource limits
# ---------------------------------------------------------------------------


def test_oversized_csi_is_bounded_and_neutralised():
    huge = "\x1b[" + "1;" * 5000 + "H"
    out = sanitize_string(huge + "after")
    assert out == "after"
    assert "\x1b" not in out


def test_huge_osc_body_is_capped_in_marker():
    body = "X" * 10000
    shown = sanitize_string(f"\x1b]0;{body}\x07", forensics=True)
    # marker payload is preview-capped
    assert "X" * 100 not in shown
