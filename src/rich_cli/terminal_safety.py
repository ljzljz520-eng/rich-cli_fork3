"""Terminal escape-sequence sanitization.

Rich CLI previews *untrusted* files (and URL / stdin / notebook content). Terminal
control protocols let such content do more than colour text: OSC 8 injects
hyperlink targets, OSC 52 writes the clipboard, OSC 0/2 rewrites the window title,
CSI cursor / erase sequences hide or overwrite previously shown text, DCS / APC may
install key macros or draw inline images, and device-status queries can make the
emulator talk back to the application.

Defence in depth works like this:

1.  Every byte of untrusted text is run through :class:`TerminalParser`, a single
    state machine that understands both the 7-bit (``ESC [``) and 8-bit (``U+009B``)
    encodings, BEL / ST string terminators, nested and aborted sequences, and
    sequences split across streaming chunks.  It emits a typed intermediate
    representation of :class:`TerminalOp` values -- raw control bytes never reach a
    rendering backend directly.
2.  An allow-list :class:`Policy` decides, for each op, whether it is kept or
    stripped.  The default ``safe`` policy keeps only SGR rendition and safe
    whitespace; ``strict`` additionally drops SGR; ``raw`` disables filtering.
3.  Every backend consumes the *same* IR/policy output.  Console, pager, HTML and
    SVG rendering all build on :func:`sanitize_to_text` (styled Rich text, no raw
    ESC) or :func:`sanitize_string` (canonical text in which only allow-listed SGR
    sequences have been re-serialised) -- there is no parsing bypass.

Stripped sequences are either dropped silently or, in *forensic* mode, rendered as
visible inline markers (e.g. ``␛[2J`` / ``␛]52;c;AAAA␇``) so that an attempt to
hide content can be seen instead of disappearing without trace.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union

from rich.ansi import AnsiDecoder
from rich.text import Text

# ---------------------------------------------------------------------------
# Policy names
# ---------------------------------------------------------------------------

POLICY_SAFE = "safe"
"""Keep SGR rendition and safe whitespace; strip everything else (default)."""

POLICY_STRICT = "strict"
"""Like ``safe`` but strip SGR as well -- plain text plus newlines/tabs."""

POLICY_RAW = "raw"
"""No filtering.  Provided as an explicit, labelled escape hatch only."""

POLICIES = (POLICY_SAFE, POLICY_STRICT, POLICY_RAW)

# ---------------------------------------------------------------------------
# C0 / C1 constants
# ---------------------------------------------------------------------------

NUL = 0x00
BEL = 0x07
BS = 0x08
TAB = 0x09
LF = 0x0A
VT = 0x0B
FF = 0x0C
CR = 0x0D
CAN = 0x18
SUB = 0x1A
ESC = 0x1B
DEL = 0x7F

# 8-bit C1 control codes
C1_DCS = 0x90
C1_SOS = 0x98
C1_APC = 0x9F
C1_CSI = 0x9B
C1_ST = 0x9C
C1_OSC = 0x9D
C1_NEL = 0x85

# Parser resource limits, so a hostile payload cannot force unbounded buffering.
MAX_CSI_PARAMS = 512
MAX_STRING_BODY = 4096
MAX_MARKER_BODY = 48

# Parser states
_GROUND = "ground"
_ESCAPE = "escape"
_CSI = "csi"
_STRING = "string"
_STRING_ESC = "string_esc"

_STRING_KINDS = ("osc", "dcs", "apc", "pm", "sos")
_TERM_BEL = "bel"
_TERM_ST7 = "st7"  # ESC \
_TERM_ST8 = "st8"  # U+009C


# ---------------------------------------------------------------------------
# Typed intermediate representation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PrintOp:
    """A run of ordinary printable text (contains no control bytes)."""

    text: str


@dataclass(frozen=True)
class ControlOp:
    """A standalone C0 (0x00-0x1F), DEL (0x7F) or single-byte C1 (0x80-0x9F)."""

    code: int


@dataclass(frozen=True)
class EscapeOp:
    """A two-plus byte escape sequence, e.g. ``ESC ( B``, ``ESC c`` or ``ESC E``."""

    intermediate: str
    final: str
    complete: bool = True


@dataclass(frozen=True)
class CsiOp:
    """A control sequence introduced by ``ESC [`` / ``U+009B``.

    ``final == "m"`` with no private/intermediate bytes is an SGR rendition
    sequence; every other CSI moves the cursor, erases content, changes modes or
    queries the device.
    """

    private: str
    params: str
    intermediate: str
    final: str
    complete: bool = True
    oversized: bool = False


@dataclass(frozen=True)
class OscOp:
    """An operating-system command string (title, clipboard, hyperlink, ...)."""

    code: Optional[int]
    payload: str
    complete: bool = True
    terminator: Optional[str] = None


@dataclass(frozen=True)
class StringOp:
    """A DCS / APC / PM / SOS string terminated by ST."""

    kind: str
    body: str
    complete: bool = True
    terminator: Optional[str] = None


TerminalOp = Union[PrintOp, ControlOp, EscapeOp, CsiOp, OscOp, StringOp]


# ---------------------------------------------------------------------------
# Stateful parser
# ---------------------------------------------------------------------------


class TerminalParser:
    """Parse text into :class:`TerminalOp` values.

    The parser is stateful on purpose.  Call :meth:`feed` repeatedly with stream
    chunks and :meth:`finish` at end of stream; pending sequences are carried
    between calls, so a control sequence split across two chunks (``ESC`` in one
    chunk, ``]52;...`` in the next) is recognised and neutralised rather than
    slipping through as two harmless looking halves.
    """

    def __init__(self) -> None:
        self._state = _GROUND
        self._esc_intermediate: List[str] = []
        self._csi_private = ""
        self._csi_params: List[str] = []
        self._csi_intermediate: List[str] = []
        self._csi_oversized = False
        self._str_kind = "osc"
        self._str_body: List[str] = []
        self._str_size = 0
        self._str_oversized = False
        self._cr_pending = False
        self._print_buffer: List[str] = []

    # -- public API ---------------------------------------------------------

    def feed(self, text: str) -> List[TerminalOp]:
        """Parse one chunk, returning any completed ops.

        State for an incomplete trailing sequence is retained for the next call.
        """
        ops: List[TerminalOp] = []
        append = ops.append
        for ch in text:
            code = ord(ch)
            if self._state == _GROUND:
                self._ground(code, ch, append)
            elif self._state == _ESCAPE:
                self._escape(code, ch, append)
            elif self._state == _CSI:
                self._csi(code, ch, append)
            elif self._state == _STRING:
                self._string(code, ch, append)
            else:  # _STRING_ESC
                self._string_esc(code, ch, append)
        if self._state == _GROUND:
            self._flush_print(append)
        return ops

    def pending(self) -> bool:
        """Whether the parser is holding an unterminated sequence or buffer."""
        return self._state != _GROUND or bool(self._print_buffer) or self._cr_pending

    def finish(self) -> List[TerminalOp]:
        """Flush end-of-stream state.

        A dangling CR is reported on its own; any unterminated escape / CSI /
        string sequence is emitted as an *incomplete* op, so the policy can strip
        or visualise it instead of leaking a bare ESC byte.
        """
        ops: List[TerminalOp] = []
        append = ops.append
        self._flush_print(append)
        if self._cr_pending:
            self._cr_pending = False
            append(ControlOp(CR))
        if self._state == _ESCAPE:
            append(EscapeOp("".join(self._esc_intermediate), "", complete=False))
        elif self._state == _CSI:
            append(
                CsiOp(
                    self._csi_private,
                    "".join(self._csi_params),
                    "".join(self._csi_intermediate),
                    "",
                    complete=False,
                    oversized=self._csi_oversized,
                )
            )
        elif self._state in (_STRING, _STRING_ESC):
            self._emit_string(append, complete=False, terminator=None)
        self._reset_sequence_state()
        self._state = _GROUND
        return ops

    # -- state handlers -----------------------------------------------------

    def _ground(self, code: int, ch: str, emit) -> None:
        if code == CR:
            # Defer the CR: CRLF must collapse to a single safe newline, while a
            # lone CR (a screen-rewrite primitive) is reported separately.
            self._flush_print(emit)
            self._cr_pending = True
        elif code == ESC:
            self._flush_pending_cr(emit)
            self._flush_print(emit)
            self._state = _ESCAPE
            self._esc_intermediate = []
        elif 0x80 <= code <= 0x9F:
            self._flush_pending_cr(emit)
            self._flush_print(emit)
            self._start_c1(code, emit)
        elif code < 0x20 or code == DEL:
            self._flush_pending_cr(emit)
            self._flush_print(emit)
            emit(ControlOp(code))
        else:
            self._flush_pending_cr(emit)
            self._print_buffer.append(ch)

    def _escape(self, code: int, ch: str, emit) -> None:
        if code == ESC:
            # ESC ESC -- restart the sequence; the abandoned ESC is inert.
            self._esc_intermediate = []
        elif 0x20 <= code <= 0x2F:
            if len(self._esc_intermediate) < 4:
                self._esc_intermediate.append(ch)
        elif 0x30 <= code <= 0x7E:
            intermediate = "".join(self._esc_intermediate)
            if ch == "[":
                self._begin_csi()
            elif ch == "]":
                self._begin_string("osc")
            elif ch == "P":
                self._begin_string("dcs")
            elif ch == "X":
                self._begin_string("sos")
            elif ch == "^":
                self._begin_string("pm")
            elif ch == "_":
                self._begin_string("apc")
            else:
                emit(EscapeOp(intermediate, ch))
                self._state = _GROUND
        elif code == CAN or code == SUB:
            # Abort the in-progress sequence.
            self._state = _GROUND
        elif 0x80 <= code <= 0x9F:
            # Abandon the ESC and process the 8-bit control in ground state.
            self._state = _GROUND
            self._start_c1(code, emit)
        elif code >= 0xA0:
            # A printable byte cancels the pending escape; the bare ESC is inert.
            self._state = _GROUND
            self._print_buffer.append(ch)
        # Other C0 controls are consumed without leaving escape state.

    def _csi(self, code: int, ch: str, emit) -> None:
        if 0x30 <= code <= 0x3F:
            if not self._csi_params and not self._csi_private and ch in "?<>=":
                self._csi_private = ch
            elif self._csi_size() < MAX_CSI_PARAMS:
                self._csi_params.append(ch)
            else:
                self._csi_oversized = True
        elif 0x20 <= code <= 0x2F:
            if self._csi_size() < MAX_CSI_PARAMS:
                self._csi_intermediate.append(ch)
            else:
                self._csi_oversized = True
        elif 0x40 <= code <= 0x7E:
            emit(
                CsiOp(
                    self._csi_private,
                    "".join(self._csi_params),
                    "".join(self._csi_intermediate),
                    ch,
                    oversized=self._csi_oversized,
                )
            )
            self._reset_sequence_state()
            self._state = _GROUND
        elif code == ESC:
            self._abort_csi(emit)
            self._state = _ESCAPE
            self._esc_intermediate = []
        elif 0x80 <= code <= 0x9F:
            self._abort_csi(emit)
            self._start_c1(code, emit)
        elif code >= 0xA0:
            # Invalid inside CSI: abort and treat the byte as text.
            self._abort_csi(emit)
            self._print_buffer.append(ch)
        elif code == CAN or code == SUB:
            self._abort_csi(emit)
        # Remaining C0 / DEL are consumed without leaving CSI state.

    def _string(self, code: int, ch: str, emit) -> None:
        if self._str_kind == "osc" and code == BEL:
            self._emit_string(emit, complete=True, terminator=_TERM_BEL)
            self._state = _GROUND
        elif code == C1_ST:
            self._emit_string(emit, complete=True, terminator=_TERM_ST8)
            self._state = _GROUND
        elif code == ESC:
            self._state = _STRING_ESC
        elif code == CAN or code == SUB:
            self._emit_string(emit, complete=False, terminator=None)
            self._state = _GROUND
        elif 0x80 <= code <= 0x9F:
            # A second C1 opener never nests: close the current string first.
            self._emit_string(emit, complete=False, terminator=None)
            self._state = _GROUND
            self._start_c1(code, emit)
        else:
            self._append_string_body(ch)

    def _string_esc(self, code: int, ch: str, emit) -> None:
        if ch == "\\":
            self._emit_string(emit, complete=True, terminator=_TERM_ST7)
            self._state = _GROUND
        elif code == ESC:
            # ESC ESC \x00 -- stay put waiting for the ST final byte.
            return
        elif code == CAN or code == SUB:
            self._emit_string(emit, complete=False, terminator=None)
            self._state = _GROUND
        else:
            # The ESC was not an ST.  Some emulators re-execute the following
            # byte together with the ESC (ESC [ could reopen as CSI); we instead
            # close the (aborted, stripped) string and reprocess only the byte in
            # ground state, so an opener "nested" inside a string body can never
            # be executed here.
            self._emit_string(emit, complete=False, terminator=None)
            self._state = _GROUND
            if code >= 0x20 or code in (TAB, LF, VT, FF, CR):
                self._ground(code, ch, emit)

    # -- helpers ------------------------------------------------------------

    def _start_c1(self, code: int, emit) -> None:
        if code == C1_CSI:
            self._begin_csi()
        elif code == C1_OSC:
            self._begin_string("osc")
        elif code == C1_DCS:
            self._begin_string("dcs")
        elif code == C1_SOS:
            self._begin_string("sos")
        elif code == C1_APC:
            self._begin_string("apc")
        elif code == 0x9E:
            self._begin_string("pm")
        elif code == C1_ST:
            emit(EscapeOp("", "\\"))
        else:
            emit(ControlOp(code))

    def _begin_csi(self) -> None:
        self._state = _CSI
        self._csi_private = ""
        self._csi_params = []
        self._csi_intermediate = []
        self._csi_oversized = False

    def _abort_csi(self, emit) -> None:
        emit(
            CsiOp(
                self._csi_private,
                "".join(self._csi_params),
                "".join(self._csi_intermediate),
                "",
                complete=False,
                oversized=self._csi_oversized,
            )
        )
        self._reset_sequence_state()

    def _begin_string(self, kind: str) -> None:
        self._state = _STRING
        self._str_kind = kind
        self._str_body = []
        self._str_size = 0
        self._str_oversized = False

    def _append_string_body(self, ch: str) -> None:
        if self._str_size < MAX_STRING_BODY:
            self._str_body.append(ch)
            self._str_size += 1
        else:
            self._str_oversized = True

    def _emit_string(self, emit, complete: bool, terminator: Optional[str]) -> None:
        body = "".join(self._str_body)
        if self._str_oversized:
            body += "…"
        if self._str_kind == "osc":
            code: Optional[int] = None
            digits = ""
            for ch in body:
                if "0" <= ch <= "9":
                    digits += ch
                else:
                    break
            if digits and (len(digits) == len(body) or body[len(digits)] == ";"):
                code = int(digits)
            emit(OscOp(code, body, complete=complete, terminator=terminator))
        else:
            emit(
                StringOp(self._str_kind, body, complete=complete, terminator=terminator)
            )
        self._reset_sequence_state()

    def _csi_size(self) -> int:
        return len(self._csi_params) + len(self._csi_intermediate)

    def _flush_pending_cr(self, emit) -> None:
        if self._cr_pending:
            self._cr_pending = False
            emit(ControlOp(CR))

    def _flush_print(self, emit) -> None:
        if self._print_buffer:
            emit(PrintOp("".join(self._print_buffer)))
            self._print_buffer = []

    def _reset_sequence_state(self) -> None:
        self._esc_intermediate = []
        self._csi_private = ""
        self._csi_params = []
        self._csi_intermediate = []
        self._csi_oversized = False
        self._str_kind = "osc"
        self._str_body = []
        self._str_size = 0
        self._str_oversized = False


def parse_terminal(text: str) -> List[TerminalOp]:
    """Convenience: parse an entire string into a list of ops."""
    parser = TerminalParser()
    ops = parser.feed(text)
    ops.extend(parser.finish())
    return ops


# ---------------------------------------------------------------------------
# Allow-list policy
# ---------------------------------------------------------------------------

KEEP = "keep"
DROP = "drop"

INFO = "info"
WARN = "warn"
DANGER = "danger"


@dataclass(frozen=True)
class Decision:
    action: str
    severity: str


def sgr_params(op: CsiOp) -> Optional[str]:
    """Return canonical SGR parameter text if *op* is a plain SGR sequence."""
    if op.final != "m" or not op.complete or op.oversized:
        return None
    if op.private or op.intermediate:
        return None
    for ch in op.params:
        if not ("0" <= ch <= "9" or ch in ";:"):
            return None
    return op.params


_CURSOR_FINALS = frozenset("ABCDdGHf")
_ERASE_FINALS = frozenset("JKP@LMXST")
_MODE_FINALS = frozenset("hl")
_DEVICE_FINALS = frozenset("ncit")
_DANGEROUS_OSC = frozenset({52, 9, 777, 99})
_TITLE_OSC = frozenset({0, 2})
_QUERY_OSC_RANGES = (range(10, 18), range(110, 118))


def _csi_severity(op: CsiOp) -> str:
    if not op.complete:
        return WARN
    if op.final in _DEVICE_FINALS:
        return DANGER
    if op.final in _MODE_FINALS and op.private:
        return DANGER
    if op.final in _CURSOR_FINALS or op.final in _ERASE_FINALS:
        return WARN
    if op.final in ("s", "u"):
        return WARN
    if op.final in _MODE_FINALS:
        return WARN
    return INFO


def _osc_severity(op: OscOp) -> str:
    if not op.complete:
        return WARN
    code = op.code
    if code in _DANGEROUS_OSC:
        return DANGER
    if code in _TITLE_OSC or code == 8:
        return WARN
    if code is not None and any(code in rng for rng in _QUERY_OSC_RANGES):
        return DANGER
    return INFO


def _string_severity(op: StringOp) -> str:
    if not op.complete:
        return WARN
    if op.kind == "dcs":
        return DANGER
    if op.kind == "apc":
        return WARN
    return INFO


def _escape_severity(op: EscapeOp) -> str:
    if not op.complete:
        return WARN
    if op.final == "c":
        return DANGER
    if op.final in ("7", "8", "D", "M"):
        return WARN
    return INFO


def _control_severity(op: ControlOp) -> str:
    return WARN if op.code == BEL else INFO


def _decide_safe(op: TerminalOp) -> Decision:
    if isinstance(op, PrintOp):
        return Decision(KEEP, INFO)
    if isinstance(op, ControlOp):
        if op.code in (LF, VT, FF, TAB, C1_NEL):
            return Decision(KEEP, INFO)
        return Decision(DROP, _control_severity(op))
    if isinstance(op, EscapeOp):
        if op.complete and not op.intermediate and op.final == "E":
            # NEL -- a safe newline.
            return Decision(KEEP, INFO)
        return Decision(DROP, _escape_severity(op))
    if isinstance(op, CsiOp):
        if sgr_params(op) is not None:
            return Decision(KEEP, INFO)
        return Decision(DROP, _csi_severity(op))
    if isinstance(op, OscOp):
        return Decision(DROP, _osc_severity(op))
    # StringOp
    return Decision(DROP, _string_severity(op))


def _decide_strict(op: TerminalOp) -> Decision:
    decision = _decide_safe(op)
    if isinstance(op, CsiOp) and sgr_params(op) is not None:
        return Decision(DROP, INFO)
    return decision


_DECIDERS = {
    POLICY_SAFE: _decide_safe,
    POLICY_STRICT: _decide_strict,
}


def decide(policy: str, op: TerminalOp) -> Decision:
    """Apply an allow-list policy to one IR op."""
    try:
        decider = _DECIDERS[policy]
    except KeyError:
        raise ValueError(
            f"unknown escape policy {policy!r}; expected one of {', '.join(POLICIES)}"
        )
    return decider(op)


# ---------------------------------------------------------------------------
# Forensic visualisation
# ---------------------------------------------------------------------------

ESC_GLYPH = "\u241b"  # ␛


def _control_glyph(code: int) -> str:
    if 0x00 <= code <= 0x1F:
        return chr(code + 0x2400)
    if code == DEL:
        return "\u2421"  # ␡
    return f"\u27e8U+{code:04X}\u27e9"  # ⟨U+XXXX⟩


def _preview(body: str) -> str:
    """Render a string body as inert, readable marker text."""
    chars: List[str] = []
    for ch in body[:MAX_MARKER_BODY]:
        code = ord(ch)
        if code < 0x20 or code == DEL:
            chars.append(_control_glyph(code))
        elif 0x80 <= code <= 0x9F:
            chars.append(f"\u27e8U+{code:04X}\u27e9")
        else:
            chars.append(ch)
    text = "".join(chars)
    if len(body) > MAX_MARKER_BODY:
        text += "…"
    return text


def _terminator_glyph(terminator: Optional[str]) -> str:
    if terminator == _TERM_BEL:
        return _control_glyph(BEL)
    if terminator == _TERM_ST7:
        return f"{ESC_GLYPH}\\"
    if terminator == _TERM_ST8:
        return "\u241c"  # ␜
    return "…"


def visualize(op: TerminalOp) -> str:
    """Render a stripped op as an inert, human-readable forensic marker."""
    if isinstance(op, ControlOp):
        return _control_glyph(op.code)
    if isinstance(op, EscapeOp):
        suffix = "" if op.complete else "…"
        return f"{ESC_GLYPH}{op.intermediate}{op.final}{suffix}"
    if isinstance(op, CsiOp):
        suffix = "" if op.complete else "…"
        params = op.params + ("…" if op.oversized else "")
        return f"{ESC_GLYPH}[{op.private}{params}{op.intermediate}{op.final}{suffix}"
    if isinstance(op, OscOp):
        return f"{ESC_GLYPH}]{_preview(op.payload)}{_terminator_glyph(op.terminator)}"
    if isinstance(op, StringOp):
        initiator = {"dcs": "P", "apc": "_", "pm": "^", "sos": "X"}.get(op.kind, "X")
        return (
            f"{ESC_GLYPH}{initiator}{_preview(op.body)}"
            f"{_terminator_glyph(op.terminator)}"
        )
    return ""


_SEVERITY_STYLES = {
    DANGER: "bold red",
    WARN: "yellow",
    INFO: "dim",
}

# ---------------------------------------------------------------------------
# Sanitization -- the single chokepoint all backends consume
# ---------------------------------------------------------------------------


class TerminalSanitizer:
    """Streaming sanitizer.  Feed chunks, then collect Text or a filtered string.

    Both output forms are produced from the *same* IR ops and the *same* policy
    decisions, so console, pager and export backends can never disagree about what
    was filtered.
    """

    def __init__(
        self,
        policy: str = POLICY_SAFE,
        forensics: bool = False,
        allow_sgr: bool = True,
    ) -> None:
        if policy not in POLICIES:
            raise ValueError(
                f"unknown escape policy {policy!r}; expected one of {', '.join(POLICIES)}"
            )
        self.policy = policy
        self.forensics = forensics
        # SGR is allow-listed by policy, but a code lexer (Pygments) tokenizes
        # the bytes of an SGR sequence into separately styled tokens, which
        # breaks the sequence apart and leaks a stray ESC.  Renderers that feed
        # the text to a code lexer therefore pass ``allow_sgr=False``; the lexer
        # supplies its own colours.  The RAW policy ignores this and passes
        # bytes through unchanged, as advertised.
        self.allow_sgr = allow_sgr
        self._parser = TerminalParser()
        self._ops: List[TerminalOp] = []
        self._raw_parts: List[str] = []

    def feed(self, chunk: str) -> "TerminalSanitizer":
        self._raw_parts.append(chunk)
        self._ops.extend(self._parser.feed(chunk))
        return self

    def finish(self) -> "TerminalSanitizer":
        self._ops.extend(self._parser.finish())
        return self

    def _resolved_ops(self) -> List[TerminalOp]:
        if self._parser.pending():
            self._ops.extend(self._parser.finish())
        return self._ops

    def to_string(self) -> str:
        """Canonical filtered string.

        Kept SGR sequences are re-serialised in canonical 7-bit form, so coloured
        log files keep working when displayed through a verbatim-text renderer;
        every other control sequence is absent (or replaced by a marker).
        """
        if self.policy == POLICY_RAW:
            # Labelled escape hatch: byte-for-byte passthrough, no filtering.
            return "".join(self._raw_parts)
        parts: List[str] = []
        for op in self._resolved_ops():
            decision = decide(self.policy, op)
            if decision.action == KEEP:
                if not self.allow_sgr and _is_sgr_op(op):
                    continue
                kept = _kept_string(op)
                if kept:
                    parts.append(kept)
            elif self.forensics:
                parts.append(visualize(op))
        return "".join(parts)

    def to_text(self) -> Text:
        """Sanitized Rich text: SGR rendered as style spans, no raw control bytes."""
        if self.policy == POLICY_RAW:
            # Legacy behaviour -- labelled as the unsafe policy it is.
            return Text.from_ansi("".join(self._raw_parts))
        decoder = AnsiDecoder()
        result = Text()
        for op in self._resolved_ops():
            decision = decide(self.policy, op)
            if decision.action != KEEP:
                if self.forensics:
                    result.append(
                        Text(
                            visualize(op),
                            style=_SEVERITY_STYLES[decision.severity],
                        )
                    )
                continue
            if not self.allow_sgr and _is_sgr_op(op):
                continue
            if isinstance(op, PrintOp):
                result.append(decoder.decode_line(op.text))
            elif isinstance(op, CsiOp):
                params = sgr_params(op)
                if params is not None:
                    decoder.decode_line(f"\x1b[{params}m")
            elif isinstance(op, ControlOp) and op.code in (LF, VT, FF, TAB, C1_NEL):
                result.append(decoder.decode_line("\n" if op.code != TAB else "\t"))
            elif isinstance(op, EscapeOp) and op.final == "E":
                result.append(decoder.decode_line("\n"))
        return result


def _is_sgr_op(op: TerminalOp) -> bool:
    return isinstance(op, CsiOp) and sgr_params(op) is not None


def _kept_string(op: TerminalOp) -> str:
    if isinstance(op, PrintOp):
        return op.text
    if isinstance(op, CsiOp):
        params = sgr_params(op)
        if params is not None:
            return f"\x1b[{params}m"
    elif isinstance(op, ControlOp):
        if op.code == TAB:
            return "\t"
        if op.code in (LF, VT, FF, C1_NEL):
            return "\n"
    elif isinstance(op, EscapeOp) and op.final == "E":
        return "\n"
    return ""


def sanitize_string(
    text: str,
    policy: str = POLICY_SAFE,
    forensics: bool = False,
    allow_sgr: bool = True,
) -> str:
    """Filter terminal control sequences out of *text*, returning canonical text.

    See :class:`TerminalSanitizer`.  Safe by default.  ``allow_sgr=False`` also
    drops SGR colour sequences (used when the text is fed to a code lexer that
    would split them into separate tokens).
    """
    sanitizer = TerminalSanitizer(
        policy=policy, forensics=forensics, allow_sgr=allow_sgr
    )
    sanitizer.feed(text)
    sanitizer.finish()
    return sanitizer.to_string()


def sanitize_to_text(
    text: str,
    policy: str = POLICY_SAFE,
    forensics: bool = False,
    allow_sgr: bool = True,
) -> Text:
    """Filter terminal control sequences, returning styled (but control-free) Text."""
    sanitizer = TerminalSanitizer(
        policy=policy, forensics=forensics, allow_sgr=allow_sgr
    )
    sanitizer.feed(text)
    sanitizer.finish()
    return sanitizer.to_text()
