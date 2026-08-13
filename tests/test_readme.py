"""Check the README against the code.

Every documentation error found in this project was in prose that no test read:
a benchmark caveat that outlived the bug it described, UI wording that survived
the engine being generalised, a paragraph saying nothing was measured on a long
run directly above the long-run measurements, and three hand-maintained counts
that went stale on the same afternoon. The tests caught none of it, because
tests read code.

So these read the README. Not its argument -- no test can check whether a
paragraph is honest -- but the parts of it that are mechanically true or false:
the commands it tells you to run, the files it points at, the constants it
quotes, the links between its own sections, and the counts in front of its
lists. Those are exactly the claims that rot silently, because changing the code
does not make them wrong-looking.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from flightrec.cli import build_parser

README = Path(__file__).resolve().parent.parent / "README.md"
ROOT = README.parent
TEXT = README.read_text(encoding="utf-8")


def code_blocks(language: str) -> list[str]:
    return re.findall(rf"```{language}\n(.*?)```", TEXT, re.S)


def subcommands() -> dict[str, object]:
    """The parser's subcommands, by name."""
    parser = build_parser()
    for action in parser._subparsers._group_actions:  # noqa: SLF001 - test-only
        if hasattr(action, "choices") and action.choices:
            return dict(action.choices)
    raise AssertionError("the CLI has no subcommands")


def flags_of(parser: object) -> set[str]:
    return {
        option
        for action in parser._actions  # noqa: SLF001 - test-only
        for option in action.option_strings
        if option.startswith("--")
    }


# --- the commands it tells you to run ----------------------------------------


def documented_invocations() -> list[str]:
    lines = []
    for block in code_blocks("bash"):
        for line in block.splitlines():
            line = line.split("#")[0].strip()
            if line.startswith("flightrec "):
                lines.append(line)
    return lines


def test_the_readme_shows_commands_to_run() -> None:
    """Guard the guard: these checks are worthless if they parse nothing."""
    assert len(documented_invocations()) >= 5


@pytest.mark.parametrize("invocation", documented_invocations())
def test_every_documented_command_exists(invocation: str) -> None:
    """`flightrec <thing>` in the README has to be a thing the CLI does."""
    available = subcommands()
    name = invocation.split()[1]

    assert name in available, f"{invocation!r}: no such command"


@pytest.mark.parametrize("invocation", documented_invocations())
def test_every_documented_flag_exists(invocation: str) -> None:
    """And every ``--flag`` shown with it has to be one that command accepts.

    This is the check that would have caught a rename: the code keeps working,
    the tests keep passing, and the README quietly tells people to type
    something that errors.
    """
    available = subcommands()
    parts = invocation.split()
    name = parts[1]
    if name not in available:
        pytest.skip("covered by the command test")

    accepted = flags_of(available[name])
    for flag in (p for p in parts[2:] if p.startswith("--")):
        assert flag.split("=")[0] in accepted, f"{invocation!r}: {flag} is not accepted"


# --- the files it points at ---------------------------------------------------


def documented_paths() -> list[str]:
    # Only paths with a directory in them: a bare "agent.py" in prose is a
    # description, "src/flightrec/demo/agent.py" is a claim.
    return sorted(set(re.findall(r"`([\w.\-]+(?:/[\w.\-]+)+\.\w+)`", TEXT)))


def test_the_readme_points_at_files() -> None:
    assert len(documented_paths()) >= 2


@pytest.mark.parametrize("path", documented_paths())
def test_every_documented_path_exists(path: str) -> None:
    """The README credited the demo agent to a sixteen-line CLI wrapper for a
    while. The path existed; it was just the wrong one -- so existence is the
    floor this can check, not the ceiling."""
    assert (ROOT / path).exists(), f"README points at {path}, which is not there"


# --- the constants it quotes --------------------------------------------------


def test_quoted_constants_match_the_code() -> None:
    """Numbers the prose states about tunables, checked against the tunables.

    Each of these was arrived at by measurement and argued for in a paragraph.
    Changing one without changing the paragraph leaves the argument attached to
    a number that is no longer true.
    """
    from flightrec.diff import GAP_OPEN, MOVE_CONFIDENCE, _MOVE_WINDOW

    for value, description in (
        (f"{GAP_OPEN}", "the gap-open penalty"),
        (f"{MOVE_CONFIDENCE}", "the move-rescue threshold"),
        (f"{_MOVE_WINDOW} positions", "the move window"),
    ):
        assert value in TEXT, f"README no longer quotes {description} as {value}"


# --- its own links ------------------------------------------------------------


def slug(heading: str) -> str:
    text = heading.lstrip("#").strip().lower()
    text = re.sub(r"[^\w\s-]", "", text.replace("—", " ").replace("–", " "))
    return re.sub(r"[\s_]+", "-", text).strip("-")


def test_internal_links_resolve() -> None:
    """A dead anchor is a promise that some other section explains this."""
    anchors = {slug(line) for line in TEXT.splitlines() if line.startswith("#")}
    links = set(re.findall(r"\]\(#([\w-]+)\)", TEXT))

    assert links, "the README cross-references itself; if it stops, drop this test"
    assert links <= anchors, f"dead anchors: {sorted(links - anchors)}"


# --- the counts in front of its lists -----------------------------------------

WORDS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7}


def counted_lists() -> list[tuple[int, str, int, int]]:
    """Every "Three things...:" preamble, with what it claims and what follows.

    Three of these went stale in a single afternoon -- one of them because
    fixing the other two added an item to a list whose preamble said "three".

    A preamble is a paragraph that opens with a number word *and ends with a
    colon*, which is what separates "Two more corrections, both to the
    benchmark:" from "Two of them broke it." The first introduces a list and can
    disagree with it; the second is prose about two things discussed in
    sentences, and counting the next list it happens to sit above produces a
    failure that is the checker's fault. The first version of this did that six
    times, and a checker that cries wolf gets its assertions loosened, which is
    the failure this file exists to prevent.
    """
    lines = TEXT.splitlines()
    found = []

    for index, line in enumerate(lines):
        match = re.match(
            r"^(?:\*\*)?(Two|Three|Four|Five|Six|Seven)\s+"
            r"(?:more\s+)?(?:things|fixes|bugs|corrections|classes)",
            line,
        )
        if not match:
            continue

        # The preamble may wrap; it runs to the next blank line.
        paragraph, cursor = [], index
        while cursor < len(lines) and lines[cursor].strip():
            paragraph.append(lines[cursor].strip())
            cursor += 1
        if not " ".join(paragraph).endswith(":"):
            continue

        while cursor < len(lines) and not lines[cursor].strip():
            cursor += 1

        items, table_rows = 0, 0
        while cursor < len(lines) and lines[cursor].strip():
            entry = lines[cursor]
            if re.match(r"^(?:[-*] |\d+\. )", entry):
                items += 1                      # continuations are indented
            elif entry.lstrip().startswith("|"):
                table_rows += 1
            cursor += 1

        if table_rows:
            items += table_rows - 2             # drop the header and separator
        found.append((index + 1, paragraph[0], WORDS[match.group(1).lower()], items))

    return found


def test_there_are_counted_lists_to_check() -> None:
    assert counted_lists(), "no counted preambles found; the regex has drifted"


@pytest.mark.parametrize(
    "line_number, preamble, claimed, actual",
    counted_lists(),
    ids=lambda value: str(value)[:40] if isinstance(value, str) else str(value),
)
def test_counted_lists_count_correctly(
    line_number: int, preamble: str, claimed: int, actual: int
) -> None:
    """"Three things are broken" above four things is a small lie that a reader
    trips over and no test has ever caught."""
    assert claimed == actual, (
        f"README line {line_number} says {claimed} but {actual} follow: {preamble!r}"
    )
