"""The `root` confinement `Tools` promises, tested against a link that leaves it.

`tools.py` opens with:

    Every path is confined to a `root` directory, so an agent cannot wander
    outside the project it was pointed at.

These tests hold that sentence to its word. They need no model, no network and
no hardware: `Tools` is pure standard library.

Creating a link is the only platform-sensitive step. POSIX gets `os.symlink`
for free; Windows needs either Developer Mode (for `os.symlink`) or a directory
junction via `mklink /J`, which is unprivileged. When neither is available the
link tests skip rather than pass vacuously.
"""
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from edge_agent.tools import Tools  # noqa: E402


# --------------------------------------------------------------- link helper
def _link_dir(link_path, target):
    """Create a directory link at `link_path` -> `target`, or skip the test."""
    try:
        os.symlink(target, link_path, target_is_directory=True)
        return
    except (OSError, NotImplementedError, AttributeError):
        pass
    if os.name == "nt":                      # junctions need no privilege
        r = subprocess.run(["cmd", "/c", "mklink", "/J", link_path, target],
                           capture_output=True, text=True)
        if r.returncode == 0:
            return
    pytest.skip("this platform will not let the test create a directory link")


def _link_file(link_path, target):
    try:
        os.symlink(target, link_path)
    except (OSError, NotImplementedError, AttributeError):
        pytest.skip("this platform will not let the test create a file link")


@pytest.fixture
def sandbox(tmp_path):
    """A project root, and a sibling directory that is *not* part of it."""
    root = tmp_path / "project"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "app.py").write_text("print('hello')\n", encoding="utf-8")
    (outside / "id_rsa").write_text("PRIVATE-KEY-MATERIAL\n", encoding="utf-8")
    return Tools(str(root)), root, outside


# ------------------------------------------------------ the plain ".." cases
def test_dotdot_read_is_refused(sandbox):
    tools, _, _ = sandbox
    with pytest.raises(ValueError, match="escapes the project root"):
        tools.read_file("../outside/id_rsa")


def test_dotdot_write_is_refused(sandbox):
    tools, _, outside = sandbox
    with pytest.raises(ValueError, match="escapes the project root"):
        tools.write_file("../outside/planted.txt", "x")
    assert not (outside / "planted.txt").exists()


def test_absolute_path_is_refused(sandbox):
    tools, _, outside = sandbox
    with pytest.raises(ValueError, match="escapes the project root"):
        tools.read_file(str(outside / "id_rsa"))


# ------------------------------------------------------- the link cases
def test_linked_directory_cannot_be_read_through(sandbox):
    tools, root, outside = sandbox
    _link_dir(str(root / "vendor"), str(outside))
    with pytest.raises(ValueError, match="escapes the project root"):
        tools.read_file("vendor/id_rsa")


def test_linked_directory_cannot_be_written_through(sandbox):
    tools, root, outside = sandbox
    _link_dir(str(root / "vendor"), str(outside))
    with pytest.raises(ValueError, match="escapes the project root"):
        tools.write_file("vendor/planted.txt", "escaped")
    assert not (outside / "planted.txt").exists()


def test_linked_directory_cannot_be_edited_through(sandbox):
    tools, root, outside = sandbox
    _link_dir(str(root / "vendor"), str(outside))
    with pytest.raises(ValueError, match="escapes the project root"):
        tools.edit_file("vendor/id_rsa", "PRIVATE", "PUBLIC")
    assert "PRIVATE" in (outside / "id_rsa").read_text(encoding="utf-8")


def test_grep_does_not_report_matches_from_outside(sandbox):
    tools, root, outside = sandbox
    _link_dir(str(root / "vendor"), str(outside))
    assert "PRIVATE-KEY-MATERIAL" not in tools.grep("PRIVATE-KEY-MATERIAL")


def test_search_files_does_not_list_files_from_outside(sandbox):
    tools, root, outside = sandbox
    _link_dir(str(root / "vendor"), str(outside))
    assert "id_rsa" not in tools.search_files("id_rsa")


def test_linked_file_is_not_read_by_grep(sandbox):
    tools, root, outside = sandbox
    _link_file(str(root / "key_alias"), str(outside / "id_rsa"))
    assert "PRIVATE-KEY-MATERIAL" not in tools.grep("PRIVATE-KEY-MATERIAL")


# ------------------------------------------------- the tools still work
def test_reading_inside_the_root_still_works(sandbox):
    tools, _, _ = sandbox
    assert "hello" in tools.read_file("app.py")


def test_writing_inside_the_root_still_works(sandbox):
    tools, root, _ = sandbox
    tools.write_file("notes/todo.md", "- ship it\n")
    assert (root / "notes" / "todo.md").read_text(encoding="utf-8") == "- ship it\n"


def test_editing_inside_the_root_still_works(sandbox):
    tools, root, _ = sandbox
    tools.edit_file("app.py", "hello", "world")
    assert (root / "app.py").read_text(encoding="utf-8") == "print('world')\n"


def test_grep_still_finds_matches_inside_the_root(sandbox):
    tools, _, _ = sandbox
    assert "app.py" in tools.grep("hello")


def test_search_files_still_lists_files_inside_the_root(sandbox):
    tools, _, _ = sandbox
    assert "app.py" in tools.search_files("*.py")


def test_a_link_that_stays_inside_the_root_is_allowed(sandbox):
    """Confinement is about leaving the root, not about links as such."""
    tools, root, _ = sandbox
    (root / "pkg").mkdir()
    (root / "pkg" / "mod.py").write_text("VALUE = 1\n", encoding="utf-8")
    _link_dir(str(root / "alias"), str(root / "pkg"))
    assert "VALUE = 1" in tools.read_file("alias/mod.py")


def test_dispatch_reports_the_refusal_instead_of_raising(sandbox):
    """The ReAct loop feeds `dispatch` output back to the model as text."""
    tools, _, _ = sandbox
    out = tools.dispatch("read_file", {"path": "../outside/id_rsa"})
    assert out.startswith("ERROR:")
    assert "escapes the project root" in out


# ------------------------------------------------------------ root itself
def test_root_is_resolved_so_a_linked_root_still_works(tmp_path):
    """If the root is reached through a link, its own files stay reachable."""
    real = tmp_path / "real_project"
    real.mkdir()
    (real / "a.txt").write_text("inside\n", encoding="utf-8")
    _link_dir(str(tmp_path / "via_link"), str(real))
    tools = Tools(str(tmp_path / "via_link"))
    assert "inside" in tools.read_file("a.txt")
