"""Links in `docs/` have to resolve on the published site, not just in a checkout.

`.github/workflows/deploy-docs.yml` runs `mkdocs gh-deploy` on every push to
main, and mkdocs serves `docs/` alone. A `../../edgeLLM/x.py` link from
`docs/curriculum/` therefore points outside the site and 404s for every reader,
while resolving fine for anyone browsing the repository locally -- which is why
it survives review.

The repository already links to source files the way that works:

    [`tools.py`](https://github.com/lkk688/edgeAI/blob/main/edgeLLM/edge_agent/src/edge_agent/tools.py)

These tests hold the rest of the docs to that, and check that the URLs so
written point at files that are actually here.

Pure standard library. No hardware, no network: the GitHub URLs are checked
against the working tree, not fetched.

    pytest tests/test_docs_links.py
"""
import io
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"
BLOB = "https://github.com/lkk688/edgeAI/blob/main/"

# Targets under these are linked relatively in pages this project has not
# revisited; they are listed rather than silently skipped.
KNOWN_RELATIVE_ESCAPES = ("jetson/", "raspberrypi/")

MD_LINK = re.compile(r"\[([^\]]*)\]\(([^)\s]+)\)")


def _markdown_files():
    return sorted(DOCS.rglob("*.md"))


def test_the_docs_directory_is_where_it_is_expected():
    assert DOCS.is_dir()
    assert len(_markdown_files()) > 10


def test_no_relative_link_escapes_the_published_site():
    """Except the jetson/raspberrypi ones, which are named above."""
    offenders = []
    for md in _markdown_files():
        text = io.open(md, encoding="utf-8", errors="replace").read()
        for m in MD_LINK.finditer(text):
            target = m.group(2)
            if target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            resolved = (md.parent / target.split("#")[0]).resolve()
            try:
                rel = resolved.relative_to(REPO).as_posix()
            except ValueError:
                continue                       # outside the repo; not ours to judge
            if rel.startswith("docs/"):
                continue                       # inside the site, mkdocs serves it
            if rel.startswith(KNOWN_RELATIVE_ESCAPES):
                continue
            offenders.append(f"{md.relative_to(REPO).as_posix()} -> {target}")
    assert offenders == [], "these leave the site and 404 for readers"


def test_every_repository_blob_url_points_at_a_file_that_exists():
    """A rewritten link that 404s is no better than the relative one it replaced."""
    offenders = []
    for md in _markdown_files():
        text = io.open(md, encoding="utf-8", errors="replace").read()
        for m in re.finditer(re.escape(BLOB) + r"([^)\s\"']+)", text):
            rel = m.group(1).split("#")[0]
            if not (REPO / rel).exists():
                offenders.append(f"{md.relative_to(REPO).as_posix()} -> {rel}")
    assert offenders == []


@pytest.mark.parametrize("page", [
    "curriculum/11_nextjs_nemotron_app.md",
    "curriculum/11b_nextjs_agent_lab.md",
    "curriculum/13_react_agent.md",
])
def test_the_heaviest_pages_carry_no_escaping_link(page):
    """Named explicitly so a regression in these is obvious in the report."""
    md = DOCS / page
    assert md.is_file()
    text = io.open(md, encoding="utf-8", errors="replace").read()
    bad = []
    for m in MD_LINK.finditer(text):
        target = m.group(2)
        if target.startswith(("http://", "https://", "#", "mailto:")):
            continue
        resolved = (md.parent / target.split("#")[0]).resolve()
        try:
            rel = resolved.relative_to(REPO).as_posix()
        except ValueError:
            continue
        if not rel.startswith("docs/") and not rel.startswith(KNOWN_RELATIVE_ESCAPES):
            bad.append(target)
    assert bad == []
