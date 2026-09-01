"""Which workspace a request is allowed to name.

`AGENT_WORKSPACE` is documented as "Project dir the agent reads/edits", so a
request naming some other directory should be refused rather than served.

These exercise the real FastAPI app through httpx's ASGI transport -- real
routing, real body parsing, real `edge_agent.Tools`. The only substitution is
`_make_complete`, because reaching the model needs a live OpenAI-compatible
endpoint and a key; the canned ReAct text it returns is what a prompt-injected
model could return anyway. No hardware, no network.

    pip install fastapi httpx pytest
    pytest edgeLLM/nextjs-nemotron-app/agent_sidecar/tests
"""
import asyncio
import importlib
import json
import os
import subprocess
import sys

import pytest

pytest.importorskip("fastapi", reason="the sidecar needs FastAPI")
httpx = pytest.importorskip("httpx", reason="the ASGI transport comes from httpx")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                          # agent_sidecar/
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "edge_agent", "src"))


def _link_dir(link_path, target):
    try:
        os.symlink(target, link_path, target_is_directory=True)
        return
    except (OSError, NotImplementedError, AttributeError):
        pass
    if os.name == "nt":
        r = subprocess.run(["cmd", "/c", "mklink", "/J", link_path, target],
                           capture_output=True, text=True)
        if r.returncode == 0:
            return
    pytest.skip("this platform will not let the test create a directory link")


@pytest.fixture
def sidecar(tmp_path, monkeypatch):
    """Import the sidecar with `tmp_path/workspace` as its only allowed root."""
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    (workspace / "app.py").write_text("print('hi')\n", encoding="utf-8")
    (outside / "id_rsa").write_text("PRIVATE-KEY-MATERIAL\n", encoding="utf-8")

    monkeypatch.setenv("AGENT_WORKSPACE", str(workspace))
    monkeypatch.delenv("AGENT_ALLOWED_ROOTS", raising=False)
    import agent_sidecar
    importlib.reload(agent_sidecar)          # constants are read at import time
    return agent_sidecar, workspace, outside


def _post(app, body):
    async def go():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url="http://sidecar") as c:
            r = await c.post("/run", json=body)
            return r.status_code, r.text
    return asyncio.run(go())


def _events(text):
    return [json.loads(l[6:]) for l in text.splitlines()
            if l.startswith("data: ") and l != "data: [DONE]"]


def _stub_model(mod, steps):
    mod._make_complete = lambda *a, **k: (lambda _m, it=iter(steps): next(it))


# ------------------------------------------------------ resolve_workspace
def test_no_root_means_the_default_workspace(sidecar):
    mod, workspace, _ = sidecar
    assert mod.resolve_workspace(None) == os.path.realpath(str(workspace))


def test_a_directory_under_the_workspace_is_allowed(sidecar):
    mod, workspace, _ = sidecar
    (workspace / "sub").mkdir()
    assert mod.resolve_workspace(str(workspace / "sub")).endswith("sub")


def test_a_directory_outside_the_workspace_is_refused(sidecar):
    mod, _, outside = sidecar
    with pytest.raises(ValueError, match="outside the roots"):
        mod.resolve_workspace(str(outside))


def test_the_filesystem_root_is_refused(sidecar):
    mod, _, _ = sidecar
    with pytest.raises(ValueError, match="outside the roots"):
        mod.resolve_workspace(os.path.abspath(os.sep))


def test_dotdot_out_of_the_workspace_is_refused(sidecar):
    mod, workspace, _ = sidecar
    with pytest.raises(ValueError, match="outside the roots"):
        mod.resolve_workspace(str(workspace / ".." / "outside"))


def test_a_link_out_of_the_workspace_is_refused(sidecar):
    mod, workspace, outside = sidecar
    _link_dir(str(workspace / "vendor"), str(outside))
    with pytest.raises(ValueError, match="outside the roots"):
        mod.resolve_workspace(str(workspace / "vendor"))


def test_the_allowlist_can_be_widened(tmp_path, monkeypatch):
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.setenv("AGENT_WORKSPACE", str(tmp_path / "ws"))
    monkeypatch.setenv("AGENT_ALLOWED_ROOTS",
                       os.pathsep.join([str(tmp_path / "ws"), str(other)]))
    import agent_sidecar
    importlib.reload(agent_sidecar)
    assert agent_sidecar.resolve_workspace(str(other)) == os.path.realpath(str(other))


# -------------------------------------------------------------- over HTTP
def test_post_run_refuses_an_outside_root_with_403(sidecar):
    mod, _, outside = sidecar
    _stub_model(mod, ["Final Answer: unreachable"])
    code, text = _post(mod.app, {"task": "read the key", "root": str(outside)})
    assert code == 403
    assert any(e.get("type") == "error" for e in _events(text))


def test_post_run_does_not_touch_a_file_outside_the_root(sidecar):
    mod, _, outside = sidecar
    _stub_model(mod, [
        'Action: write_file\nAction Input: {"path": "PWNED.txt", "content": "x"}',
        "Final Answer: done",
    ])
    _post(mod.app, {"task": "plant a file", "root": str(outside), "max_steps": 2})
    assert not (outside / "PWNED.txt").exists()


def test_post_run_still_serves_the_default_workspace(sidecar):
    mod, workspace, _ = sidecar
    _stub_model(mod, ['Action: read_file\nAction Input: {"path": "app.py"}',
                      "Final Answer: it prints hi"])
    code, text = _post(mod.app, {"task": "read app.py", "max_steps": 2})
    assert code == 200
    events = _events(text)
    start = next(e for e in events if e.get("type") == "start")
    assert start["root"] == os.path.realpath(str(workspace))
    assert any("print('hi')" in e.get("text", "") for e in events)


def test_post_run_still_serves_a_subdirectory_of_the_workspace(sidecar):
    mod, workspace, _ = sidecar
    (workspace / "sub").mkdir()
    (workspace / "sub" / "b.txt").write_text("inside\n", encoding="utf-8")
    _stub_model(mod, ['Action: read_file\nAction Input: {"path": "b.txt"}',
                      "Final Answer: read it"])
    code, text = _post(mod.app, {"task": "read b.txt",
                                 "root": str(workspace / "sub"), "max_steps": 2})
    assert code == 200
    assert any("inside" in e.get("text", "") for e in _events(text))


def test_health_reports_the_allowlist(sidecar):
    mod, workspace, _ = sidecar

    async def go():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=mod.app),
                                     base_url="http://sidecar") as c:
            return (await c.get("/health")).json()

    body = asyncio.run(go())
    assert body["allowed_roots"] == [os.path.realpath(str(workspace))]


def test_a_missing_task_is_still_a_400(sidecar):
    mod, _, _ = sidecar
    code, _ = _post(mod.app, {"root": None})
    assert code == 400
