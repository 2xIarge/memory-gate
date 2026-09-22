"""Probe the *installed* langchain/langgraph API surface.

Everything this project depends on is asserted here against the installed
release, not against the master branch source we read earlier.
Output is written to a file so Windows console encoding cannot mangle it.
"""

from __future__ import annotations

import inspect
import io
import sys
from pathlib import Path

OUT = Path(__file__).with_name("probe_report.txt")
buf = io.StringIO()


def say(*a):
    print(*a, file=buf)


def section(title):
    say("")
    say("=" * 70)
    say(title)
    say("=" * 70)


section("VERSIONS")
for mod in ("langchain", "langchain_core", "langgraph"):
    m = __import__(mod)
    say(f"{mod}: {getattr(m, '__version__', '?')}")
say(f"python: {sys.version.split()[0]}")

section("IMPORTS")
imports = {
    "create_agent": "langchain.agents",
    "AgentMiddleware": "langchain.agents.middleware",
    "SummarizationMiddleware": "langchain.agents.middleware",
    "ModelRequest": "langchain.agents.middleware",
    "AgentState": "langchain.agents.middleware",
    "ModelRetryMiddleware": "langchain.agents.middleware",
    "ModelFallbackMiddleware": "langchain.agents.middleware",
    "HumanInTheLoopMiddleware": "langchain.agents.middleware",
    "interrupt": "langgraph.types",
    "Command": "langgraph.types",
    "GraphBubbleUp": "langgraph.errors",
    "GraphInterrupt": "langgraph.errors",
    "RemoveMessage": "langchain_core.messages",
    "REMOVE_ALL_MESSAGES": "langchain_core.messages",
    "GenericFakeChatModel": "langchain_core.language_models.fake_chat_models",
    "InMemorySaver": "langgraph.checkpoint.memory",
}
resolved = {}
for name, path in imports.items():
    try:
        mod = __import__(path, fromlist=[name])
        obj = getattr(mod, name)
        resolved[name] = obj
        say(f"OK    {name:32s} <- {path}")
    except Exception as exc:  # noqa: BLE001
        say(f"FAIL  {name:32s} <- {path}   {type(exc).__name__}: {exc}")

section("PrivateStateAttr (needed for state_schema extension)")
for path in ("langchain.agents.middleware", "langchain.agents.middleware.types", "langchain.agents"):
    try:
        mod = __import__(path, fromlist=["PrivateStateAttr"])
        obj = getattr(mod, "PrivateStateAttr")
        say(f"OK    PrivateStateAttr <- {path}   {obj!r}")
        break
    except Exception as exc:  # noqa: BLE001
        say(f"FAIL  PrivateStateAttr <- {path}   {type(exc).__name__}: {exc}")

section("AgentMiddleware hooks present on base class")
amw = resolved.get("AgentMiddleware")
if amw is not None:
    for h in ("before_agent", "before_model", "after_model", "after_agent",
              "wrap_model_call", "wrap_tool_call", "abefore_model", "awrap_model_call"):
        say(f"  {h:20s} {'yes' if hasattr(amw, h) else 'NO'}")
    say(f"  state_schema attr: {getattr(amw, 'state_schema', 'MISSING')!r}")

section("ModelRequest.override signature")
mr = resolved.get("ModelRequest")
if mr is not None:
    say(f"  fields: {[f.name for f in __import__('dataclasses').fields(mr)]}"
        if __import__("dataclasses").is_dataclass(mr) else "  not a dataclass")
    if hasattr(mr, "override"):
        say(f"  override{inspect.signature(mr.override)}")
    else:
        say("  override MISSING")

section("SummarizationMiddleware public attrs + before_model")
smw = resolved.get("SummarizationMiddleware")
if smw is not None:
    sig = inspect.signature(smw.__init__)
    say(f"  __init__{sig}")
    say(f"  has before_model : {smw.before_model is not amw.before_model if amw else '?'}")
    say(f"  has abefore_model: {smw.abefore_model is not amw.abefore_model if amw else '?'}")

section("GraphBubbleUp guard in INSTALLED model_retry / model_fallback")
for name in ("ModelRetryMiddleware", "ModelFallbackMiddleware"):
    cls = resolved.get(name)
    if cls is None:
        say(f"  {name}: not importable")
        continue
    src_file = Path(inspect.getfile(cls))
    text = src_file.read_text(encoding="utf-8", errors="replace")
    has_guard = "GraphBubbleUp" in text
    n_except_exc = text.count("except Exception")
    say(f"  {name}")
    say(f"    file: {src_file}")
    say(f"    mentions GraphBubbleUp: {has_guard}")
    say(f"    bare 'except Exception' count: {n_except_exc}")
    if has_guard:
        for i, line in enumerate(text.splitlines(), 1):
            if "GraphBubbleUp" in line:
                say(f"      L{i}: {line.strip()}")

section("before_model node naming in INSTALLED factory")
try:
    import langchain.agents.factory as fac

    fac_file = Path(inspect.getfile(fac))
    text = fac_file.read_text(encoding="utf-8", errors="replace")
    say(f"  file: {fac_file}")
    for i, line in enumerate(text.splitlines(), 1):
        s = line.strip()
        if ".before_model" in s and ("add_node" in s or "entry_node" in s
                                      or "loop_entry_node" in s or "pairwise" in s):
            say(f"    L{i}: {s[:150]}")
except Exception as exc:  # noqa: BLE001
    say(f"  FAIL {type(exc).__name__}: {exc}")

section("wrap_model_call composition direction in INSTALLED factory")
try:
    for i, line in enumerate(text.splitlines(), 1):
        s = line.strip()
        if "outermost" in s or "right-to-left" in s or s.startswith("for h in reversed"):
            say(f"    L{i}: {s[:150]}")
except Exception as exc:  # noqa: BLE001
    say(f"  FAIL {type(exc).__name__}: {exc}")

OUT.write_text(buf.getvalue(), encoding="utf-8")
print(f"written -> {OUT}")
