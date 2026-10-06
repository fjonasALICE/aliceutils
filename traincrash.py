#!/usr/bin/env python3
"""
traincrash.py - Explain an AliHyperloop train-test crash with an OpenRouter agent.

Usage:
    python traincrash.py https://alimonitor.cern.ch/train-workdir/tests/0077/00774559/
    python traincrash.py <url> --config /path/to/traincrash.json
    python traincrash.py <url> --config traincrash.cern.json
"""

import os
import sys
import subprocess

# ---------------------------------------------------------------------------
# Bootstrap: ensure we're running inside a venv with pydantic-ai + requests
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(SCRIPT_DIR, ".venv_traincrash")


def _bootstrap() -> None:
    """If not already inside the project venv, create it and re-exec."""
    venv_python = os.path.join(VENV_DIR, "bin", "python")
    if os.path.abspath(sys.executable) == os.path.abspath(venv_python):
        return

    if not os.path.isfile(venv_python):
        import venv as _venv

        print(f"[setup] Creating virtual environment at {VENV_DIR} …")
        _venv.create(VENV_DIR, with_pip=True)
        pip = os.path.join(VENV_DIR, "bin", "pip")
        print("[setup] Installing dependencies (pydantic-ai, requests) …")
        subprocess.check_call([pip, "install", "--quiet", "pydantic-ai", "requests"])
        print("[setup] Done. Re-starting inside venv …\n")

    os.execv(venv_python, [venv_python] + sys.argv)


_bootstrap()

# ---------------------------------------------------------------------------
# Main imports (only reached when running inside the venv)
# ---------------------------------------------------------------------------
import argparse
import json
import re
import tempfile
import time
from pathlib import Path
from urllib.parse import quote

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

import requests
from pydantic_ai import Agent, UsageLimits
from pydantic_ai.exceptions import ModelHTTPError, UsageLimitExceeded
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.openrouter import OpenRouterProvider

MAX_LINES = 200
MAX_HITS = 80
MAX_DIGEST = 120
MAX_BYTES = 2_000_000
LINE_CLIP = 500
REPRO_TIMEOUT = 30 * 60
SKIP_FILES = {"network.onnx", "profile.linux-perf.txt", "performanceMetrics.json"}
HYPERLOOP = Path(SCRIPT_DIR) / "hyperlooptraintest.py"
GITHUB_HEADERS = {
    "User-Agent": "aliceutils-traincrash",
    "Accept": "application/vnd.github+json",
}

_DIGEST_RE = re.compile(
    r"\[ERROR\]|\[FATAL\]|^ERROR:|SEVERE:|runtime_error|test_exitcode"
    r"|O2 exited with (?!0\b)\d+|O2 exit code is (?!0\b)\d+"
    r"|\(exit (?!0\b)\d+\)"
)

INSTRUCTIONS = (
    "You investigate why an ALICE Hyperloop train test crashed. "
    "A crash digest from the remote stdout.log is provided. "
    "Use the tools to read the published logs, the task configuration, and the matching O2Physics sources on GitHub. "
    "A crash is often a misconfiguration of the failing task or of a task it depends on, not a bug in the source. However, bugs in the source can also cause crashes. "
    "Check configuration.json, dpl-config.json, and full_config.json for that task's options and for the options of tasks that feed it. "
    "Explain which device failed, the error, and whether the cause is a config setting or the source. "
    "Call reproduce only when the logs, the configuration, and the source are not enough to explain the crash. "
    "Reproducing downloads input files and runs the workflow locally, and can take a long time. "
    "Read the function that raises the error. Do not page through a whole file. "
    "Reply with a short diagnosis: what failed, why, and the relevant file and lines. "
    "End with a section titled \"Recommended action\": the concrete next step "
    "(a code change, a config change, or a check), naming the file and what to change."
)


class Context:
    """Downloads and caches one train-test directory for the agent tools."""

    def __init__(self, url: str) -> None:
        self.url = url
        self.cache = Path(tempfile.mkdtemp(prefix="traincrash_"))
        self.texts: dict[str, str] = {}
        self.index: list[str] | None = None
        self.tag: str | None = None
        self.ref: str | None = None
        self.tree: list[str] | None = None
        self.tree_note = ""
        self.repro: str | None = None

    def load(self, name: str) -> str:
        if name in self.texts:
            return self.texts[name]
        dest = self.cache / name
        if dest.is_file():
            text = dest.read_text(errors="replace")
            self.texts[name] = text
            return text
        resp = requests.get(self.url + name, timeout=120)
        resp.raise_for_status()
        if len(resp.content) > MAX_BYTES:
            raise RuntimeError(f"{name} is too large ({len(resp.content)} bytes)")
        text = resp.content.decode("utf-8", errors="replace")
        dest.write_text(text)
        self.texts[name] = text
        return text


def normalize_url(url: str) -> str:
    """Ensure the URL ends with / and downgrade alimonitor https to http."""
    url = url.strip().rstrip("/") + "/"
    if not (url.startswith("http://") or url.startswith("https://")):
        raise SystemExit(f"Expected an http(s) train-test URL, got: {url}")
    if url.startswith("https://alimonitor.cern.ch"):
        url = "http://" + url[len("https://") :]
        print("Note: alimonitor.cern.ch uses plain HTTP – switched https→http")
    return url


def clip(line: str) -> str:
    line = line.rstrip("\n")
    if len(line) <= LINE_CLIP:
        return line
    return line[:LINE_CLIP] + " ...[truncated]"


def format_slice(text: str, start: int, end: int) -> str:
    lines = text.splitlines()
    n = len(lines)
    if n == 0:
        return "(empty)"
    start = max(1, start)
    end = min(n, end if end else n)
    if start > n:
        return f"start_line {start} is past end of file ({n} lines)"
    if end < start:
        end = start
    truncated = False
    if end - start + 1 > MAX_LINES:
        end = start + MAX_LINES - 1
        truncated = True
    body = "\n".join(f"{i}: {clip(lines[i - 1])}" for i in range(start, end + 1))
    if truncated:
        body += f"\n... truncated to {MAX_LINES} lines; file has {n} lines"
    return body


def ref_candidates(tag: str) -> list[str]:
    """Package tag, then the same tag without a trailing build number."""
    refs = [tag]
    stripped = re.sub(r"-\d+$", "", tag)
    if stripped and stripped != tag:
        refs.append(stripped)
    return refs


def build_digest(ctx: Context) -> str:
    """Package line, status.json, and the log lines that describe the crash."""
    try:
        stdout = ctx.load("stdout.log")
    except (requests.RequestException, RuntimeError) as exc:
        raise SystemExit(f"Could not download stdout.log: {exc}") from exc

    match = re.search(r"O2Physics::(\S+)", stdout)
    ctx.tag = match.group(1) if match else None

    parts: list[str] = []
    for line in stdout.splitlines():
        if line.startswith("PACKAGES ="):
            parts.append(line)
            break
    if ctx.tag:
        parts.append(f"O2Physics tag: {ctx.tag}")

    try:
        status = ctx.load("status.json").strip()
        parts.append("status.json:\n" + status)
    except (requests.RequestException, RuntimeError) as exc:
        parts.append(f"status.json unavailable: {exc}")

    hits: list[str] = []
    for i, line in enumerate(stdout.splitlines(), start=1):
        if _DIGEST_RE.search(line):
            hits.append(f"{i}: {clip(line)}")
            if len(hits) >= MAX_DIGEST:
                hits.append(f"... digest truncated at {MAX_DIGEST} matching lines")
                break
    if hits:
        parts.append("Matching log lines:\n" + "\n".join(hits))
    else:
        lines = stdout.splitlines()
        tail = lines[-80:]
        start_no = len(lines) - len(tail) + 1
        body = "\n".join(f"{i}: {clip(line)}" for i, line in enumerate(tail, start=start_no))
        parts.append("No crash lines matched. Last 80 lines of stdout.log:\n" + body)
    return "\n\n".join(parts)


def list_names(html: str) -> list[str]:
    """Filenames from the alimonitor directory table, skipping binaries and perf dumps."""
    names: list[str] = []
    for _href, name in re.findall(r'<a href="([^"]+)">\s*<tt>([^<]+)</tt>', html):
        if name.endswith("/") or name in SKIP_FILES or name.endswith((".onnx", ".root")):
            continue
        names.append(name)
    return names


def fetch_index(ctx: Context) -> list[str]:
    if ctx.index is not None:
        return ctx.index
    try:
        resp = requests.get(ctx.url, timeout=60)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"failed to list {ctx.url}: {exc}") from exc
    ctx.index = list_names(resp.text)
    return ctx.index


def tool_list_files(ctx: Context) -> str:
    print("  list_files", flush=True)
    try:
        names = fetch_index(ctx)
    except RuntimeError as exc:
        return str(exc)
    if not names:
        return "No text files found."
    return "\n".join(names)


def tool_read_file(ctx: Context, name: str, start_line: int, end_line: int) -> str:
    print(f"  read_file {name} {start_line}-{end_line}", flush=True)
    name = name.strip()
    if not re.fullmatch(r"[A-Za-z0-9._-]+", name):
        return f"Refusing to read {name!r}"
    try:
        allowed = fetch_index(ctx)
    except RuntimeError as exc:
        return str(exc)
    if name not in allowed:
        return f"{name} is not an available text file. Available: {', '.join(allowed)}"
    try:
        text = ctx.load(name)
    except (requests.RequestException, RuntimeError) as exc:
        return f"failed to read {name}: {exc}"
    return format_slice(text, start_line, end_line)


def tool_search_log(ctx: Context, pattern: str) -> str:
    print(f"  search_log {pattern!r}", flush=True)
    try:
        regex = re.compile(pattern)
    except re.error as exc:
        return f"invalid regex: {exc}"
    try:
        text = ctx.load("stdout.log")
    except (requests.RequestException, RuntimeError) as exc:
        return f"failed to read stdout.log: {exc}"
    hits: list[str] = []
    total = 0
    for i, line in enumerate(text.splitlines(), start=1):
        if regex.search(line):
            total += 1
            if len(hits) < MAX_HITS:
                hits.append(f"{i}: {clip(line)}")
    if not hits:
        return "no matches"
    body = "\n".join(hits)
    if total > MAX_HITS:
        body += f"\n... {total} matches, showing {MAX_HITS}"
    return body


def ensure_tree(ctx: Context) -> str | None:
    """Fetch the O2Physics git tree once. Returns an error string, or None."""
    if ctx.tree is not None:
        return None
    if not ctx.tag:
        return "No O2Physics package tag found in stdout.log"
    tried: list[str] = []
    for ref in ref_candidates(ctx.tag):
        url = (
            "https://api.github.com/repos/AliceO2Group/O2Physics/git/trees/"
            f"{quote(ref, safe='')}?recursive=1"
        )
        try:
            resp = requests.get(url, headers=GITHUB_HEADERS, timeout=120)
        except requests.RequestException as exc:
            return f"GitHub request failed: {exc}"
        if resp.status_code == 404:
            tried.append(ref)
            continue
        if resp.status_code != 200:
            return f"GitHub tree for {ref} returned {resp.status_code}: {resp.text[:300]}"
        data = resp.json()
        ctx.ref = ref
        ctx.tree = [
            item["path"] for item in data.get("tree", []) if item.get("type") == "blob"
        ]
        if data.get("truncated"):
            ctx.tree_note = " (tree truncated by GitHub)"
        return None
    return f"No GitHub tag found for {ctx.tag}. Tried: {', '.join(tried)}"


def tool_find_o2physics(ctx: Context, name: str) -> str:
    print(f"  find_o2physics {name!r}", flush=True)
    name = name.strip()
    if not name:
        return "name is empty"
    err = ensure_tree(ctx)
    if err:
        return err
    assert ctx.tree is not None
    needle = name.lower()
    matches = [path for path in ctx.tree if needle in path.lower()]
    if not matches:
        return f"no paths containing {name!r} at ref {ctx.ref}{ctx.tree_note}"
    shown = matches[:40]
    body = "\n".join(shown)
    extra = ""
    if len(matches) > len(shown):
        extra = f"\n... {len(matches)} paths, showing {len(shown)}"
    return f"ref {ctx.ref}{ctx.tree_note}\n{body}{extra}"


def tool_read_o2physics(ctx: Context, path: str, start_line: int, end_line: int) -> str:
    print(f"  read_o2physics {path} {start_line}-{end_line}", flush=True)
    path = path.strip().lstrip("/")
    if not path or path.startswith("..") or "/../" in f"/{path}/":
        return f"Refusing to read {path!r}"
    err = ensure_tree(ctx)
    if err:
        return err
    url = (
        "https://raw.githubusercontent.com/AliceO2Group/O2Physics/"
        f"{quote(ctx.ref or '', safe='')}/{quote(path, safe='/')}"
    )
    try:
        resp = requests.get(url, headers=GITHUB_HEADERS, timeout=60)
    except requests.RequestException as exc:
        return f"GitHub request failed: {exc}"
    if resp.status_code != 200:
        return f"GitHub file {path} at {ctx.ref} returned {resp.status_code}"
    if len(resp.content) > MAX_BYTES:
        return f"{path} is too large ({len(resp.content)} bytes)"
    return f"ref {ctx.ref} {path}\n" + format_slice(
        resp.content.decode("utf-8", errors="replace"), start_line, end_line
    )


def tool_reproduce(ctx: Context) -> str:
    print("  reproduce", flush=True)
    if ctx.repro is not None:
        return ctx.repro
    if not HYPERLOOP.is_file():
        return f"hyperlooptraintest.py not found at {HYPERLOOP}"
    started = time.time()
    cmd = [sys.executable, str(HYPERLOOP), ctx.url, "--workdir", str(Path.cwd())]
    timed_out = False
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=REPRO_TIMEOUT
        )
        out, err, code = proc.stdout or "", proc.stderr or "", str(proc.returncode)
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        out = exc.stdout or ""
        err = exc.stderr or ""
        if isinstance(out, bytes):
            out = out.decode(errors="replace")
        if isinstance(err, bytes):
            err = err.decode(errors="replace")
        code = "timeout"

    logs = [
        path
        for path in Path.cwd().glob("traintest_*/stdout.log")
        if path.stat().st_mtime >= started - 1
    ]
    logs.sort(key=lambda path: path.stat().st_mtime)
    parts: list[str] = []
    if timed_out:
        parts.append("Reproduction timed out after 30 minutes.")
    parts.append(f"hyperlooptraintest.py exit: {code}")
    if logs:
        log = logs[-1]
        text = log.read_text(errors="replace")
        lines = text.splitlines()
        tail = lines[-MAX_LINES:]
        start_no = len(lines) - len(tail) + 1
        parts.append(f"work directory: {log.parent}")
        parts.append("local stdout.log (last 200 lines):")
        parts.append("\n".join(f"{i}: {clip(line)}" for i, line in enumerate(tail, start=start_no)))
    else:
        parts.append("No local stdout.log was produced.")
        runner = err or out
        if runner:
            tail = runner.splitlines()[-80:]
            parts.append("runner output (tail):")
            parts.append("\n".join(tail))
    ctx.repro = "\n".join(parts)
    return ctx.repro


def load_config(path: Path) -> dict[str, str]:
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {path}: {exc}") from exc
    key = data.get("api_key") if isinstance(data, dict) else None
    model = data.get("model") if isinstance(data, dict) else None
    if not isinstance(key, str) or not key.strip():
        raise SystemExit(f"{path} must contain a non-empty api_key")
    if not isinstance(model, str) or not model.strip():
        raise SystemExit(f"{path} must contain a non-empty model")
    cfg = {"api_key": key.strip(), "model": model.strip()}
    base_url = data.get("base_url")
    if base_url is not None:
        if not isinstance(base_url, str) or not base_url.strip():
            raise SystemExit(f"{path} base_url must be a non-empty string")
        cfg["base_url"] = base_url.strip().rstrip("/")
    return cfg


def build_model(cfg: dict[str, str]):
    if cfg.get("base_url"):
        return OpenAIChatModel(
            cfg["model"],
            provider=OpenAIProvider(base_url=cfg["base_url"], api_key=cfg["api_key"]),
            profile=OpenAIModelProfile(openai_supports_strict_tool_definition=False),
        )
    return OpenRouterModel(
        cfg["model"],
        provider=OpenRouterProvider(api_key=cfg["api_key"]),
    )


def investigate(ctx: Context, cfg: dict[str, str], digest: str) -> str:
    model = build_model(cfg)
    agent = Agent(model, instructions=INSTRUCTIONS)

    @agent.tool_plain
    def list_files() -> str:
        """List text files published with this train test. Skips binaries and large perf dumps."""
        return tool_list_files(ctx)

    @agent.tool_plain
    def read_file(name: str, start_line: int, end_line: int) -> str:
        """Read up to 200 lines of a train-test text file. Line numbers are 1-based."""
        return tool_read_file(ctx, name, start_line, end_line)

    @agent.tool_plain
    def search_log(pattern: str) -> str:
        """Regex-search stdout.log. Returns up to 80 matching lines with line numbers."""
        return tool_search_log(ctx, pattern)

    @agent.tool_plain
    def find_o2physics(name: str) -> str:
        """Find O2Physics source paths whose names contain this substring, at the package tag."""
        return tool_find_o2physics(ctx, name)

    @agent.tool_plain
    def read_o2physics(path: str, start_line: int, end_line: int) -> str:
        """Read up to 200 lines of an O2Physics file at the package tag. Path is repo-relative."""
        return tool_read_o2physics(ctx, path, start_line, end_line)

    @agent.tool_plain
    def reproduce() -> str:
        """Reproduce the train test locally with hyperlooptraintest.py. Slow. Use only if logs and source are not enough."""
        return tool_reproduce(ctx)

    prompt = f"Train test URL: {ctx.url}\n\nCrash digest:\n{digest}"
    result = agent.run_sync(prompt, usage_limits=UsageLimits(request_limit=50))
    return result.output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Investigate an AliHyperloop train-test crash with an OpenRouter agent.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python traincrash.py "
            "https://alimonitor.cern.ch/train-workdir/tests/0077/00774559/"
        ),
    )
    parser.add_argument("url", help="Train-test directory URL on alimonitor")
    parser.add_argument(
        "--config",
        default=None,
        metavar="FILE",
        help="JSON file with api_key, model, and optional base_url (default: traincrash.json next to this script)",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config).expanduser() if args.config else Path(SCRIPT_DIR) / "traincrash.json"
    if not cfg_path.is_file():
        raise SystemExit(
            f"Missing {cfg_path}. Copy traincrash.json.example and set api_key and model."
        )
    cfg = load_config(cfg_path)
    ctx = Context(normalize_url(args.url))
    via = cfg.get("base_url", "OpenRouter")
    print(f"Investigating {ctx.url} with {cfg['model']} via {via} …")
    digest = build_digest(ctx)
    try:
        print(investigate(ctx, cfg, digest))
    except ModelHTTPError as exc:
        detail = exc.body.get("message") if isinstance(exc.body, dict) else None
        detail = detail or exc.message
        hint = f" Try {exc.suggested_model_id!r}." if exc.suggested_model_id else ""
        raise SystemExit(f"{via} rejected {cfg['model']}: {detail}.{hint}") from None
    except UsageLimitExceeded as exc:
        raise SystemExit(str(exc)) from None


if __name__ == "__main__":
    main()
