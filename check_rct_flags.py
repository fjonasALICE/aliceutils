#!/usr/bin/env python3
"""Print RCT quality flags for each reconstruction pass of a run list.

Reads run numbers from a text file (commas, whitespace, and # comments) and
queries the CCDB objects at RCT/Flags/RunFlags. Each object is a
map<timestamp, bitmask> for one (run, pass). The latest uploaded version is
used. Requires the ALICE environment (root on PATH) and the rich package.

Example:
  eval $(alienv -w /home/fjonas/alice/sw --no-refresh printenv O2Physics/latest)
  python3 check_rct_flags.py rct_example_runs.txt
  # writes rct_flags.html and prints a per-flag run count
"""

import argparse
import html
import json
import subprocess
import sys
import tempfile
import threading
import urllib.request
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

CCDB = "http://alice-ccdb.cern.ch"
BROWSE = CCDB + "/browse/RCT/Flags/RunFlags"

# Bit positions match o2::aod::rctsel::RCTSelectionFlags.
FLAGS = (
    (0, "CPV bad", False),
    (1, "EMC bad", False),
    (2, "EMC limAcc", True),
    (3, "FDD bad", False),
    (4, "FT0 bad", False),
    (5, "FV0 bad", False),
    (6, "HMP bad", False),
    (7, "ITS bad", False),
    (8, "ITS limAcc", True),
    (9, "MCH bad", False),
    (10, "MCH limAcc", True),
    (11, "MFT bad", False),
    (12, "MFT limAcc", True),
    (13, "MID bad", False),
    (14, "MID limAcc", True),
    (15, "PHS bad", False),
    (16, "TOF bad", False),
    (17, "TOF limAcc", True),
    (18, "TPC badTrk", False),
    (19, "TPC badPID", False),
    (20, "TPC limAcc", True),
    (21, "TRD bad", False),
    (22, "ZDC bad", False),
)

ROOT_MACRO = r"""
#include "CCDB/CcdbApi.h"
#include <fstream>
#include <iostream>
#include <map>
#include <string>

void fetch_rct(const char* queryFile, const char* outFile) {
  o2::ccdb::CcdbApi api;
  api.init("http://alice-ccdb.cern.ch");
  std::ifstream in(queryFile);
  std::ofstream out(outFile);
  std::string run, pass;
  long ts = 0;
  while (in >> run >> pass >> ts) {
    std::map<std::string, std::string> md;
    md["run"] = run;
    md["passName"] = pass;
    std::cerr << "RCT_BEGIN " << run << " " << pass << std::endl;
    auto* obj = api.retrieveFromTFileAny<std::map<unsigned long, unsigned int>>(
        "RCT/Flags/RunFlags", md, ts);
    if (!obj || obj->empty()) {
      out << run << " " << pass << " MISSING\n";
      std::cerr << "RCT_DONE " << run << " " << pass << " missing" << std::endl;
      continue;
    }
    for (const auto& kv : *obj) {
      out << run << " " << pass << " " << kv.first << " " << kv.second << "\n";
    }
    std::cerr << "RCT_DONE " << run << " " << pass << " ok" << std::endl;
  }
}
"""


def parse_runs(text):
    runs = []
    for raw in text.replace(",", " ").split():
        token = raw.strip()
        if not token or token.startswith("#"):
            continue
        if not token.isdigit():
            raise SystemExit(f"not a run number: {token}")
        if token not in runs:
            runs.append(token)
    if not runs:
        raise SystemExit("no run numbers given")
    return runs


def load_runs(path, inline):
    chunks = []
    if path:
        chunks.append(Path(path).read_text())
    if inline:
        chunks.append(inline.replace(",", " "))
    return parse_runs("\n".join(chunks))


def browse_objects():
    request = urllib.request.Request(BROWSE, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.load(response)
    return payload["objects"]


def latest_per_pass(objects, runs):
    """Keep the newest CCDB object for each (run, passName)."""
    wanted = set(runs)
    best = {}
    for obj in objects:
        run = str(obj.get("run", ""))
        if run not in wanted:
            continue
        pass_name = obj.get("passName") or "?"
        version = int(obj.get("version") or 0)
        created = int(obj.get("createTime") or 0)
        key = (run, pass_name)
        rank = (version, created)
        if key not in best or rank > best[key][0]:
            best[key] = (rank, obj)
    return best


def fetch_flags(queries, console):
    """queries: list of (run, pass, timestamp). Returns (run, pass) -> [(ts, bits)]."""
    with tempfile.TemporaryDirectory(prefix="rct_") as tmp:
        tmp = Path(tmp)
        macro = tmp / "fetch_rct.C"
        query_file = tmp / "queries.txt"
        out_file = tmp / "flags.txt"
        macro.write_text(ROOT_MACRO)
        query_file.write_text(
            "".join(f"{run} {pass_name} {ts}\n" for run, pass_name, ts in queries)
        )
        console.print(f"Reading [bold]{len(queries)}[/bold] RCT objects from CCDB")
        proc = subprocess.Popen(
            [
                "root",
                "-l",
                "-b",
                "-q",
                f'{macro}("{query_file}","{out_file}")',
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        root_log = []

        def _drain_stdout():
            root_log.append(proc.stdout.read())

        drain = threading.Thread(target=_drain_stdout)
        drain.start()
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
        )
        with progress:
            task = progress.add_task("starting ROOT", total=len(queries))
            for line in proc.stderr:
                parts = line.split()
                if len(parts) < 3 or parts[0] not in ("RCT_BEGIN", "RCT_DONE"):
                    continue
                label = f"{parts[1]} {parts[2]}"
                if parts[0] == "RCT_BEGIN":
                    progress.update(task, description=label)
                else:
                    state = parts[3] if len(parts) > 3 else "ok"
                    style = "red" if state == "missing" else "green"
                    progress.update(task, description=f"[{style}]{label}[/{style}]", advance=1)
        drain.join()
        returncode = proc.wait()
        if returncode != 0 or not out_file.exists():
            sys.stderr.write("".join(root_log))
            raise SystemExit(f"root failed with exit code {returncode}")
        flags = {}
        for line in out_file.read_text().splitlines():
            run, pass_name, *rest = line.split()
            flags.setdefault((run, pass_name), [])
            if rest == ["MISSING"]:
                continue
            ts, bits = rest
            flags[(run, pass_name)].append((int(ts), int(bits)))
        return flags


def describe(bits):
    bad = [name for bit, name, limited in FLAGS if bits & (1 << bit) and not limited]
    limited = [name for bit, name, is_limited in FLAGS if bits & (1 << bit) and is_limited]
    return bad, limited


def flags_text(entries):
    if not entries:
        return Text("missing", style="bold red")
    # Collapse identical bitmasks; keep distinct intervals.
    unique = []
    for ts, bits in entries:
        if not unique or unique[-1][1] != bits:
            unique.append((ts, bits))
    text = Text()
    for i, (ts, bits) in enumerate(unique):
        if i:
            text.append("\n")
        bad, limited = describe(bits)
        if len(unique) > 1:
            text.append(f"@{ts} ", style="dim")
        if not bad and not limited:
            text.append("good", style="green")
        else:
            if bad:
                text.append(", ".join(bad), style="red")
            if limited:
                if bad:
                    text.append(", ")
                text.append(", ".join(limited), style="yellow")
        text.append(f"  0x{bits:x}", style="dim")
    return text


def render(runs, chosen, flags, console):
    table = Table(title="RCT flags", show_lines=False, header_style="bold")
    table.add_column("Run", style="bold", no_wrap=True)
    table.add_column("Period", no_wrap=True)
    table.add_column("Pass", no_wrap=True)
    table.add_column("Flags")

    n_missing_runs = 0
    for run in runs:
        rows = [(pass_name, obj) for (r, pass_name), (_, obj) in chosen.items() if r == run]
        rows.sort(key=lambda item: item[0])
        if not rows:
            n_missing_runs += 1
            table.add_row(run, "-", "-", Text("no RCT object", style="bold red"))
            continue
        for pass_name, obj in rows:
            table.add_row(
                run,
                obj.get("periodName") or "-",
                pass_name,
                flags_text(flags.get((run, pass_name), [])),
            )

    console.print(table)
    console.print(
        f"{len(runs)} runs, {len(chosen)} pass objects, {n_missing_runs} runs without RCT",
        style="dim",
    )
    console.print("red = bad, yellow = limited acceptance (MC reproducible)", style="dim")
    render_summary(runs, chosen, flags, console)


def render_summary(runs, chosen, flags, console):
    counts = flag_counts(runs, chosen, flags)
    n_runs = len(runs)
    summary = Table(title="Runs with each flag set", header_style="bold")
    summary.add_column("Flag")
    summary.add_column("Runs", justify="right")
    for bit, name, limited in FLAGS:
        n = counts[bit]
        style = "yellow" if limited else "red"
        if n == 0:
            style = "dim"
        summary.add_row(Text(name, style=style), f"{n} / {n_runs}")
    console.print(summary)


def flag_counts(runs, chosen, flags):
    """Count runs where the flag is set on any pass, including part of the run."""
    counts = {bit: 0 for bit, _, _ in FLAGS}
    for run in runs:
        bits = 0
        for (listed_run, pass_name) in chosen:
            if listed_run != run:
                continue
            for _, value in flags.get((run, pass_name), []):
                bits |= value
        for bit, _, _ in FLAGS:
            if bits & (1 << bit):
                counts[bit] += 1
    return counts


def write_html(path, runs, chosen, flags):
    n_runs = len(runs)
    counts = flag_counts(runs, chosen, flags)
    detail_rows = []
    for run in runs:
        passes = [(pass_name, obj) for (listed, pass_name), (_, obj) in chosen.items() if listed == run]
        passes.sort(key=lambda item: item[0])
        if not passes:
            detail_rows.append(
                "<tr><td class='run'>{}</td><td>-</td><td>-</td><td><span class='bad'>no RCT object</span></td></tr>".format(
                    html.escape(run)
                )
            )
            continue
        for pass_name, obj in passes:
            period = html.escape(obj.get("periodName") or "-")
            detail_rows.append(
                "<tr><td class='run'>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                    html.escape(run),
                    period,
                    html.escape(pass_name),
                    entries_html(flags.get((run, pass_name), [])),
                )
            )

    summary_rows = []
    for bit, name, limited in FLAGS:
        n = counts[bit]
        kind = "lim" if limited else "bad"
        if n == 0:
            kind = "ok"
        width = 0 if n_runs == 0 else round(100 * n / n_runs)
        summary_rows.append(
            "<tr><td><span class='{kind}'>{name}</span></td>"
            "<td class='num'>{n} / {n_runs}</td>"
            "<td><div class='bar'><div class='fill {kind}' style='width:{width}%'></div></div></td></tr>".format(
                kind=kind, name=html.escape(name), n=n, n_runs=n_runs, width=width
            )
        )

    document = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>RCT flags</title>
<style>
  body {{ font-family: "Segoe UI", sans-serif; margin: 2rem; color: #1c1c1c; background: #f7f7f5; }}
  h1 {{ margin-bottom: 0.2rem; }}
  p.note {{ color: #555; }}
  table {{ border-collapse: collapse; background: white; margin: 1.2rem 0 2rem; width: 100%; }}
  th, td {{ border-bottom: 1px solid #e4e4e0; padding: 0.45rem 0.7rem; text-align: left; vertical-align: top; }}
  th {{ background: #222; color: white; position: sticky; top: 0; }}
  td.run {{ font-weight: 650; white-space: nowrap; }}
  td.num {{ white-space: nowrap; text-align: right; font-variant-numeric: tabular-nums; }}
  .bad {{ color: #b42318; font-weight: 650; }}
  .lim {{ color: #b54708; font-weight: 650; }}
  .good, .ok {{ color: #067647; font-weight: 650; }}
  .hex, .ts {{ color: #777; font-size: 0.85em; }}
  .interval {{ margin: 0.15rem 0; }}
  .bar {{ background: #eee; height: 0.7rem; border-radius: 4px; min-width: 8rem; }}
  .fill {{ height: 100%; border-radius: 4px; }}
  .fill.bad {{ background: #d92d20; }}
  .fill.lim {{ background: #dc6803; }}
  .fill.ok {{ background: #e4e4e0; }}
</style>
</head>
<body>
<h1>RCT flags</h1>
<p class="note">{n_runs} runs. Red is bad, orange is limited acceptance (MC reproducible).</p>
<table>
  <thead><tr><th>Run</th><th>Period</th><th>Pass</th><th>Flags</th></tr></thead>
  <tbody>
    {details}
  </tbody>
</table>
<h2>Runs with each flag set</h2>
<table>
  <thead><tr><th>Flag</th><th>Runs</th><th></th></tr></thead>
  <tbody>
    {summary}
  </tbody>
</table>
</body>
</html>
""".format(n_runs=n_runs, details="\n    ".join(detail_rows), summary="\n    ".join(summary_rows))
    Path(path).write_text(document)
    return path


def entries_html(entries):
    if not entries:
        return "<span class='bad'>missing</span>"
    unique = []
    for ts, bits in entries:
        if not unique or unique[-1][1] != bits:
            unique.append((ts, bits))
    chunks = []
    for ts, bits in unique:
        bad, limited = describe(bits)
        parts = []
        if len(unique) > 1:
            parts.append(f"<span class='ts'>@{ts}</span>")
        if not bad and not limited:
            parts.append("<span class='good'>good</span>")
        else:
            parts.extend(f"<span class='bad'>{html.escape(name)}</span>" for name in bad)
            parts.extend(f"<span class='lim'>{html.escape(name)}</span>" for name in limited)
        parts.append(f"<span class='hex'>0x{bits:x}</span>")
        chunks.append("<div class='interval'>{}</div>".format(" ".join(parts)))
    return "".join(chunks)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_file", nargs="?", help="text file of run numbers")
    parser.add_argument("--runs", help="comma- or space-separated run numbers")
    parser.add_argument("--html", default="rct_flags.html", help="HTML report path (default: rct_flags.html)")
    args = parser.parse_args()
    if not args.run_file and not args.runs:
        parser.error("give a run file and/or --runs")

    console = Console()
    runs = load_runs(args.run_file, args.runs)
    console.print(f"[bold]{len(runs)}[/bold] runs to check")
    with console.status("[bold]Querying CCDB[/bold] RCT/Flags/RunFlags"):
        objects = browse_objects()
    chosen = latest_per_pass(objects, runs)
    console.print(f"Found [bold]{len(chosen)}[/bold] reconstruction passes")
    queries = []
    for (run, pass_name), (_, obj) in sorted(chosen.items()):
        start = int(obj["validFrom"])
        stop = int(obj["validUntil"])
        ts = start if stop <= start else (start + stop) // 2
        queries.append((run, pass_name, ts))
    flags = fetch_flags(queries, console) if queries else {}
    render(runs, chosen, flags, console)
    html_path = write_html(args.html, runs, chosen, flags)
    console.print(f"Wrote [bold]{html_path}[/bold]")


if __name__ == "__main__":
    main()
