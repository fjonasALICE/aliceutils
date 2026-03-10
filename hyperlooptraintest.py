#!/usr/bin/env python3
"""
hyperlooptraintest.py - Download and run an AliHyperloop train test locally via apptainer.

Usage:
    python hyperlooptraintest.py <url>
    python hyperlooptraintest.py https://alimonitor.cern.ch/train-workdir/tests/0063/00632029/
"""

import sys
import os
import shutil
import subprocess

# ---------------------------------------------------------------------------
# Bootstrap: ensure we're running inside a venv with rich + requests installed
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(SCRIPT_DIR, ".venv_hyperloop")


def _bootstrap() -> None:
    """If not already inside the project venv, create it and re-exec."""
    venv_python = os.path.join(VENV_DIR, "bin", "python")
    # Already running the right python?
    if os.path.abspath(sys.executable) == os.path.abspath(venv_python):
        return

    needs_install = not os.path.isfile(venv_python)

    if needs_install:
        import venv as _venv
        print(f"[setup] Creating virtual environment at {VENV_DIR} …")
        _venv.create(VENV_DIR, with_pip=True)
        pip = os.path.join(VENV_DIR, "bin", "pip")
        print("[setup] Installing dependencies (requests, rich) …")
        subprocess.check_call([pip, "install", "--quiet", "requests", "rich"])
        print("[setup] Done. Re-starting inside venv …\n")

    os.execv(venv_python, [venv_python] + sys.argv)


_bootstrap()

# ---------------------------------------------------------------------------
# Main imports (only reached when running inside the venv)
# ---------------------------------------------------------------------------
import argparse
import uuid
from pathlib import Path
from typing import List, Optional

import requests
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

console = Console()

FILES_TO_DOWNLOAD = ["stdout.log", "configuration.json", "OutputDirector.json", "env.sh"]
OPTIONAL_FILES = {"OutputDirector.json"}
SIF_PATH = Path(SCRIPT_DIR) / "el9.sif"
SIF_IMAGE = "docker://alisw/slc9-builder:latest"

# Marker lines in stdout.log
_RUN_CMD_TRIGGER = "you can achieve this with the following reduced command line:"
_ALIEN_PATHS_TRIGGER = "The corresponding AliEn paths are"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_url(url: str) -> str:
    """Ensure URL ends with / and downgrades https→http for alimonitor (plain HTTP only)."""
    url = url.rstrip("/") + "/"
    # alimonitor.cern.ch serves over plain HTTP; https causes SSL handshake failures
    if url.startswith("https://alimonitor.cern.ch"):
        url = "http://" + url[len("https://"):]
        console.print(
            "  [yellow]Note: alimonitor.cern.ch uses plain HTTP – switched https→http[/yellow]"
        )
    return url


def make_work_dir(base: Path) -> Path:
    uid = uuid.uuid4().hex[:8]
    d = base / f"traintest_{uid}"
    d.mkdir(parents=True, exist_ok=False)
    return d


def download_file(url: str, dest: Path) -> bool:
    try:
        resp = requests.get(url, timeout=120, stream=True)
        resp.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=65536):
                fh.write(chunk)
        return True
    except requests.RequestException as exc:
        console.print(f"    [red]Error: {exc}[/red]")
        return False


def extract_run_command(stdout_log: Path) -> Optional[str]:
    """Return the line immediately after the 'reduced command line' marker."""
    lines = stdout_log.read_text(errors="replace").splitlines()
    for i, line in enumerate(lines):
        if _RUN_CMD_TRIGGER in line:
            for candidate in lines[i + 1 :]:
                stripped = candidate.strip()
                if stripped:
                    return stripped
    return None


def extract_alien_paths(stdout_log: Path) -> List[str]:
    """Return all non-empty lines after the AliEn paths marker until a blank line."""
    lines = stdout_log.read_text(errors="replace").splitlines()
    paths: List[str] = []
    collecting = False
    for line in lines:
        if _ALIEN_PATHS_TRIGGER in line:
            collecting = True
            continue
        if collecting:
            stripped = line.strip()
            if not stripped:
                break
            paths.append(stripped)
    return paths


def ensure_sif() -> None:
    if SIF_PATH.exists():
        console.print(f"  [green]✓[/green] SIF image: [cyan]{SIF_PATH}[/cyan]")
        return
    console.print(
        f"  [yellow]SIF not found – pulling {SIF_IMAGE} → {SIF_PATH} …[/yellow]"
    )
    result = subprocess.run(
        ["apptainer", "pull", str(SIF_PATH), SIF_IMAGE], check=False
    )
    if result.returncode != 0:
        console.print("  [red]✗ Failed to pull SIF image. Is apptainer installed?[/red]")
        sys.exit(1)
    console.print(f"  [green]✓[/green] SIF pulled successfully.")


def run_in_container(work_dir: Path) -> int:
    """Execute env.sh + run.sh inside the apptainer container."""
    ensure_sif()

    apptainer_cmd = [
        "apptainer", "exec",
        "--bind", "/cvmfs:/cvmfs",
        "--bind", f"{work_dir}:/workdir",
        "--cleanenv",
        str(SIF_PATH),
        "bash", "-c",
        "cd /workdir && source env.sh && bash run.sh",
    ]

    console.print(
        Panel(
            " \\\n  ".join(apptainer_cmd),
            title="[bold blue]Apptainer Command[/bold blue]",
            border_style="blue",
            padding=(1, 2),
        )
    )

    console.rule("[bold blue]Container Output[/bold blue]")
    result = subprocess.run(apptainer_cmd, cwd=str(work_dir))
    console.rule()
    return result.returncode


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    console.print(
        Panel(
            "[bold cyan]AliHyperloop Train Test Runner[/bold cyan]\n"
            "[dim]Downloads train-test files and runs the workflow locally via apptainer[/dim]",
            border_style="cyan",
            padding=(1, 4),
        )
    )

    parser = argparse.ArgumentParser(
        description="Download and run an AliHyperloop train test locally.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:\n  python hyperlooptraintest.py "
               "https://alimonitor.cern.ch/train-workdir/tests/0063/00632029/",
    )
    parser.add_argument(
        "url",
        help="URL to the train-test directory on alimonitor",
    )
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="Download and prepare files only; skip the container execution",
    )
    parser.add_argument(
        "--workdir",
        default=None,
        metavar="DIR",
        help="Base directory in which to create the traintest_XXXX folder "
             "(default: current working directory)",
    )
    parser.add_argument(
        "--configuration",
        default=None,
        metavar="FILE",
        help="Path to a local configuration.json to use instead of downloading it from the train-test URL",
    )
    parser.add_argument(
        "--input-data",
        default=None,
        metavar="FILE",
        help="Path to a local file to use as input_data.txt instead of extracting AliEn paths from stdout.log",
    )
    args = parser.parse_args()

    base_url = normalize_url(args.url)
    base_dir = Path(args.workdir).resolve() if args.workdir else Path.cwd()

    console.print(f"  [bold]Source URL     :[/bold] {base_url}")
    console.print(f"  [bold]Base dir       :[/bold] {base_dir}")
    if args.configuration:
        console.print(f"  [bold]configuration  :[/bold] [yellow]{args.configuration}[/yellow] [dim](override)[/dim]")
    if args.input_data:
        console.print(f"  [bold]input_data.txt :[/bold] [yellow]{args.input_data}[/yellow] [dim](override)[/dim]")

    # ── 1. Create work directory ────────────────────────────────────────────
    console.rule()
    work_dir = make_work_dir(base_dir)
    console.print(
        f"\n[bold green]✓ Work directory created:[/bold green] [cyan]{work_dir}[/cyan]\n"
    )

    # ── 2. Download files ───────────────────────────────────────────────────
    console.print("[bold]Downloading files …[/bold]\n")
    status_table = Table(show_header=True, header_style="bold magenta", box=None)
    status_table.add_column("File", style="cyan", width=30)
    status_table.add_column("URL")
    status_table.add_column("Status", justify="center", width=10)

    custom_config = Path(args.configuration).resolve() if args.configuration else None
    files_to_download = [f for f in FILES_TO_DOWNLOAD if not (f == "configuration.json" and custom_config)]

    all_ok = True
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
        console=console,
    ) as progress:
        for fname in files_to_download:
            task = progress.add_task(f"  {fname}", total=None)
            url = base_url + fname
            dest = work_dir / fname
            ok = download_file(url, dest)
            if ok:
                badge = "[green]✓ OK[/green]"
            elif fname in OPTIONAL_FILES:
                badge = "[yellow]– skipped[/yellow]"
            else:
                badge = "[red]✗ FAILED[/red]"
                all_ok = False
            status_table.add_row(fname, url, badge)
            progress.remove_task(task)

    console.print(status_table)

    if custom_config:
        shutil.copy2(custom_config, work_dir / "configuration.json")
        console.print(
            f"[green]✓[/green] Copied custom configuration: "
            f"[cyan]{custom_config}[/cyan] → [cyan]{work_dir / 'configuration.json'}[/cyan]"
        )

    if not all_ok:
        console.print(
            "\n[bold red]✗ One or more downloads failed. Aborting.[/bold red]"
        )
        sys.exit(1)

    # ── 3. Parse stdout.log ─────────────────────────────────────────────────
    stdout_log = work_dir / "stdout.log"

    console.print("\n[bold]Extracting run command …[/bold]")
    run_cmd = extract_run_command(stdout_log)
    if not run_cmd:
        console.print(
            "[bold red]✗ Could not locate the reduced run command in stdout.log.[/bold red]"
        )
        sys.exit(1)
    console.print(
        Panel(
            Text(run_cmd, overflow="fold"),
            title="[bold green]Run Command[/bold green]",
            border_style="green",
            padding=(1, 2),
        )
    )

    # ── 4. Write run.sh ─────────────────────────────────────────────────────
    run_sh = work_dir / "run.sh"
    run_sh.write_text(f"#!/bin/bash\nset -e\n\n{run_cmd}\n")
    run_sh.chmod(0o755)
    console.print(f"\n[green]✓[/green] Written [cyan]{run_sh}[/cyan]")

    # ── 5. Write/copy input_data.txt ────────────────────────────────────────
    input_data = work_dir / "input_data.txt"
    if args.input_data:
        custom_input = Path(args.input_data).resolve()
        shutil.copy2(custom_input, input_data)
        console.print(
            f"[green]✓[/green] Copied custom input data: "
            f"[cyan]{custom_input}[/cyan] → [cyan]{input_data}[/cyan]"
        )
    else:
        console.print("\n[bold]Extracting AliEn input paths …[/bold]")
        alien_paths = extract_alien_paths(stdout_log)
        if not alien_paths:
            console.print("  [yellow]Warning: No AliEn paths found – input_data.txt will be empty.[/yellow]")
        else:
            console.print(f"  [green]Found {len(alien_paths)} path(s):[/green]")
            for p in alien_paths:
                console.print(f"    [cyan]{p}[/cyan]")
        input_data.write_text("\n".join(alien_paths) + ("\n" if alien_paths else ""))
        console.print(f"[green]✓[/green] Written [cyan]{input_data}[/cyan]")

    if args.no_run:
        console.print(
            "\n[yellow]--no-run flag set. Skipping container execution.[/yellow]"
        )
        console.print(f"\n[bold]Work directory:[/bold] {work_dir}")
        sys.exit(0)

    # ── 6. Run in container ─────────────────────────────────────────────────
    console.print("\n[bold]Launching apptainer container …[/bold]\n")
    rc = run_in_container(work_dir)

    if rc == 0:
        console.print(
            Panel(
                f"[bold green]✓ Workflow completed successfully![/bold green]\n"
                f"Work directory: [cyan]{work_dir}[/cyan]",
                border_style="green",
                padding=(1, 4),
            )
        )
    else:
        console.print(
            Panel(
                f"[bold red]✗ Container exited with code {rc}[/bold red]",
                border_style="red",
                padding=(1, 4),
            )
        )
        sys.exit(rc)


if __name__ == "__main__":
    main()
