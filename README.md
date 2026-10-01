# aliceutils

Small utility scripts for analysis workflows.

## `hyperlooptraintest.py`

Downloads an AliHyperloop train-test bundle from alimonitor, prepares a local runnable
directory, and executes the test – either inside an Apptainer container using the exact
same software as on Hyperloop, or with a locally installed package via `alienv`.
The container image used is [`alisw/slc9-builder`](https://hub.docker.com/r/alisw/slc9-builder).
The goal is to have an easy and reproducible environment for local debugging and execution.
Using the same software stack as the test machine is achieved by running in an EL9 container
and loading the same environment variables as the train test, which point to CVMFS.

### Quickstart

**Run with the exact Hyperloop software** (requires `apptainer` and `/cvmfs`):

```bash
python hyperlooptraintest.py https://alimonitor.cern.ch/train-workdir/tests/0063/00632029/
```

**Run with a local software package** (no `apptainer` or `/cvmfs` needed):

```bash
python hyperlooptraintest.py \
  https://alimonitor.cern.ch/train-workdir/tests/0063/00632029/ \
  --local --package O2Physics
```

**Submit as a Slurm job** (instead of running immediately):

```bash
python hyperlooptraintest.py \
  https://alimonitor.cern.ch/train-workdir/tests/0063/00632029/ \
  --sbatch
```

**Re-run an existing local work directory** (no download/regeneration):

```bash
python hyperlooptraintest.py /home/fjonas/aliceutils/traintest_20260513_135606
```

**Help**
```bash
python hyperlooptraintest.py --help
```

The script auto-bootstraps its own virtual environment at `.venv_hyperloop` and installs
`requests` + `rich` on first run.

### What it does

Given a train-test URL, the script:

1. creates a fresh `traintest_<id>` working directory
2. downloads required files (`stdout.log`, `configuration.json`, `env.sh`;
   `OutputDirector.json` is optional)
3. extracts the full train command from `stdout.log` (falls back to the reduced command if needed) and writes it to `run.sh`
4. extracts AliEn input paths into `input_data.txt` (or uses your override file)
5. optionally rewrites parent-file resolution for derived inputs when `--derived` is set
6. executes `env.sh` + `run.sh`, or submits them via `sbatch` with `--sbatch`

Given a local `traintest_*` directory path instead of a URL, the script:

1. reuses the directory as-is
2. does not download or regenerate files
3. runs existing `env.sh` + `run.sh` (or submits with `--sbatch`)

### Prerequisites

- Python 3
- `apptainer` in `PATH` and access to `/cvmfs` — **only required without `--local`**  
  (the container run binds `/cvmfs:/cvmfs`)
- `alienv` available — **only required with `--local`**
- `sbatch` available — **only required with `--sbatch`**

### Using a local software package (`--local`)

When `--local` is set, `env.sh` is not downloaded from alimonitor but generated locally:

1. calls `alienv list` to find all installed versions of the requested package family
2. presents an interactive numbered list – pick the version you want
3. runs `alienv printenv <selected-tag>` and writes the result as `env.sh`

The test is then run directly in the current shell environment instead of inside an
Apptainer container, so neither `apptainer` nor `/cvmfs` are needed.

If `--package` is omitted you will be prompted to enter the package family at runtime.

### Options

| Flag | Description |
|---|---|
| `--local` | With URL source: generate `env.sh` from local `alienv` instead of downloading it. With local-directory source: run in local shell instead of container. |
| `--sbatch` | Submit the prepared execution as a Slurm job (`sbatch`) instead of running it immediately. |
| `--package PKG` | Package family to search for with `--local` (e.g. `O2Physics`). Prompted if omitted. |
| `--no-run` | Skip execution (`URL` source still performs preparation/download). |
| `--workdir DIR` | Base directory where the `traintest_<id>` folder is created (default: current directory). |
| `--configuration FILE` | Use a local `configuration.json` instead of downloading one. |
| `--input-data FILE` | Use a local file as `input_data.txt` instead of extracting AliEn paths from `stdout.log`. |
| `--aod-memory-rate-limit-mb MB` | Override `--aod-memory-rate-limit` in the run command (value provided in MB). |
| `--derived` | Force derived-input mode: parent file lookup stays enabled and parent paths are resolved via AliEn. |

### Notes

- The positional `source` argument is auto-detected as either URL or local directory path.
- The script normalizes the URL and switches `https://alimonitor.cern.ch/...` to `http://...`
- Each run creates a new timestamped working folder; existing folders are never overwritten.
- With `--sbatch`, the script writes `run.sbatch` in the work directory and submits it.
- Derived-input handling is opt-in: by default behavior is unchanged, and only `--derived` enables parent-path rewriting to fetch parents via AliEn.

## `check_rct_flags.py`

Looks up RCT quality flags in CCDB (`RCT/Flags/RunFlags`) for each reconstruction pass of a run list. Prints a table in the terminal, a summary of how many runs have each flag set, and writes the same report to an HTML file.

Requires the ALICE environment so that `root` is on `PATH`, plus the `rich` package (already installed with the O2 Python modules and in `.venv_hyperloop`).

### Quickstart

```bash
eval $(alienv -w /home/fjonas/alice/sw --no-refresh printenv O2Physics/latest)
python3 check_rct_flags.py rct_example_runs.txt
```

Run numbers can also be passed directly:

```bash
python3 check_rct_flags.py --runs "535069, 535084"
```

The run file accepts commas, whitespace, and `#` comments. The latest CCDB version is used for each run and pass. A run counts toward a flag when that flag is set on any pass, including for only part of the run.

### Options

| Flag | Description |
|---|---|
| `run_file` | Text file of run numbers. |
| `--runs` | Comma- or space-separated run numbers. Can be combined with `run_file`. |
| `--html FILE` | HTML report path (default: `rct_flags.html`). |
