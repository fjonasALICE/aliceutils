# aliceutils

Small utility scripts for analysis workflows.

## `hyperlooptraintest.py`

Downloads an AliHyperloop train-test bundle from alimonitor, prepares a local runnable
directory, and executes the test – either inside an Apptainer container using the exact
same software as on Hyperloop, or with a locally installed package via `alienv`.
The container image used is [`alisw/slc9-builder`](https://hub.docker.com/r/alisw/slc9-builder).
The goal is to have an easy and reproducable environment for local debugging and running.
Using the same software stack as the testmachine is achieved by creating a el9 virtual machine and loading the same environment variables as the traintest, which are pointing to cvmfs.
The container with el9 is needed to ensure that all the cvmfs packages work correctly.

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
3. extracts the reduced run command from `stdout.log` and writes it to `run.sh`
4. extracts AliEn input paths into `input_data.txt` (or uses your override file)
5. executes `env.sh` + `run.sh`

### Prerequisites

- Python 3
- `apptainer` in `PATH` and access to `/cvmfs` — **only required without `--local`**  
  (the container run binds `/cvmfs:/cvmfs`)
- `alienv` available — **only required with `--local`**

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
| `--local` | Generate `env.sh` from a locally installed package via `alienv` instead of downloading it. |
| `--package PKG` | Package family to search for with `--local` (e.g. `O2Physics`). Prompted if omitted. |
| `--no-run` | Prepare files only, skip execution. |
| `--workdir DIR` | Base directory where the `traintest_<id>` folder is created (default: current directory). |
| `--configuration FILE` | Use a local `configuration.json` instead of downloading one. |
| `--input-data FILE` | Use a local file as `input_data.txt` instead of extracting AliEn paths from `stdout.log`. |

### Notes

- The script normalizes the URL and switches `https://alimonitor.cern.ch/...` to `http://...`
- Each run creates a new timestamped working folder; existing folders are never overwritten.
