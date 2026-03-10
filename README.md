# aliceutils

Small utility scripts for analysis workflows.

## `hyperlooptraintest.py`

`hyperlooptraintest.py` downloads an AliHyperloop train-test bundle from alimonitor,
prepares a local runnable directory, and optionally executes the test inside an
Apptainer container.

### What it does

Given a train-test URL, the script:
- creates a fresh `traintest_<id>` working directory
- downloads required files (`stdout.log`, `configuration.json`, `env.sh`; `OutputDirector.json` is optional)
- extracts the reduced run command from `stdout.log` and writes it to `run.sh`
- extracts AliEn input paths into `input_data.txt` (or uses your override file)
- runs `env.sh` + `run.sh` inside `el9.sif` with Apptainer (unless `--no-run` is used)

The script auto-bootstraps its own virtual environment at
`.venv_hyperloop` and installs `requests` + `rich` on first run.

### Prerequisites

- Python 3
- `apptainer` available in `PATH`
- access to `/cvmfs` (the container run binds `/cvmfs:/cvmfs`)

### Basic usage

```bash
python hyperlooptraintest.py https://alimonitor.cern.ch/train-workdir/tests/0063/00632029/
```

Notes:
- the script normalizes the URL and switches `https://alimonitor.cern.ch/...` to `http://...`
- each run creates a new working folder in the current directory by default

### Useful options

- `--no-run`  
  Prepare files only, skip container execution.
- `--workdir DIR`  
  Base directory where the `traintest_<id>` folder is created.
- `--configuration FILE`  
  Use a local `configuration.json` instead of downloading one.
- `--input-data FILE`  
  Use a local file as `input_data.txt` instead of extracting paths from `stdout.log`.

Example:

```bash
python hyperlooptraintest.py \
  https://alimonitor.cern.ch/train-workdir/tests/0063/00632029/ \
  --workdir /tmp \
  --no-run
```
