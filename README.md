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

### Using a local software package

Instead of using the `env.sh` downloaded from the train-test URL, you can source a locally
installed package via `alienv`. Pass `--local` together with `--package <family>`:

```bash
python hyperlooptraintest.py \
  https://alimonitor.cern.ch/train-workdir/tests/0063/00632029/ \
  --local --package O2Physics
```

When `--local` is set the script:

1. skips downloading `env.sh` from alimonitor
2. calls `alienv list` to find all installed versions of the requested package family
3. presents an interactive numbered list – pick the version you want
4. runs `alienv printenv <selected-tag>` and writes the result as `env.sh` in the work directory

If you omit `--package` you will be prompted to enter the package family at runtime.

### Useful options

- `--no-run`  
  Prepare files only, skip container execution.
- `--workdir DIR`  
  Base directory where the `traintest_<id>` folder is created.
- `--configuration FILE`  
  Use a local `configuration.json` instead of downloading one.
- `--input-data FILE`  
  Use a local file as `input_data.txt` instead of extracting paths from `stdout.log`.
- `--local`  
  Generate `env.sh` from a locally installed package via `alienv` (see above).
- `--package PKG`  
  Package family to search for when `--local` is set (e.g. `O2Physics`).

Example – prepare only, using a local package build:

```bash
python hyperlooptraintest.py \
  https://alimonitor.cern.ch/train-workdir/tests/0063/00632029/ \
  --local --package O2Physics \
  --workdir /tmp \
  --no-run
```
