# testdrive-dl-on-macmini

A personal passion project for testing Python deep-learning and convolutional
neural network (CNN) projects on a Mac Mini. The experiments use PyTorch,
torchvision, torchaudio, TensorFlow, Keras, and related machine-learning
libraries.

The repository is intended to be opened and run from Visual Studio Code. Conda
provides isolated environments so the projects do not depend on the host
Python installation.

The projects are learning exercises based on published books and e-books. They
are not presented as original implementations or production applications.

## Repository structure

```text
testdrive-dl-on-macmini/
├── .gitignore
├── README.md
├── CONDA_ENV_TORCH_VISION_AUDIO.YML
├── CONDA_ENV_TORCH_VISION_AUDIO-PYTHON3.11.YML
└── testdrive-cnn/
    ├── dummies-dl-cnn/
    ├── dummies-ml-cnn/
    ├── easysteps-cnn/
    ├── hello-cnn/
    ├── nostarch-cnn/
    ├── packtpub-deep-cnn/
    └── packtpub-cnn/
```

The `testdrive-cnn` directory contains the initial CNN learning and reference
projects. The project names identify different learning tracks or source
materials; they are not intended to imply production applications.

## Conda environment strategy

The preferred approach is to maintain YAML files as portable module lists, not
as complete exports of one machine's solved environment. A portable YAML should
describe the direct packages that the projects need and allow Conda to resolve
platform-specific and Python-version-specific transitive dependencies.

In practice:

- Do not copy every package from `conda env export` into a project YAML.
- Avoid exact build strings and platform-specific packages such as `cpython`,
  `libpython`, `python-gil`, and `python_abi`.
- Leave Python unpinned when the package set supports several Python 3.x
  versions.
- Use a range such as `python>=3.10,<3.14` when a project needs a supported
  Python family but should not select Python 3.14.
- Create a targeted YAML such as
  `CONDA_ENV_TORCH_VISION_AUDIO-PYTHON3.11.YML` only when a framework or dependency
  requires a particular Python version.
- Keep TensorFlow-specific constraints isolated from the general PyTorch/CNN
  environment when possible.

The goal is for a module-list YAML to work across Python 3.x versions without
unnecessary edits. This cannot always be guaranteed: TensorFlow, PyTorch,
compiled scientific packages, and older book examples may support different
Python ranges. When the solver cannot find a compatible set, document the
specific Python version in the YAML filename and comments.

The original environment file was exported from Python 3.14 and included
Python-3.14-specific transitive packages. Those entries should not be copied
into a portable environment or mechanically changed to Python 3.11.

### Recommended YAML families

Use names that communicate the role of each file:

```text
CONDA_ENV_CNN.YML                         # portable direct module list
CONDA_ENV_TORCH_VISION_AUDIO.YML          # portable PyTorch-focused list
CONDA_ENV_TORCH_VISION_AUDIO-PYTHON3.11.YML # Python-targeted compatibility file
CONDA_ENV_TENSORFLOW-PYTHON3.11.YML       # TensorFlow compatibility file
```

The portable file should contain direct dependencies with minimal or no
version pins. A targeted file may pin Python and selected framework versions,
but should still avoid exporting the entire transitive environment.

For example:

```yaml
name: testdrive-dl-mm
channels:
  - conda-forge
  - defaults
dependencies:
  - python
  - pip
  - numpy
  - scipy
  - pandas
  - matplotlib
  - scikit-learn
  - pillow
  - pytorch
  - torchvision
  - torchaudio
  - tqdm
  - pip:
      - tensorflow
      - keras
```

If TensorFlow fails to solve or install with the otherwise portable module
list, create a TensorFlow-specific YAML with the Python version required by
the selected TensorFlow release. The targeted file is a compatibility record,
not the preferred default.

## Initial setup on macOS with VS Code

### Install prerequisites

Install Git, Miniconda or Anaconda, Visual Studio Code, the VS Code Python
extension, and the Jupyter extension if notebooks will be used. On an Apple
Silicon Mac Mini, use installers and packages compatible with the machine's
architecture.

### Clone the repository

```shell
git clone <repository-url>
cd testdrive-dl-on-macmini
```

Replace `<repository-url>` with the HTTPS or SSH URL for this GitHub
repository.

### Create the Conda environment

Check existing environments:

```shell
conda env list
```

From the repository root, prefer the portable module-list YAML:

```shell
conda env create -f ./CONDA_ENV_TORCH_VISION_AUDIO.YML
```

If the portable file cannot solve because TensorFlow or another framework
requires an older Python version, use the targeted file:

```shell
conda env create --no-pin -f ./CONDA_ENV_TORCH_VISION_AUDIO-PYTHON3.11.YML
```

Activate the environment:

```shell
conda activate testdrive-dl-mm
```

If the environment already exists and the YAML changed, update it with:

```shell
conda env update -n testdrive-dl-mm -f ./CONDA_ENV_TORCH_VISION_AUDIO.YML --prune
```

Use `--prune` carefully: it removes packages installed in the environment
that are no longer listed in the YAML.

### Select the environment in VS Code

```shell
code .
```

In VS Code:

1. Press `Ctrl+Shift+P`.
2. Select **Python: Select Interpreter**.
3. Select the `testdrive-dl-mm` Conda environment.
4. For notebooks, select the same environment as the notebook kernel.

The interpreter shown in the status bar should be the project environment,
not `base`.

## Verify the environment

```shell
python --version
python -c "import torch, torchvision, torchaudio, tensorflow, keras, tqdm; print('PyTorch:', torch.__version__); print('TensorFlow:', tensorflow.__version__); print('Keras:', keras.__version__); print('tqdm:', tqdm.__version__)"
```

A PyTorch hardware check is also useful:

```shell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

`CUDA available: False` is expected for a CPU-only environment.

## Running a CNN project

Activate the environment before running project code:

```shell
conda activate testdrive-dl-mm
cd ./testdrive-cnn/hello-cnn
python ./main.py
```

The actual script name varies by project. For notebooks, open the notebook in
VS Code and select the `testdrive-dl-mm` kernel.

Keep generated artifacts out of Git unless they are intentionally part of the
project. This normally includes downloaded datasets, model checkpoints, cache
directories, logs, notebook checkpoints, and large generated media files.

## Conda environment lifecycle

### Inspect environments and packages

```shell
conda env list
conda list
conda list -n testdrive-dl-mm
conda list -n testdrive-dl-mm "^requests$"
conda list pefile
conda list "^pefile$"
```

The regular package search may show related names. The expression surrounded by
`^` and `$` matches only the exact package name.

### Install or update packages

With the environment activated, use Conda for Conda-managed packages:

```shell
conda install <package-name>
conda update <package-name>
```

For packages installed through the YAML `pip:` section:

```shell
python -m pip install <package-name>
python -m pip list
```

Using `python -m pip` ensures that pip belongs to the active Conda
environment. Updating everything at once can introduce unrelated changes:

```shell
conda update --all
```

Use that broader update only when all projects can be retested afterward.

### Export the environment

Full export, including transitive dependencies and build information:

```shell
conda env export -n testdrive-dl-mm > environment-full.yml
```

Export without exact Conda build strings:

```shell
conda env export -n testdrive-dl-mm --no-builds > environment-no-builds.yml
```

Export only packages explicitly requested in Conda history:

```shell
conda env export -n testdrive-dl-mm --from-history > environment-history.yml
```

`--from-history` may omit pip-installed packages and indirect dependencies.
Review the result before using it as the project YAML.

For this repository, use exports as diagnostic or review artifacts. Update the
portable module-list YAML intentionally rather than replacing it with a full
export. If a Python-specific workaround is required, update the targeted YAML
and explain the compatibility reason in a comment or commit message.

### Recreate or clone an environment

```shell
conda env create -f environment.yml
conda env create -f environment.yml -n myenv2
conda activate myenv2
```

Create from the repository's portable environment file:

```shell
conda env create -f ./CONDA_ENV_TORCH_VISION_AUDIO.YML
```

### Remove an environment

```shell
conda remove -n test_env --all
```

To remove this project's environment:

```shell
conda deactivate
conda remove -n testdrive-dl-mm --all
```

This removes the Conda environment, not the Git repository or its YAML files.

## Conda cache cleanup

```shell
conda clean --packages     # unused package caches
conda clean --tarballs     # downloaded package archives
conda clean --index-cache  # repodata index cache
conda clean --all          # everything above
```

Review the proposed cleanup before confirming, especially if rebuilding
environments would be expensive.

## Troubleshooting Python compatibility

If Conda reports incompatible Python versions, inspect the YAML and current
Conda configuration:

```shell
grep -nE "python|3\.14|tensorflow" ./CONDA_ENV_TORCH_VISION_AUDIO*.YML
conda config --show pinned_packages
```

Check for a Python pin, an old TensorFlow pin, or copied transitive packages
such as `cpython`, `libpython`, `python-gil`, or `python_abi`.

If a persistent pin such as `python=3.14` is displayed, either use
`--no-pin` for a targeted environment or remove the exact pin:

```shell
conda config --remove pinned_packages "python=3.14"
```

If VS Code uses the wrong interpreter, check the active executable:

```shell
conda activate testdrive-dl-mm
python -c "import sys; print(sys.executable)"
```

Then run **Python: Select Interpreter** and choose that environment.

## Attribution and project scope

This is a personal learning and experimentation project. The CNN projects are
based on examples, concepts, and exercises from published books or e-books.
They are not my original source material. Retain the relevant author,
publisher, book, and license or copyright notices in each project directory.
Do not redistribute book text, proprietary datasets, or other copyrighted
materials that are not permitted for redistribution.

When adapting an example, document the source in the project README or a
source-notes file. Clearly distinguish original changes, experiments, and
observations from the published material.

## Git workflow

Before committing environment changes:

```shell
git status
git diff
```

Commit the README, YAML, and code changes when they are ready:

```shell
git add README.md CONDA_ENV_TORCH_VISION_AUDIO.YML
git commit -m "Document portable Conda setup and project workflow"
git push
```

Keep commits focused. A package update, a new CNN project, and documentation
can be separate commits when that makes the project history easier to follow.

## Recommended lifecycle

1. Clone or pull the repository.
2. Create the portable module-list environment YAML.
3. Select it as the VS Code interpreter and notebook kernel.
4. Run and test the CNN projects.
5. Add or update packages only when required.
6. Verify the environment and rerun affected projects.
7. Export a review copy if dependencies changed.
8. If compatibility requires it, create or update a targeted Python-version
   YAML and record the reason.
9. Update the portable YAML intentionally rather than importing a complete
   machine-specific export.
10. Commit related YAML, documentation, and code changes.
11. Periodically remove and recreate the environment to confirm setup is
    reproducible.
