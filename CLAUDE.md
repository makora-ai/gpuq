# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`gpuq` is a lightweight, multi-vendor GPU querying library for CUDA and HIP. It is a Python package with a small C extension (`gpuq.C`). Its main constraints shape every change:
- **No build-time, install-time or Python runtime dependencies.** `GitPython` is optional and only used for version metadata.
- **Runtime dependencies are soft.** If `libcudart`/`libamdhip64` are missing, the package reports 0 GPUs instead of raising, unless the caller explicitly asks about a provider (`required=...`, `checkcuda()`, etc.).
- **Fully typed and `mypy --strict` clean**, tests included. Python >= 3.10.

## Commands

```bash
pip install -e '.[dev]'            # builds the C extension in place; re-run after editing gpuq/csrc/*
mypy --strict gpuq tests           # type check (CI runs exactly this)
pytest tests/                      # all tests
pytest tests/test_mock.py::test_count   # single test
black gpuq tests                   # format
scripts/pre-commit.sh [-f]         # black (check, or format with -f) + mypy + pytest; needs an active venv
python tests/query_system.py       # print what the real system reports (manual sanity check, not a test)
scripts/sbom.sh [out_dir]          # CycloneDX + SPDX SBOM of the package into sbom/ (needs jq; syft or docker)
```

The C extension is compiled with `-O3 -Werror`, so C warnings break the build. CI (`.github/workflows/test.yaml`) runs mypy and pytest on Python 3.10–3.14 on Ubuntu and macOS. Wheels are built with cibuildwheel only on pushes to `main` and on tags.

## Architecture

**C layer (`gpuq/csrc/`)**: `gpuq.c` defines the Python module and the `Properties` C type (a `PyMemberDef` table over `GpuProp` in `types.h`). `cuda.c` and `amd.c` lazily `dlopen` `libcudart.so` (and `.so.12` / `.so.13`) and `libamdhip64.so` on first use. If loading fails, it is retried on every later call. When the plain soname isn't found, they fall back to "location hints": directory prefixes set from Python through `C._set_location_hints` (limited by `MAX_HINTS`/`MAX_HINT_LEN`). `gpuq/impl.py::_restore_default_hints` sets these at import time to `/opt/cuda`, `/opt/rocm`, and the pip `nvidia/cuda_runtime`, `torch/lib` and `triton/lib` directories. The C side enumerates all devices from all providers under one global ordinal (`ord`). `gpuq/C.pyi` is the hand-written stub for this module; keep it in sync when C members or functions change. It has already drifted a little from the `GpuPropMembers` table in `gpuq.c`.

**Implementation abstraction (`gpuq/impl.py`)**: the `Implementation` ABC has two concrete classes:
- `GenuineImplementation` calls into `gpuq.C`.
- `MockImplementation` fabricates devices from its constructor arguments (`gpuq.mock(...)`). Many of those arguments default to env vars: `MAKO_MOCK_GPU_CUDA`, `MAKO_MOCK_GPU_HIP`, `MAKO_MOCK_GPU_NAME`, `CUDA_VISIBLE_DEVICES`, `HIP_VISIBLE_DEVICES`.

Setting `MAKO_MOCK_GPU` to a truthy value makes the process-wide default implementation a mock. The active implementation is stored per thread (`_current_implementation` in `gpuq/__init__.py`). An `Implementation` works as a context manager (`with G.mock(cuda_count=2): ...`) or through `.set()`. It also has the same `query`/`count`/`get`/... methods as the module-level API.

**Public API (`gpuq/__init__.py`)**: `query`, `count`, `get`, `check*`, `has*`. The logic for visible devices lives here. `impl.save_visible()` temporarily removes `CUDA_VISIBLE_DEVICES`/`HIP_VISIBLE_DEVICES` so the C layer can see every device. Each device's global `index` is then mapped to a local (visible) index. Because of this env mutation the library is not thread-safe, and that is documented behaviour. `query`'s `provider` and `required` arguments are intentionally independent; read its docstring before changing their semantics.

**Datatypes and runtime info**: `gpuq/datatypes.py` defines the `Provider` flag enum and the Python `Properties` wrapper around the C object. `gpuq/cuda.py` and `gpuq/hip.py` provide lazy runtime info (utilisation, memory, PIDs) by parsing `nvidia-smi`/`rocm-smi` output with regexes. Each has a `*Mock` counterpart used by `MockImplementation`.

**Versioning**: `gpuq/version.py` holds the version string; bump it there. Repo and commit info comes from GitPython when running from a checkout. The custom `sdist`/`build_py` commands in `setup.py` write a temporary `gpuq/_dist_info.py` so that installed packages keep this metadata.

## Testing notes

- `tests/conftest.py` has an autouse fixture that asserts `gpuq._default_impl is None` before each test and resets both the default and the thread-local implementation afterwards. Tests must not leave global implementation state behind.
- Most tests use `G.mock(...)` or patch `gpuq.cuda` / `gpuq.hip` helpers (for example, a mocked `nvidia-smi` output) so they pass on machines without GPUs, which is how CI runs.

## SBOM (WEB-1164)

- **What:** `scripts/sbom.sh` pip-installs the package as it ships (no `[dev]` extra) into a temp dir and scans it with syft v1.55.0. It writes `sbom/gpuq.cdx.json` and `sbom/gpuq.spdx.json` and prints a license summary. gpuq has no runtime Python deps, so the SBOM lists only `pkg:pypi/gpuq`. The CUDA/HIP runtimes are `dlopen`ed system libraries and are not listed. Its license comes from `license=` in `setup.py`. Without that line the component has no license.
- **CI:** the `sbom_sources` job in `test.yaml` runs on every push and PR and uploads `sbom/` as the `sbom-sources-<sha>` artifact (90 days). On pushes to `main` only, `dependency_track_upload` joins the tailnet and uploads the CycloneDX file to Dependency-Track project **`gpuq`**, version `main`. It is best effort (`continue-on-error`) and nothing depends on it. It uses the org-wide `DEPENDENCYTRACK_URL`, `DEPENDENCYTRACK_API_KEY` and `TS_OAUTH_*`.
- **No image:** the repo builds no container, so there is no image SBOM or attestation.
- **Pitfalls:** keep `SYFT_FILE_METADATA_SELECTION=none` (otherwise syft adds one component per file). Dependency-Track ingests CycloneDX only, not SPDX. When syft runs in its container as a non-root user it needs a writable `HOME`/`TMPDIR` (the script mounts a tmpfs). Without one it silently writes an empty file under `-q`.
