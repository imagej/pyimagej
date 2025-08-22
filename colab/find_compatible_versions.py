#!/usr/bin/env python3
"""
find_compatible_versions.py

Try to find compatible versions of a small list of top-level packages by repeatedly
installing candidate versions inside short-lived virtual environments and running
basic import tests.

Intended to be run inside Google Colab (or any Linux VM with internet access and
`python3 -m venv`).

This is a pragmatic, best-effort tool — it tries a small candidate set per package
and reports which versions successfully install and import together.

Usage (basic):
    python3 find_compatible_versions.py

Options:
    --max-versions N    number of candidate versions to try per package (default 5)
    --use-colab-tf      detect and fix TensorFlow to the Colab runtime version (recommended)
    --packages pkg1,pkg2  comma-separated list to override the default packages

"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from typing import List, Dict

# Top-level packages we care about. You can override via --packages.
DEFAULT_PACKAGES = ["tensorflow", "csbdeep", "stardist", "cellpose", "pyimagej"]

# Map package -> import test snippet (a Python statement to run inside the venv)
IMPORT_TESTS = {
    "tensorflow": "import tensorflow as tf; print('tf', tf.__version__)",
    "csbdeep": "import csbdeep; print('csbdeep', getattr(csbdeep, '__version__', 'unknown'))",
    "stardist": "import stardist; print('stardist', getattr(stardist, '__version__', 'unknown'))",
    "cellpose": "import cellpose; print('cellpose', getattr(cellpose, '__version__', 'unknown'))",
    "pyimagej": "import imagej; print('pyimagej', getattr(imagej, '__version__', 'unknown'))",
}

PYPI_URL = "https://pypi.org/pypi/{package}/json"


def fetch_versions(package: str, max_versions: int) -> List[str]:
    try:
        with urllib.request.urlopen(PYPI_URL.format(package=package)) as r:
            data = json.load(r)
        versions = list(data.get("releases", {}).keys())
        # sort loosely by version string (newest last in semantic order is hard—keep simple)
        versions.sort(key=lambda s: tuple(int(x) if x.isdigit() else x for x in s.replace('-', '.').split('.')))
        return versions[-max_versions:][::-1]
    except Exception as e:
        print(f"Warning: couldn't fetch versions for {package}: {e}")
        return []


def run_in_venv(commands: List[str], venv_dir: str) -> subprocess.CompletedProcess:
    # Build a shell command that activates venv and runs the commands
    # We use bash -lc so activation script works.
    joined = " && ".join(commands)
    shell_cmd = f". {venv_dir}/bin/activate && {joined}"
    return subprocess.run(shell_cmd, shell=True, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def create_venv(tmpdir: str) -> str:
    venv_dir = os.path.join(tmpdir, "venv")
    subprocess.check_call([sys.executable, "-m", "venv", venv_dir])
    return venv_dir


def try_install_and_import(venv_dir: str, install_pkgs: List[str], import_test: str, pip_extra: List[str]=None) -> Dict:
    pip_extra = pip_extra or []
    cmds = [f"python -m pip install -U pip setuptools wheel"]
    cmds.append("python -m pip install -U %s" % (" ".join(install_pkgs + pip_extra)))
    # run import test
    cmds.append(f"python - <<'PY'\n{import_test}\nPY")
    result = run_in_venv(cmds, venv_dir)
    return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-versions", type=int, default=5)
    parser.add_argument("--use-colab-tf", action='store_true')
    parser.add_argument("--packages", type=str, default=','.join(DEFAULT_PACKAGES))
    args = parser.parse_args()

    packages = [p.strip() for p in args.packages.split(',') if p.strip()]

    # If requested, detect TF version in the current runtime and lock it.
    fixed_tf_version = None
    if args.use_colab_tf:
        try:
            import tensorflow as tf
            fixed_tf_version = tf.__version__
            print(f"Detected TensorFlow in host runtime: {fixed_tf_version}")
        except Exception:
            print("No host TensorFlow available to detect; continuing without fixed TF")

    results = {"tried": {}, "successful": {}}

    # We'll iterate packages and try to pick a working version for each relative to already chosen pins.
    chosen_pins = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        for pkg in packages:
            print(f"\n=== Trying package {pkg} ===")
            results['tried'][pkg] = []
            candidates = []
            if pkg == 'tensorflow' and fixed_tf_version:
                candidates = [fixed_tf_version]
            else:
                candidates = fetch_versions(pkg, args.max_versions)
                if not candidates:
                    print(f"No candidates found for {pkg}; skipping")
                    continue

            success_for_pkg = None
            for ver in candidates:
                print(f"- Testing {pkg}=={ver}")
                # create fresh venv for each attempt to avoid cross-contamination
                attempt_dir = os.path.join(tmpdir, f"attempt_{pkg}_{ver}")
                os.makedirs(attempt_dir, exist_ok=True)
                venv_dir = create_venv(attempt_dir)

                # Build install list: previously chosen pins + this package==ver
                install_list = [f"{k}=={v}" for k, v in chosen_pins.items()]
                install_list.append(f"{pkg}=={ver}")

                # For pyimagej we may need apt packages (java). We still attempt pip install and will report if import fails.
                import_test = IMPORT_TESTS.get(pkg, f"import {pkg}; print('imported')")

                try:
                    res = try_install_and_import(venv_dir, install_list, import_test)
                except Exception as e:
                    res = {"returncode": 1, "stdout": "", "stderr": str(e)}

                res_entry = {"version": ver, "returncode": res['returncode'], 'stdout': res['stdout'], 'stderr': res['stderr']}
                results['tried'][pkg].append(res_entry)

                if res['returncode'] == 0:
                    # success: record pin and stop searching for this package
                    chosen_pins[pkg] = ver
                    results['successful'][pkg] = ver
                    success_for_pkg = ver
                    print(f"SUCCESS: {pkg}=={ver}")
                    break
                else:
                    print(f"FAILED: {pkg}=={ver} -> returncode {res['returncode']}")
                    # fallback: continue to next candidate

            if not success_for_pkg:
                print(f"No working version found (within candidates) for {pkg}. Check stderr of attempts to diagnose.")

    # Print summary JSON
    out = {
        "chosen_pins": chosen_pins,
        "details": results
    }
    print('\n=== Summary (JSON) ===')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
