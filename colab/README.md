Colab compatibility helper
=========================

Purpose
-------
This folder contains a small tool and instructions to help find a set of package versions for running a notebook that uses TensorFlow, Cellpose, Stardist, CSBDeep and PyImageJ in Google Colab. The tool automates attempts to install candidate versions inside short-lived virtualenvs and runs basic import checks to report which combinations succeed.

When to use
-----------
Use this when you need a reproducible, pinned set of PyPI package versions that actually import successfully together on a Colab-like Linux environment. You should run the tool inside a Colab runtime (or any Linux VM with internet access).

Files
-----
- `find_compatible_versions.py` — Python script that iteratively tries candidate versions for specified top-level packages inside temporary virtual environments and reports successful pins.
- `colab_install_cell.txt` — a copy-paste block you can drop into a Colab cell to prepare the runtime, detect the active TensorFlow version, and run the pinning tool.

Quick workflow
--------------
1. Open a new Colab notebook (Runtime -> Change runtime type -> GPU if you need it).
2. Paste the contents of `colab/colab_install_cell.txt` into a cell and run it. This will install system Java (for PyImageJ), upgrade pip, fetch the helper script, and run it.
3. Follow the prompts or examine the printed JSON to see working versions for each package. Use the successful pins to create a `requirements.txt` and re-run the Colab cell that installs those exact pinned versions.

Notes and caveats
-----------------
- PyImageJ typically requires Java on the system. The tool will attempt `apt` installation in Colab; if you run the script in a non-root environment without `apt` you must install Java yourself.
- The script focuses on top-level importability (i.e., `import package`). Some packages have heavier runtime checks (GPU/CUDA) that are not verified by the script.
- This tool attempts only a limited number of candidate versions per package (to keep runs reasonable). If no compatible version is found, increase the candidate count inside `find_compatible_versions.py`.

License
-------
MIT
