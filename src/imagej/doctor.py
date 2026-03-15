"""
The PyImageJ doctor provides standalone tools for diagnosing your
environment configuration, and offers advice for correcting problems.

To run the diagnostics:

    import imagej.doctor
    imagej.doctor.checkup()

To enable debug-level logging:

    import imagej.doctor
    imagej.doctor.debug_to_stderr()
    import imagej
    ij = imagej.init()
"""

import importlib
import importlib.metadata
import logging
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path


def _execute(command):
    try:
        return (
            subprocess.check_output(command, stderr=subprocess.STDOUT).decode().rstrip()
        )
    except Exception as e:
        return str(e)


def _check_url(url, timeout=5):
    """Try to reach a URL, returning True if the host is accessible.

    An HTTP error response (e.g. 403, 404) still counts as accessible,
    since the server responded. Only a connection failure or timeout means
    the host is unreachable.
    """
    try:
        urllib.request.urlopen(url, timeout=timeout)
        return True
    except urllib.error.HTTPError:
        return True  # Server responded; host is reachable
    except Exception:
        return False


def checkup(output=print):
    """
    Check your environment for health problems that could prevent PyImageJ from
    functioning.
    """
    output("")
    advice = []

    # -- Python --

    output("Checking Python:")
    output(f"--> Python {sys.version}")
    output(f"--> Python executable = {sys.executable}")
    output("")

    # -- Environment --

    output("Checking environment:")

    if "CONDA_PREFIX" in os.environ:
        conda_prefix = os.environ["CONDA_PREFIX"]
        output(f"--> CONDA_PREFIX = {conda_prefix}")
        actual_exe = Path(sys.executable).resolve()
        expected_exes = [
            (Path(conda_prefix) / "bin" / "python").resolve(),
            (Path(conda_prefix) / "python.exe").resolve(),
        ]
        if actual_exe in expected_exes:
            output("--> Python executable matches Conda environment.")
        else:
            output("--> Python executable is NOT from that Conda environment!")
            indent = "\n    * "
            advice.append(
                "Are you sure you're using the correct Python executable? "
                "I expected one of these:"
                + indent
                + indent.join(map(str, expected_exes))
            )
        if not shutil.which("mamba") and not shutil.which("micromamba"):
            advice.append(
                "Consider using mamba or micromamba instead of conda "
                "for faster package installation: https://mamba.readthedocs.io/"
            )
    elif "VIRTUAL_ENV" in os.environ:
        output(f"--> VIRTUAL_ENV = {os.environ['VIRTUAL_ENV']}")
        output("--> Running inside a Python virtual environment.")
    else:
        output("--> No virtual environment detected (no CONDA_PREFIX or VIRTUAL_ENV).")
        advice.append(
            "No virtual environment is active. Using a Conda environment or "
            "Python venv is recommended to avoid dependency conflicts."
        )
    output("")

    # -- Python dependencies --

    output("Checking Python dependencies:")

    dependencies = {
        "jgo": "jgo",
        "scyjava": "scyjava",
        "imglyb": "imglyb",
        "jpype1": "jpype",
        "pyimagej": "imagej",
    }
    for package_name, module_name in dependencies.items():
        try:
            m = importlib.import_module(module_name)
            try:
                version = importlib.metadata.version(package_name)
                output(f"--> {package_name} {version}: {m.__file__}")
            except importlib.metadata.PackageNotFoundError:
                output(f"--> {package_name}: {m.__file__}")
        except ImportError:
            output(f"--> {package_name}: MISSING")
            advice.append(f"Are you sure the {package_name} package is installed?")

    output("")

    # -- Java --

    output("Checking Java:")

    java_home = os.environ.get("JAVA_HOME")
    if java_home:
        output(f"--> JAVA_HOME = {java_home}")
        if not Path(java_home).is_dir():
            output(f"--> JAVA_HOME directory does NOT exist!")
            advice.append(
                f"JAVA_HOME points to a non-existent path: {java_home}. "
                "Fix or unset JAVA_HOME."
            )
    else:
        output("--> JAVA_HOME is NOT set!")
        advice.append(
            "Activate a conda environment with openjdk installed, "
            "or set JAVA_HOME manually."
        )

    java_executable = shutil.which("java")
    output(f"--> Java executable = {java_executable or 'NOT FOUND!'}")
    if not java_executable:
        advice.append("Install openjdk using conda or your system package manager.")

    output(f"$ java -version\n{_execute(['java', '-version'])}")
    output("")

    # -- Maven artifact cache --

    output("Checking Maven artifact cache:")

    try:
        import jgo.config

        settings = jgo.config.GlobalSettings()
        settings_dict = settings.to_dict()

        jgo_cache = settings_dict.get("cache_dir")
        output(f"--> jgo cache directory = {jgo_cache}")
        if jgo_cache and not Path(jgo_cache).exists():
            output("--> jgo cache does not exist yet; it will be created on first use.")

        m2_repo = settings_dict.get("repo_cache")
        output(f"--> Maven local repository = {m2_repo}")
        if not m2_repo or not Path(m2_repo).exists():
            output(
                "--> Maven local repository does not exist yet; "
                "it will be populated on first use."
            )
    except ImportError:
        pass  # jgo missing; already reported above

    output("")

    # -- Network --

    output("Checking network access to Maven repositories:")

    maven_repos = [
        ("Maven Central", "https://repo.maven.apache.org/maven2"),
        ("SciJava Maven", "https://maven.scijava.org/content/groups/public"),
    ]
    for name, url in maven_repos:
        reachable = _check_url(url)
        status = "OK" if reachable else "NOT accessible!"
        output(f"--> {name} ({url}): {status}")
        if not reachable:
            advice.append(
                f"{name} at {url} is not accessible. "
                "Check your internet connection or firewall/proxy settings."
            )

    output("")

    if advice:
        output("Questions and advice for you:")
        for line in advice:
            output(f"--> {line}")
    else:
        output("Great job! All looks good.")


def debug_to_stderr(logger=None, debug_maven=False):
    """
    Enable debug logging to the standard error stream.

    :param logger: The logger for which debug logging should go to stderr,
                   or None to enable it for all known loggers across
                   PyImageJ's dependency stack (e.g.: jgo, imglyb, scyjava).
    :param debug_maven: Enable Maven debug logging. It's very verbose,
                        so this flag is False by default, but if jgo is having
                        problems resolving the environment, such as failure to
                        download needed JAR files, try setting this flag to
                        True for more details on where things go wrong.
    """
    if logger is None:
        debug_to_stderr(logging.getLogger("jgo"))
        debug_to_stderr("scyjava._logger")
        debug_to_stderr("scyjava.config._logger")
        debug_to_stderr("imagej._logger")
        debug_to_stderr("imagej.dims._logger")
        if debug_maven:
            # Tell scyjava to tell jgo to tell Maven to enable
            # debug logging via its -X command line flag.
            try:
                import scyjava.config

                scyjava.config.set_verbose(2)
            except ImportError:
                logging.exception("Failed to enable scyjava verbose mode.")
        return

    elif type(logger) is str:
        module_name, logger_attr = logger.rsplit(".", 1)
        try:
            m = importlib.import_module(module_name)
            logger = getattr(m, logger_attr)
        except ImportError:
            logging.exception("Failed to enable debug logging for %s.", logger)
            return

    logger.addHandler(logging.StreamHandler(sys.stderr))
    logger.setLevel(logging.DEBUG)


if __name__ == "__main__":
    checkup()
