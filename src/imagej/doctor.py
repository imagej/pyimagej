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

import importlib, logging, os, shutil, subprocess, sys
from pathlib import Path


def _execute(command):
    try:
        return (
            subprocess.check_output(command, stderr=subprocess.STDOUT).decode().rstrip()
        )
    except Exception as e:
        return str(e)


def checkup(output=print):
    """Check your environment for health problems that could prevent PyImageJ from functioning."""
    output("")
    advice = []

    output("Checking Python:")

    output(f"--> Python executable = {sys.executable}")
    output("")

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
            output(f"--> Python executable matches Conda environment.")
        else:
            output(f"--> Python executable is NOT from that Conda environment!")
            indent = "\n    * "
            advice.append(
                "Are you sure you're using the correct Python executable? I expected one of these:"
                + indent
                + indent.join(map(str, expected_exes))
            )
    else:
        output("--> It looks like you are NOT running inside a Conda environment.")
        advice.append("Did you intend to activate a Conda environment?")
    output("")

    output("Checking Python dependencies:")

    dependencies = {
        "jgo": "jgo",
        "scyjava": "scyjava",
        "imglyb": "imglyb",
        "pyimagej": "imagej",
    }
    for package_name, module_name in dependencies.items():
        try:
            m = importlib.import_module(module_name)
            output(f"--> {package_name}: {m.__file__}")
        except ImportError:
            output(f"--> {package_name}: MISSING")
            advice.append(f"Are you sure the {package_name} package is installed?")

    output("")

    output("Checking Maven:")

    mvn_executable = shutil.which("mvn")
    output(f"--> Maven executable = {mvn_executable or 'NOT FOUND!'}")
    if mvn_executable:
        output(f"$ mvn -v\n{_execute([mvn_executable, '-v'])}")
        output("")
    else:
        advice.append("Install maven using conda or your system package manager")

    output("Checking Java:")

    if "JAVA_HOME" in os.environ:
        output(f"--> JAVA_HOME = {os.environ['JAVA_HOME']}")
    else:
        output("--> JAVA_HOME is NOT set!")
        advice.append(
            "Activate a conda environment with openjdk installed, or set JAVA_HOME manually."
        )
    java_executable = shutil.which("java")
    output(f"--> Java executable = {java_executable or 'NOT FOUND!'}")
    if not java_executable:
        advice.append("Install openjdk using conda or your system package manager")

    output(f"$ java -version\n{_execute(['java', '-version'])}")
    output("")

    # TODO: More checks still needed!
    # - Does java executable match JAVA_HOME?
    # - Firewall configuration?
    # - Can mvn retrieve artifacts? (try mvn dependency:copy with suitable timeout?)
    # - Is Maven Central accessible? Is maven.scijava.org accessible?
    # - Where is your Maven repository cache?
    # - Where is jgo caching your environments?
    # - Are you using mamba? (quality of life advice)

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
        debug_to_stderr("jgo.jgo._logger")
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

    elif type(logger) == str:
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
