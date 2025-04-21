import argparse
import subprocess
import pytest

import imagej
from scyjava import config


def pytest_addoption(parser):
    """
    Set up the command line parser for ImageJ location and headless mode
    :param parser: pytest's parser, passed in automatically
    :return: None
    """
    parser.addoption(
        "--ij",
        action="store",
        default=None,
        help="directory or endpoint (see imagej.init)",
    )
    parser.addoption(
        "--headless",
        type=str2bool,
        action="store",
        default=True,
        help="Start in headless mode",
    )
    parser.addoption(
        "--legacy",
        type=str2bool,
        action="store",
        default=True,
        help="Include the original ImageJ",
    )


@pytest.fixture(scope="session")
def ij(request):
    """
    Create an ImageJ instance to be used by the whole testing environment
    :param request: Pytest variable passed in to fixtures
    """
    # get test configuration
    ij_dir = request.config.getoption("--ij")
    legacy = request.config.getoption("--legacy")
    headless = request.config.getoption("--headless")
    # add the nashorn (JavaScript) endpoint if needed,
    # nashorn was bundled with the JDK from Java 8 to 14
    if capture_java_version() > 14:
        config.endpoints.append("org.openjdk.nashorn:nashorn-core")
    imagej.when_imagej_starts(lambda ij: setattr(ij, "_testing", True))
    # initialize the ImageJ gateway
    mode = "headless" if headless else "interactive"
    ij = imagej.init(ij_dir, mode=mode, add_legacy=legacy)

    yield ij

    ij.dispose()


def capture_java_version() -> int:
    """Capture the installed Java version.

    This function captures the JDK version installed in the current
    venv without starting the JVM by parsing the Java "-version"
    output string.

    :return: The major Java version (8, 11, 21 etc...).
    """
    try:
        # capture the Java version string
        java_ver_str = subprocess.run(
            ["java", "-version"], capture_output=True, text=True
        )
        # extract the Java version from the string
        java_ver = java_ver_str.stderr.split("\n")[0].split(" ")[2]
        java_ver = java_ver.strip('"').split(".")
        major_ver_arr = [int(java_ver[i]) for i in range(2)]
        # find major Java version
        if major_ver_arr[0] == 1:
            return major_ver_arr[1]  # Java 8-10
        else:
            return major_ver_arr[0]  # Java 11+
    except FileNotFoundError:
        raise RuntimeError("No Java installation found.")


def str2bool(v):
    """
    Convert string inputs into bool
    :param v: A string
    :return: Corresponding boolean
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected")
