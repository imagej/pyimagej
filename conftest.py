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
    parser.addoption(
        "--java",
        action="store",
        default=None,
        help="version of Java to cache and use (e.g. 8, 11, 17, 21)",
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
    java_version = request.config.getoption("--java")
    config.set_java_constraints(fetch=True, vendor='zulu', version=java_version)
    # JavaScript is used in the tests. But the Nashorn engine was
    # dropped from the JDK at v15, so we readd it here if needed.
    if int(java_version) >= 15:
        config.endpoints.append("org.openjdk.nashorn:nashorn-core")
    imagej.when_imagej_starts(lambda ij: setattr(ij, "_testing", True))
    # initialize the ImageJ gateway
    mode = "headless" if headless else "interactive"
    ij = imagej.init(ij_dir, mode=mode, add_legacy=legacy)

    yield ij

    ij.dispose()


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
