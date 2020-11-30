import imagej
import pytest
import argparse


def pytest_addoption(parser):
    """
    Set up the command line parser for IJ location and headless mode
    :param parser: pytest's parser, passed in automatically
    :return: None
    """
    parser.addoption(
        "--ij", action="store", default="sc.fiji:fiji", help="directory to IJ"
    )
    parser.addoption(
        "--headless", type=str2bool, action="store", default=True, help="Start in headless mode"
    )


@pytest.fixture(scope='session')
def ij_fixture(request):
    """
    Create an ImageJ instance to be used by the whole testing environment
    :param request: Pytest variable passed in to fixtures
    """
    ij_dir = request.config.getoption('--ij')
    headless = request.config.getoption('--headless')

    ij_wrapper = imagej.init(ij_dir, headless=headless)

    yield ij_wrapper

    ij_wrapper.dispose()


def str2bool(v):
    """
    Convert string inputs into bool
    :param v: A string
    :return: Corresponding boolean
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')
