import numpy as np
import pytest

# -- Fixtures --


@pytest.fixture(scope="module")
def arr():
    empty_array = np.zeros([512, 512])
    return empty_array


# -- Helpers --


def ensure_gui_available(ij):
    if ij.ui().isHeadless():
        pytest.skip("No GUI. Skipping test.")


def ensure_legacy_disabled(ij):
    if ij.legacy and ij.legacy.isActive():
        pytest.skip("Original ImageJ installed. Skipping test.")


def ensure_legacy_enabled(ij):
    if not ij.legacy or not ij.legacy.isActive():
        pytest.skip("No original ImageJ. Skipping test.")


# -- Tests --


def test_run_plugin(ij_fixture):
    ensure_legacy_enabled(ij_fixture)

    ramp = ij_fixture.IJ.createImage("Tile1", "8-bit ramp", 10, 10, 1)
    ij_fixture.py.run_plugin("Gaussian Blur...", args={"sigma": 3}, imp=ramp)
    values = [ramp.getPixel(x, y)[0] for x in range(10) for y in range(10)]
    # fmt: off
    assert values == [
         30,  30,  30,  30,  30,  30,  30,  30,  30,  30,
         45,  45,  45,  45,  45,  45,  45,  45,  45,  45,
         62,  62,  62,  62,  62,  62,  62,  62,  62,  62,
         82,  82,  82,  82,  82,  82,  82,  82,  82,  82,
        104, 104, 104, 104, 104, 104, 104, 104, 104, 104,  # noqa: E131
        126, 126, 126, 126, 126, 126, 126, 126, 126, 126,  # noqa: E131
        148, 148, 148, 148, 148, 148, 148, 148, 148, 148,  # noqa: E131
        168, 168, 168, 168, 168, 168, 168, 168, 168, 168,  # noqa: E131
        185, 185, 185, 185, 185, 185, 185, 185, 185, 185,  # noqa: E131
        200, 200, 200, 200, 200, 200, 200, 200, 200, 200   # noqa: E131
    ]
    # fmt: on


def test_get_imageplus_synchronizes_from_imagej_to_imagej2(ij_fixture, arr):
    ensure_legacy_enabled(ij_fixture)
    ensure_gui_available(ij_fixture)

    original = arr[0, 0]
    ds = ij_fixture.py.to_java(arr)
    ij_fixture.ui().show(ds)
    macro = """run("Add...", "value=5");"""
    ij_fixture.py.run_macro(macro)

    assert arr[0, 0] == original + 5


def test_synchronize_from_imagej_to_numpy(ij_fixture, arr):
    ensure_legacy_enabled(ij_fixture)
    ensure_gui_available(ij_fixture)

    original = arr[0, 0]
    ds = ij_fixture.py.to_dataset(arr)
    ij_fixture.ui().show(ds)
    imp = ij_fixture.py.active_imageplus()
    imp.getProcessor().add(5)
    ij_fixture.py.sync_image(imp)

    assert arr[0, 0] == original + 5


def test_window_to_numpy_converts_active_image_to_xarray(ij_fixture, arr):
    ensure_legacy_enabled(ij_fixture)
    ensure_gui_available(ij_fixture)

    ds = ij_fixture.py.to_dataset(arr)
    ij_fixture.ui().show(ds)
    new_arr = ij_fixture.py.active_xarray()
    assert (arr == new_arr.values).all


def test_functions_throw_warning_if_legacy_not_enabled(ij_fixture):
    ensure_legacy_disabled(ij_fixture)

    with pytest.raises(AttributeError):
        ij_fixture.py.sync_image(None)
    with pytest.raises(ImportError):
        ij_fixture.py.active_imageplus()
