import random

import numpy as np
import pytest
import scyjava as sj

# -- Fixtures --


@pytest.fixture(scope="module")
def arr():
    empty_array = np.zeros([512, 512])
    return empty_array


@pytest.fixture(scope="module")
def results_table(ij):
    if ij.legacy and ij.legacy.isActive():
        ResultsTable = sj.jimport("ij.measure.ResultsTable")
        rt = ResultsTable.getResultsTable()

        # add column headers
        for i in range(5):
            rt.setHeading(i, f"Column {i}")

        # add data rows
        for i in range(3):
            rt.incrementCounter()
            for j in range(5):
                rt.addValue(f"Column {j}", random.randint(1, 100))
    else:
        pytest.skip("No original ImageJ. Skipping fixture.")

    return rt


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


def test_convert_imageplus_to_python(ij):
    ensure_legacy_enabled(ij)

    w = 30
    h = 20
    imp = ij.IJ.createImage("Ramp", "16-bit ramp", w, h, 2, 3, 5)
    xarr = ij.py.from_java(imp)
    assert xarr.dims == ("t", "pln", "row", "col", "ch")
    assert xarr.shape == (5, 3, h, w, 2)

    index = 0
    for c in range(imp.getNChannels()):
        for z in range(imp.getNSlices()):
            for t in range(imp.getNFrames()):
                index += 1
                ip = imp.getStack().getProcessor(index)
                # NB: The commented out loop is super slow because Python + JNI.
                # Instead, we grab the whole plane as a Java array and massage it.
                # for y in range(imp.getHeight()):
                #     for x in range(imp.getWidth()):
                #         assert plane[y,x] == xarr[t,z,y,x,c]
                plane = np.frombuffer(
                    bytearray(ip.getPixels()), count=w * h, dtype=np.uint16
                ).reshape(h, w)
                assert all((plane == xarr[t, z, :, :, c]).data.flatten())


def test_run_plugin(ij):
    ensure_legacy_enabled(ij)

    ramp = ij.IJ.createImage("Tile1", "8-bit ramp", 10, 10, 1)
    ij.py.run_plugin("Gaussian Blur...", args={"sigma": 3}, imp=ramp)
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


def test_get_imageplus_synchronizes_from_imagej_to_imagej2(ij, arr):
    ensure_legacy_enabled(ij)
    ensure_gui_available(ij)

    original = arr[0, 0]
    ds = ij.py.to_java(arr)
    ij.ui().show(ds)
    macro = """run("Add...", "value=5");"""
    ij.py.run_macro(macro)

    assert arr[0, 0] == original + 5


def test_synchronize_from_imagej_to_numpy(ij, arr):
    ensure_legacy_enabled(ij)
    ensure_gui_available(ij)

    original = arr[0, 0]
    ds = ij.py.to_dataset(arr)
    ij.ui().show(ds)
    imp = ij.py.active_imageplus()
    imp.getProcessor().add(5)
    ij.py.sync_image(imp)

    assert arr[0, 0] == original + 5


def test_window_to_numpy_converts_active_image_to_xarray(ij, arr):
    ensure_legacy_enabled(ij)
    ensure_gui_available(ij)

    ds = ij.py.to_dataset(arr)
    ij.ui().show(ds)
    new_arr = ij.py.active_xarray()
    assert (arr == new_arr.values).all


def test_functions_throw_warning_if_legacy_not_enabled(ij):
    ensure_legacy_disabled(ij)

    with pytest.raises(ImportError):
        ij.py.active_imageplus()


def test_results_table_to_pandas_dataframe(ij, results_table):
    ensure_legacy_enabled(ij)

    df = ij.py.from_java(results_table)
    for col in range(5):
        rt_col = list(results_table.getColumn(col))
        df_col = df[f"Column {col}"].tolist()
        assert rt_col == df_col
