import argparse
import sys
import unittest
import imagej
import pytest
import scyjava as sj
import numpy as np
import xarray as xr

from jpype import JObject, JException


class TestImageJ(object):
    def test_frangi(self, ij_fixture):
        input_array = np.array([[1000, 1000, 1000, 2000, 3000], [5000, 8000, 13000, 21000, 34000]])
        result = np.zeros(input_array.shape)
        ij_fixture.op().filter().frangiVesselness(ij_fixture.py.to_java(result), ij_fixture.py.to_java(input_array), [1, 1], 4)
        correct_result = np.array([[0, 0, 0, 0.94282, 0.94283], [0, 0, 0, 0.94283, 0.94283]])
        result = np.ndarray.round(result, decimals=5)
        assert (result == correct_result).all()

    def test_gaussian(self, ij_fixture):
        input_array = np.array([[1000, 1000, 1000, 2000, 3000], [5000, 8000, 13000, 21000, 34000]])
        sigmas = [10.0] * 2
        output_array = ij_fixture.op().filter().gauss(ij_fixture.py.to_java(input_array), sigmas)
        result = []
        correct_result = [8440, 8440, 8439, 8444]
        ra = output_array.randomAccess()
        for x in [0, 1]:
            for y in [0, 1]:
                ra.setPosition(x, y)
                result.append(ra.get().get())
        assert result == correct_result

    def test_top_hat(self, ij_fixture):
        ArrayList = sj.jimport('java.util.ArrayList')
        HyperSphereShape = sj.jimport('net.imglib2.algorithm.neighborhood.HyperSphereShape')
        Views = sj.jimport('net.imglib2.view.Views')

        result = []
        correct_result = [0, 0, 0, 1000, 2000, 4000, 7000, 12000, 20000, 33000]

        input_array = np.array([[1000, 1000, 1000, 2000, 3000], [5000, 8000, 13000, 21000, 34000]])
        output_array = np.zeros(input_array.shape)
        java_out = Views.iterable(ij_fixture.py.to_java(output_array))
        java_in = ij_fixture.py.to_java(input_array)
        shapes = ArrayList()
        shapes.add(HyperSphereShape(5))

        ij_fixture.op().morphology().topHat(java_out, java_in, shapes)
        itr = java_out.iterator()
        while itr.hasNext():
            result.append(itr.next().get())

        assert result == correct_result

    def test_image_math(self, ij_fixture):
        Views = sj.jimport('net.imglib2.view.Views')

        input_array = np.array([[1, 1, 2], [3, 5, 8]])
        result = []
        correct_result = [192, 198, 205, 192, 198, 204]
        java_in = Views.iterable(ij_fixture.py.to_java(input_array))
        java_out = ij_fixture.op().image().equation(java_in, "64 * (Math.sin(0.1 * p[0]) + Math.cos(0.1 * p[1])) + 128")

        itr = java_out.iterator()
        while itr.hasNext():
            result.append(itr.next().get())
        assert result == correct_result

    def test_plugins_load_using_pairwise_stitching(self, ij_fixture):
        macro = """
        newImage("Tile1", "8-bit random", 512, 512, 1);
        newImage("Tile2", "8-bit random", 512, 512, 1);
        """
        plugin = 'Pairwise stitching'
        args = {'first_image': 'Tile1', 'second_image': 'Tile2'}

        ij_fixture.script().run('macro.ijm', macro, True).get()
        ij_fixture.py.run_plugin(plugin, args)
        WindowManager = sj.jimport('ij.WindowManager')
        result_name = WindowManager.getCurrentImage().getTitle()

        ij_fixture.script().run('macro.ijm', 'run("Close All");', True).get()

        assert result_name == 'Tile1<->Tile2'


@pytest.fixture(scope='module')
def get_xarr():
    def _get_xarr(option='C'):
        if option == 'C':
            xarr = xr.DataArray(np.random.rand(5, 4, 6, 12, 3), dims=['t', 'z', 'y', 'x', 'c'],
                                coords={'x': list(range(0, 12)), 'y': list(np.arange(0, 12, 2)), 'c': [0, 1, 2],
                                        'z': list(np.arange(10, 50, 10)), 't': list(np.arange(0, 0.05, 0.01))},
                                attrs={'Hello': 'Wrld'})
        elif option == 'F':
            xarr = xr.DataArray(np.ndarray([5, 4, 3, 6, 12], order='F'), dims=['t', 'z', 'c', 'y', 'x'],
                                coords={'x': range(0, 12), 'y': np.arange(0, 12, 2),
                                        'z': np.arange(10, 50, 10), 't': np.arange(0, 0.05, 0.01)},
                                attrs={'Hello': 'Wrld'})
        else:
            xarr = xr.DataArray(np.random.rand(1, 2, 3, 4, 5))

        return xarr
    return _get_xarr


def assert_xarray_equal_to_dataset(ij_fixture, xarr):
    dataset = ij_fixture.py.to_java(xarr)

    axes = [dataset.axis(axnum) for axnum in range(5)]
    labels = [axis.type().getLabel() for axis in axes]

    for label, vals in xarr.coords.items():
        cur_axis = axes[labels.index(label.upper())]
        for loc in range(len(vals)):
            assert vals[loc] == cur_axis.calibratedValue(loc)

    if np.isfortran(xarr.values):
        expected_labels = [dim.upper() for dim in xarr.dims]
    else:
        expected_labels = ['X', 'Y', 'Z', 'T', 'C']

    assert expected_labels == labels
    assert xarr.attrs == ij_fixture.py.from_java(dataset.getProperties())


def assert_inverted_xarr_equal_to_xarr(dataset, ij_fixture, xarr):
    # Reversing back to xarray yields original results
    invert_xarr = ij_fixture.py.from_java(dataset)
    assert (xarr.values == invert_xarr.values).all()
    assert list(xarr.dims) == list(invert_xarr.dims)
    for key in xarr.coords:
        assert (xarr.coords[key] == invert_xarr.coords[key]).all()
    assert xarr.attrs == invert_xarr.attrs


class TestXarrayConversion(object):
    def test_cstyle_array_with_labeled_dims_converts(self, ij_fixture, get_xarr):
        assert_xarray_equal_to_dataset(ij_fixture, get_xarr())

    def test_fstyle_array_with_labeled_dims_converts(self, ij_fixture, get_xarr):
        assert_xarray_equal_to_dataset(ij_fixture, get_xarr('F'))

    def test_dataset_converts_to_xarray(self, ij_fixture, get_xarr):
        xarr = get_xarr()
        dataset = ij_fixture.py.to_java(xarr)
        assert_inverted_xarr_equal_to_xarr(dataset, ij_fixture, xarr)

    def test_rgb_image_maintains_correct_dim_order_on_conversion(self, ij_fixture, get_xarr):
        xarr = get_xarr()
        dataset = ij_fixture.py.to_java(xarr)

        axes = [dataset.axis(axnum) for axnum in range(5)]
        labels = [axis.type().getLabel() for axis in axes]
        assert ['X', 'Y', 'Z', 'T', 'C'] == labels

        # Test that automatic axis swapping works correctly
        raw_values = ij_fixture.py.rai_to_numpy(dataset)
        assert (xarr.values == np.moveaxis(raw_values, 0, -1)).all()

        assert_inverted_xarr_equal_to_xarr(dataset, ij_fixture, xarr)

    def test_no_coords_or_dims_in_xarr(self, ij_fixture, get_xarr):
        xarr = get_xarr('NoDims')
        dataset = ij_fixture.py.from_java(xarr)
        assert_inverted_xarr_equal_to_xarr(dataset, ij_fixture, xarr)


@pytest.fixture(scope="module")
def arr():
    empty_array = np.zeros([512, 512])
    return empty_array


class TestIJ1ToIJ2Synchronization(object):
    def test_get_image_plus_synchronizes_from_ij1_to_ij2(self, ij_fixture, arr):
        if not ij_fixture.legacy.isActive():
            pytest.skip("No IJ1.  Skipping test.")
        if ij_fixture.ui().isHeadless():
            pytest.skip("No GUI.  Skipping test")

        original = arr[0, 0]
        ds = ij_fixture.py.to_java(arr)
        ij_fixture.ui().show(ds)
        macro = """run("Add...", "value=5");"""
        ij_fixture.py.run_macro(macro)
        imp = ij_fixture.py.active_image_plus()

        assert arr[0, 0] == original + 5

    def test_synchronize_from_ij1_to_numpy(self, ij_fixture, arr):
        if not ij_fixture.legacy.isActive():
            pytest.skip("No IJ1.  Skipping test.")
        if ij_fixture.ui().isHeadless():
            pytest.skip("No GUI.  Skipping test")

        original = arr[0, 0]
        ds = ij_fixture.py.to_dataset(arr)
        ij_fixture.ui().show(ds)
        imp = ij_fixture.py.active_image_plus()
        imp.getProcessor().add(5)
        ij_fixture.py.synchronize_ij1_to_ij2(imp)

        assert arr[0, 0] == original + 5

    def test_window_to_numpy_converts_active_image_to_xarray(self, ij_fixture, arr):
        if not ij_fixture.legacy.isActive():
            pytest.skip("No IJ1.  Skipping test.")
        if ij_fixture.ui().isHeadless():
            pytest.skip("No GUI.  Skipping test")

        ds = ij_fixture.py.to_dataset(arr)
        ij_fixture.ui().show(ds)
        new_arr = ij_fixture.py.active_xarray()
        assert (arr == new_arr.values).all

    def test_functions_throw_warning_if_legacy_not_enabled(self, ij_fixture):
        if ij_fixture.legacy.isActive():
            pytest.skip("IJ1 installed.  Skipping test")

        with pytest.raises(AttributeError):
            ij_fixture.py.synchronize_ij1_to_ij2(None)
        with pytest.raises(ImportError):
            ij_fixture.py.active_image_plus()
