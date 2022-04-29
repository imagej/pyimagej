import argparse
import random
import sys

import pytest
import imagej.dims as dims
import scyjava as sj
import numpy as np
import xarray as xr

from jpype import JObject, JException, JArray, JInt, JLong


class TestImageJ(object):
    def test_frangi(self, ij_fixture):
        input_array = np.array(
            [[1000, 1000, 1000, 2000, 3000], [5000, 8000, 13000, 21000, 34000]]
        )
        result = np.zeros(input_array.shape)
        ij_fixture.op().filter().frangiVesselness(
            ij_fixture.py.to_java(result), ij_fixture.py.to_java(input_array), [1, 1], 4
        )
        correct_result = np.array(
            [[0, 0, 0, 0.94282, 0.94283], [0, 0, 0, 0.94283, 0.94283]]
        )
        result = np.ndarray.round(result, decimals=5)
        assert (result == correct_result).all()

    def test_gaussian(self, ij_fixture):
        input_array = np.array(
            [[1000, 1000, 1000, 2000, 3000], [5000, 8000, 13000, 21000, 34000]]
        )
        sigmas = [10.0] * 2
        output_array = (
            ij_fixture.op().filter().gauss(ij_fixture.py.to_java(input_array), sigmas)
        )
        result = []
        correct_result = [8435, 8435, 8435, 8435]
        ra = output_array.randomAccess()
        for x in [0, 1]:
            for y in [0, 1]:
                ra.setPosition(x, y)
                result.append(ra.get().get())
        assert result == correct_result

    def test_top_hat(self, ij_fixture):
        ArrayList = sj.jimport("java.util.ArrayList")
        HyperSphereShape = sj.jimport(
            "net.imglib2.algorithm.neighborhood.HyperSphereShape"
        )
        Views = sj.jimport("net.imglib2.view.Views")

        result = []
        correct_result = [0, 0, 0, 1000, 2000, 4000, 7000, 12000, 20000, 33000]

        input_array = np.array(
            [[1000, 1000, 1000, 2000, 3000], [5000, 8000, 13000, 21000, 34000]]
        )
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
        Views = sj.jimport("net.imglib2.view.Views")

        input_array = np.array([[1, 1, 2], [3, 5, 8]])
        result = []
        correct_result = [192, 198, 205, 192, 198, 204]
        java_in = Views.iterable(ij_fixture.py.to_java(input_array))
        java_out = (
            ij_fixture.op()
            .image()
            .equation(
                java_in, "64 * (Math.sin(0.1 * p[0]) + Math.cos(0.1 * p[1])) + 128"
            )
        )

        itr = java_out.iterator()
        while itr.hasNext():
            result.append(itr.next().get())
        assert result == correct_result

    def test_run_plugin(self, ij_fixture):
        if not ij_fixture.legacy:
            pytest.skip("No original ImageJ. Skipping test.")

        noise = ij_fixture.IJ.createImage("Tile1", "8-bit random", 10, 10, 1)
        before = [noise.getPixel(x, y)[0] for x in range(10) for y in range(10)]
        ij_fixture.py.run_plugin("Gaussian Blur...", args={"sigma": 3}, imp=noise)
        after = [noise.getPixel(x, y)[0] for x in range(10) for y in range(10)]
        print(f"before = {before}")
        print(f"after = {after}")

    def test_plugins_load_using_pairwise_stitching(self, ij_fixture):
        try:
            sj.jimport("plugin.Stitching_Pairwise")
        except TypeError:
            pytest.skip("No Pairwise Stitching plugin available. Skipping test.")

        if not ij_fixture.legacy:
            pytest.skip("No original ImageJ. Skipping test.")
        if ij_fixture.ui().isHeadless():
            pytest.skip("No GUI. Skipping test.")

        tile1 = ij_fixture.IJ.createImage("Tile1", "8-bit random", 512, 512, 1)
        tile2 = ij_fixture.IJ.createImage("Tile2", "8-bit random", 512, 512, 1)
        args = {"first_image": tile1.getTitle(), "second_image": tile2.getTitle()}
        ij_fixture.py.run_plugin("Pairwise stitching", args)
        result_name = ij_fixture.WindowManager.getCurrentImage().getTitle()

        ij_fixture.IJ.run("Close All", "")

        assert result_name == "Tile1<->Tile2"


@pytest.fixture(scope="module")
def get_xarr():
    def _get_xarr(option="C"):
        if option == "C":
            xarr = xr.DataArray(
                np.random.rand(5, 4, 6, 12, 3),
                dims=["t", "pln", "row", "col", "ch"],
                coords={
                    "col": list(range(12)),
                    "row": list(range(0, 12, 2)),
                    "ch": [0, 1, 2],
                    "pln": list(range(10, 50, 10)),
                    "t": list(np.arange(0, 0.05, 0.01)),
                },
                attrs={"Hello": "World"},
            )
        elif option == "F":
            xarr = xr.DataArray(
                np.ndarray([5, 4, 3, 6, 12], order="F"),
                dims=["t", "pln", "ch", "row", "col"],
                coords={
                    "col": list(range(12)),
                    "row": list(range(0, 12, 2)),
                    "pln": list(range(10, 50, 10)),
                    "t": list(np.arange(0, 0.05, 0.01)),
                },
                attrs={"Hello": "World"},
            )
        else:
            xarr = xr.DataArray(np.random.rand(1, 2, 3, 4, 5))

        return xarr

    return _get_xarr


@pytest.fixture(scope="module")
def get_imgplus():
    def _get_imgplus(ij_fixture):
        """Get a 7D ImgPlus."""
        # get java resources
        Random = sj.jimport("java.util.Random")
        Axes = sj.jimport("net.imagej.axis.Axes")
        UnsignedByteType = sj.jimport(
            "net.imglib2.type.numeric.integer.UnsignedByteType"
        )
        DatasetService = ij_fixture.get("net.imagej.DatasetService")

        # test image parameters
        foo = Axes.get("foo")
        bar = Axes.get("bar")
        shape = [13, 17, 5, 2, 3, 7, 11]
        axes = [Axes.X, Axes.Y, foo, bar, Axes.CHANNEL, Axes.TIME, Axes.Z]

        # create image
        dataset = DatasetService.create(UnsignedByteType(), shape, "fabulous7D", axes)
        imgplus = dataset.typedImg(UnsignedByteType())

        # fill the image with noise
        rng = Random(123456789)
        t = UnsignedByteType()

        for t in imgplus:
            t.set(rng.nextInt(256))

        return imgplus

    return _get_imgplus


def assert_xarray_equal_to_dataset(ij_fixture, xarr):
    dataset = ij_fixture.py.to_java(xarr)
    axes = [dataset.axis(axnum) for axnum in range(5)]
    labels = [axis.type().getLabel() for axis in axes]

    for label, vals in xarr.coords.items():
        cur_axis = axes[labels.index(dims._convert_dim(label, direction="java"))]
        for loc in range(len(vals)):
            assert vals[loc] == cur_axis.calibratedValue(loc)

    if np.isfortran(xarr.values):
        expected_labels = [
            dims._convert_dim(dim, direction="java") for dim in xarr.dims
        ]
    else:
        expected_labels = ["X", "Y", "Z", "Time", "Channel"]

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


def assert_permuted_rai_equal_to_source_rai(imgplus):
    # get java resources
    Axes = sj.jimport("net.imagej.axis.Axes")

    # define extra axes
    foo = Axes.get("foo")
    bar = Axes.get("bar")

    # permute the rai to python order
    axis_types = [axis.type() for axis in imgplus.dim_axes]
    permute_order = dims.prioritize_rai_axes_order(
        axis_types, dims._python_rai_ref_order()
    )
    permuted_rai = dims.reorganize(imgplus, permute_order)

    # extract values for assertion
    oc = imgplus.dimensionIndex(Axes.CHANNEL)
    ox = imgplus.dimensionIndex(Axes.X)
    oy = imgplus.dimensionIndex(Axes.Y)
    oz = imgplus.dimensionIndex(Axes.Z)
    ot = imgplus.dimensionIndex(Axes.TIME)
    of = imgplus.dimensionIndex(foo)
    ob = imgplus.dimensionIndex(bar)

    nc = permuted_rai.dimensionIndex(Axes.CHANNEL)
    nx = permuted_rai.dimensionIndex(Axes.X)
    ny = permuted_rai.dimensionIndex(Axes.Y)
    nz = permuted_rai.dimensionIndex(Axes.Z)
    nt = permuted_rai.dimensionIndex(Axes.TIME)
    nf = permuted_rai.dimensionIndex(foo)
    nb = permuted_rai.dimensionIndex(bar)

    oc_len = imgplus.dimension(oc)
    ox_len = imgplus.dimension(ox)
    oy_len = imgplus.dimension(oy)
    oz_len = imgplus.dimension(oz)
    ot_len = imgplus.dimension(ot)
    of_len = imgplus.dimension(of)
    ob_len = imgplus.dimension(ob)

    nc_len = permuted_rai.dimension(nc)
    nx_len = permuted_rai.dimension(nx)
    ny_len = permuted_rai.dimension(ny)
    nz_len = permuted_rai.dimension(nz)
    nt_len = permuted_rai.dimension(nt)
    nf_len = permuted_rai.dimension(nf)
    nb_len = permuted_rai.dimension(nb)

    # assert the number of pixels of each dimension
    assert oc_len == nc_len
    assert ox_len == nx_len
    assert oy_len == ny_len
    assert oz_len == nz_len
    assert ot_len == nt_len
    assert of_len == nf_len
    assert ob_len == nb_len

    # get RandomAccess
    imgplus_access = imgplus.randomAccess()
    permuted_rai_access = permuted_rai.randomAccess()

    # assert pixels between source and permuted rai
    for c in range(oc_len):
        imgplus_access.setPosition(c, oc)
        permuted_rai_access.setPosition(c, nc)
        for x in range(ox_len):
            imgplus_access.setPosition(x, ox)
            permuted_rai_access.setPosition(x, nx)
            for y in range(oy_len):
                imgplus_access.setPosition(y, oy)
                permuted_rai_access.setPosition(y, ny)
                for z in range(oz_len):
                    imgplus_access.setPosition(z, oz)
                    permuted_rai_access.setPosition(z, nz)
                    for t in range(ot_len):
                        imgplus_access.setPosition(t, ot)
                        permuted_rai_access.setPosition(t, nt)
                        for f in range(of_len):
                            imgplus_access.setPosition(f, of)
                            permuted_rai_access.setPosition(f, nf)
                            for b in range(ob_len):
                                imgplus_access.setPosition(b, ob)
                                permuted_rai_access.setPosition(b, nb)
                                sample_name = f"C: {c}, X: {x}, Y: {y}, Z: {z}, T: {t}, F: {f}, B: {b}"
                                assert (
                                    imgplus_access.get() == permuted_rai_access.get()
                                ), sample_name


class TestXarrayConversion(object):
    def test_cstyle_array_with_labeled_dims_converts(self, ij_fixture, get_xarr):
        assert_xarray_equal_to_dataset(ij_fixture, get_xarr())

    def test_fstyle_array_with_labeled_dims_converts(self, ij_fixture, get_xarr):
        assert_xarray_equal_to_dataset(ij_fixture, get_xarr("F"))

    def test_7d_rai_to_python_permute(self, ij_fixture, get_imgplus):
        assert_permuted_rai_equal_to_source_rai(get_imgplus(ij_fixture))

    def test_dataset_converts_to_xarray(self, ij_fixture, get_xarr):
        xarr = get_xarr()
        dataset = ij_fixture.py.to_java(xarr)
        assert_inverted_xarr_equal_to_xarr(dataset, ij_fixture, xarr)

    def test_rgb_image_maintains_correct_dim_order_on_conversion(
        self, ij_fixture, get_xarr
    ):
        xarr = get_xarr()
        dataset = ij_fixture.py.to_java(xarr)

        axes = [dataset.axis(axnum) for axnum in range(5)]
        labels = [axis.type().getLabel() for axis in axes]
        assert ["X", "Y", "Z", "Time", "Channel"] == labels

        # Test that automatic axis swapping works correctly
        numpy_image = ij_fixture.py.initialize_numpy_image(dataset)
        raw_values = ij_fixture.py.rai_to_numpy(dataset, numpy_image)
        assert (xarr.values == np.moveaxis(raw_values, 0, -1)).all()

        assert_inverted_xarr_equal_to_xarr(dataset, ij_fixture, xarr)

    def test_no_coords_or_dims_in_xarr(self, ij_fixture, get_xarr):
        xarr = get_xarr("NoDims")
        dataset = ij_fixture.py.from_java(xarr)
        assert_inverted_xarr_equal_to_xarr(dataset, ij_fixture, xarr)


@pytest.fixture(scope="module")
def arr():
    empty_array = np.zeros([512, 512])
    return empty_array


class TestSynchronization(object):
    def test_get_imageplus_synchronizes_from_imagej_to_imagej2(self, ij_fixture, arr):
        if not ij_fixture.legacy:
            pytest.skip("No original ImageJ. Skipping test.")
        if ij_fixture.ui().isHeadless():
            pytest.skip("No GUI. Skipping test.")

        original = arr[0, 0]
        ds = ij_fixture.py.to_java(arr)
        ij_fixture.ui().show(ds)
        macro = """run("Add...", "value=5");"""
        ij_fixture.py.run_macro(macro)
        imp = ij_fixture.py.active_imageplus()

        assert arr[0, 0] == original + 5

    def test_synchronize_from_imagej_to_numpy(self, ij_fixture, arr):
        if not ij_fixture.legacy:
            pytest.skip("No original ImageJ. Skipping test.")
        if ij_fixture.ui().isHeadless():
            pytest.skip("No GUI. Skipping test.")

        original = arr[0, 0]
        ds = ij_fixture.py.to_dataset(arr)
        ij_fixture.ui().show(ds)
        imp = ij_fixture.py.active_imageplus()
        imp.getProcessor().add(5)
        ij_fixture.py.sync_image(imp)

        assert arr[0, 0] == original + 5

    def test_window_to_numpy_converts_active_image_to_xarray(self, ij_fixture, arr):
        if not ij_fixture.legacy:
            pytest.skip("No original ImageJ. Skipping test.")
        if ij_fixture.ui().isHeadless():
            pytest.skip("No GUI. Skipping test.")

        ds = ij_fixture.py.to_dataset(arr)
        ij_fixture.ui().show(ds)
        new_arr = ij_fixture.py.active_xarray()
        assert (arr == new_arr.values).all

    def test_functions_throw_warning_if_legacy_not_enabled(self, ij_fixture):
        if ij_fixture.legacy and ij_fixture.legacy.isActive():
            pytest.skip("Original ImageJ installed. Skipping test.")

        with pytest.raises(AttributeError):
            ij_fixture.py.sync_image(None)
        with pytest.raises(ImportError):
            ij_fixture.py.active_imageplus()


@pytest.fixture(scope="module")
def get_nparr():
    def _get_nparr():
        return np.random.rand(1, 2, 3, 4, 5)

    return _get_nparr


@pytest.fixture(scope="module")
def get_img(ij_fixture):
    def _get_img():
        # Create img
        CreateNamespace = sj.jimport("net.imagej.ops.create.CreateNamespace")
        dims = JArray(JLong)([1, 2, 3, 4, 5])
        ns = ij_fixture.op().namespace(CreateNamespace)
        img = ns.img(dims)

        # Populate img with random data
        cursor = img.cursor()
        while cursor.hasNext():
            val = random.random()
            cursor.next().set(val)

        return img

    return _get_img


def assert_ndarray_equal_to_img(img, nparr):
    cursor = img.cursor()
    arr = JArray(JInt)(5)
    while cursor.hasNext():
        y = cursor.next().get()
        cursor.localize(arr)
        # TODO: Imglib has inverted dimensions - extract this behavior into a helper function
        x = nparr[tuple(arr[::-1])]
        assert x == y


def convert_ndarray_and_assert_equality(ij_fixture, nparr):
    img = ij_fixture.py.to_java(nparr)
    assert_ndarray_equal_to_img(img, nparr)


def convert_img_and_assert_equality(ij_fixture, img):
    nparr = ij_fixture.py.from_java(img)
    assert_ndarray_equal_to_img(img, nparr)


class TestNumpyConversion(object):
    def test_ndarray_converts_to_img(self, ij_fixture, get_nparr):
        convert_ndarray_and_assert_equality(ij_fixture, get_nparr())

    def test_img_converts_to_ndarray(self, ij_fixture, get_img):
        convert_img_and_assert_equality(ij_fixture, get_img())


class TestRAIArraylike(object):
    @pytest.fixture
    def img(self):
        # Create img
        ArrayImgs = sj.jimport("net.imglib2.img.array.ArrayImgs")
        img = ArrayImgs.bytes(2, 3, 4)
        # Insert a different value into each index
        tmp_val = 1
        cursor = img.cursor()
        while cursor.hasNext():
            cursor.next().set(tmp_val)
            tmp_val += 1
        # Return the new img
        return img

    def test_slice_index(self, ij_fixture, img):
        assert img[0, 0, 0].get() == 1

    def test_slice_index_negative(self, ij_fixture, img):
        assert img[-1, -1, -1].get() == 24

    def test_slice_2d(self, ij_fixture, img):
        Views = sj.jimport("net.imglib2.view.Views")
        expected = Views.hyperSlice(img, 0, 0)
        actual = img[0, :, :]
        for i in range(3):
            for j in range(4):
                assert expected[i, j] == actual[i, j]

    def test_slice_2d_negative(self, ij_fixture, img):
        Views = sj.jimport("net.imglib2.view.Views")
        expected = Views.hyperSlice(img, 0, 1)
        actual = img[-1, :, :]
        for i in range(3):
            for j in range(4):
                assert expected[i, j] == actual[i, j]

    def test_slice_1d(self, ij_fixture, img):
        Views = sj.jimport("net.imglib2.view.Views")
        expected = Views.hyperSlice(Views.hyperSlice(img, 0, 0), 0, 0)
        actual = img[0, 0, :]
        for i in range(4):
            assert expected[i] == actual[i]

    def test_slice_1d_negative(self, ij_fixture, img):
        Views = sj.jimport("net.imglib2.view.Views")
        expected = Views.hyperSlice(Views.hyperSlice(img, 0, 1), 0, 1)
        actual = img[-1, -2, :]
        for i in range(4):
            assert expected[i] == actual[i]

    def test_slice_int(self, ij_fixture, img):
        Views = sj.jimport("net.imglib2.view.Views")
        expected = Views.hyperSlice(img, 0, 0)
        actual = img[0]
        for i in range(3):
            for j in range(4):
                assert expected[i, j] == actual[i, j]

    def test_slice_not_enough_dims(self, ij_fixture, img):
        Views = sj.jimport("net.imglib2.view.Views")
        expected = Views.hyperSlice(Views.hyperSlice(img, 0, 0), 0, 0)
        actual = img[0, 0]
        for i in range(4):
            assert expected[i] == actual[i]

    def test_step(self, ij_fixture, img):
        # Create a stepped img via Views
        Views = sj.jimport("net.imglib2.view.Views")
        steps = JArray(JLong)([1, 1, 2])
        expected = Views.subsample(img, steps)
        # Create a stepped img via slicing notation
        actual = img[:, :, ::2]
        for i in range(2):
            for j in range(3):
                for k in range(2):
                    assert expected[i, j, k] == actual[i, j, k]

    def test_step_not_enough_dims(self, ij_fixture, img):
        # Create a stepped img via Views
        Views = sj.jimport("net.imglib2.view.Views")
        steps = JArray(JLong)([2, 1, 1])
        expected = Views.subsample(img, steps)
        expected = Views.dropSingletonDimensions(expected)
        # Create a stepped img via slicing notation
        actual = img[::2]
        for i in range(3):
            for j in range(4):
                assert expected[i, j] == actual[i, j]

    def test_slice_and_step(self, ij_fixture, img):
        # Create a stepped img via Views
        Views = sj.jimport("net.imglib2.view.Views")
        intervaled = Views.hyperSlice(img, 0, 0)
        steps = JArray(JLong)([1, 2])
        expected = Views.subsample(intervaled, steps)
        # Create a stepped img via slicing notation
        actual = img[:1, :, ::2]
        for i in range(3):
            for j in range(2):
                assert expected[i, j] == actual[i, j]

    def test_shape(self, ij_fixture, img):
        assert hasattr(img, "shape")
        assert img.shape == (2, 3, 4)

    def test_dtype(self, ij_fixture, img):
        assert hasattr(img, "dtype")
        ByteType = sj.jimport("net.imglib2.type.numeric.integer.ByteType")
        assert img.dtype == ByteType

    def test_dtype(self, ij_fixture, img):
        assert hasattr(img, "ndim")
        assert img.ndim == 3

    def test_transpose1d(self, ij_fixture, img):
        img = img[0, 0]
        transpose = img.T
        for i in range(2):
            assert transpose[i] == img[i]

    def test_transpose2d(self, ij_fixture, img):
        img = img[0]
        transpose = img.T
        for i in range(3):
            for j in range(2):
                assert transpose[i, j] == img[j, i]

    def test_transpose3d(self, ij_fixture, img):
        transpose = img.T
        for i in range(4):
            for j in range(3):
                for k in range(2):
                    assert transpose[i, j, k] == img[k, j, i]
