import random

import numpy as np
import pytest
import scyjava as sj
import xarray as xr

# TODO: Change to scyjava.new_jarray once we have that function.
from jpype import JArray, JInt, JLong

import imagej.dims as dims

# -- Fixtures --


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
        shape = [7, 8, 4, 2, 3, 5, 6]
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


@pytest.fixture(scope="module")
def get_nparr():
    def _get_nparr():
        return np.random.rand(1, 2, 3, 4, 5)

    return _get_nparr


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


# -- Helpers --


def assert_inverted_xarr_equal_to_xarr(dataset, ij_fixture, xarr):
    # Reversing back to xarray yields original results
    invert_xarr = ij_fixture.py.from_java(dataset)
    assert (xarr.values == invert_xarr.values).all()
    assert list(xarr.dims) == list(invert_xarr.dims)
    for key in xarr.coords:
        assert (xarr.coords[key] == invert_xarr.coords[key]).all()
    assert xarr.attrs == invert_xarr.attrs


def assert_ndarray_equal_to_ndarray(narr_1, narr_2):
    assert (narr_1 == narr_2).all()


def assert_ndarray_equal_to_img(img, nparr):
    cursor = img.cursor()
    arr = JArray(JInt)(5)
    while cursor.hasNext():
        y = cursor.next().get()
        cursor.localize(arr)
        # TODO: Imglib has inverted dimensions - extract this behavior into a
        # helper function
        x = nparr[tuple(arr[::-1])]
        assert x == y


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
                                sample_name = (
                                    f"C: {c}, X: {x}, Y: {y}, Z: {z}, "
                                    f"T: {t}, F: {f}, B: {b}"
                                )
                                assert (
                                    imgplus_access.get() == permuted_rai_access.get()
                                ), sample_name


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


def convert_img_and_assert_equality(ij_fixture, img):
    nparr = ij_fixture.py.from_java(img)
    assert_ndarray_equal_to_img(img, nparr)


def convert_ndarray_and_assert_equality(ij_fixture, nparr):
    img = ij_fixture.py.to_java(nparr)
    assert_ndarray_equal_to_img(img, nparr)


# -- Tests --


def test_ndarray_converts_to_img(ij_fixture, get_nparr):
    convert_ndarray_and_assert_equality(ij_fixture, get_nparr())


def test_img_converts_to_ndarray(ij_fixture, get_img):
    convert_img_and_assert_equality(ij_fixture, get_img())


def test_cstyle_array_with_labeled_dims_converts(ij_fixture, get_xarr):
    assert_xarray_equal_to_dataset(ij_fixture, get_xarr())


def test_fstyle_array_with_labeled_dims_converts(ij_fixture, get_xarr):
    assert_xarray_equal_to_dataset(ij_fixture, get_xarr("F"))


def test_7d_rai_to_python_permute(ij_fixture, get_imgplus):
    assert_permuted_rai_equal_to_source_rai(get_imgplus(ij_fixture))


def test_dataset_converts_to_xarray(ij_fixture, get_xarr):
    xarr = get_xarr()
    dataset = ij_fixture.py.to_java(xarr)
    assert_inverted_xarr_equal_to_xarr(dataset, ij_fixture, xarr)


def test_rgb_image_maintains_correct_dim_order_on_conversion(ij_fixture, get_xarr):
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


def test_no_coords_or_dims_in_xarr(ij_fixture, get_xarr):
    xarr = get_xarr("NoDims")
    dataset = ij_fixture.py.from_java(xarr)
    assert_inverted_xarr_equal_to_xarr(dataset, ij_fixture, xarr)


def test_direct_to_xarray_conversions(ij_fixture, get_imgplus, get_nparr, get_xarr):
    # fetch test images
    xarr = get_xarr()
    narr = get_nparr()
    imgplus = get_imgplus(ij_fixture)

    # to_xarray conversions
    xarr_same = ij_fixture.py.to_xarray(xarr)
    # xarr_rename = ij_fixture.py.to_xarray(
    #     xarr, dim_order=["Time", "Z", "Y", "X", "Channel"]
    # )
    # xarr_from_narr = ij_fixture.py.to_xarray(
    #     narr, dim_order=["t", "pln", "row", "col", "ch"]
    # )
    xarr_no_dims = ij_fixture.py.to_xarray(narr)
    xarr_from_imgplus = ij_fixture.py.to_xarray(imgplus)

    # check for expected dims
    assert xarr_same.dims == ("t", "pln", "row", "col", "ch")
    # assert xarr_rename.dims == ("Time", "Z", "Y", "X", "Channel")
    # assert xarr_from_narr.dims == ("t", "pln", "row", "col", "ch")
    assert xarr_no_dims.dims == ("dim_0", "dim_1", "dim_2", "dim_3", "dim_4")
    assert xarr_from_imgplus.dims == ("bar", "foo", "t", "pln", "row", "col", "ch")

    # check for same data
    assert_ndarray_equal_to_ndarray(xarr_same.data, xarr.data)
    # assert_ndarray_equal_to_ndarray(xarr_rename.data, xarr.data)
    # assert_ndarray_equal_to_ndarray(xarr_from_narr.data, narr)
    assert_ndarray_equal_to_ndarray(xarr_no_dims.data, narr)
    assert_ndarray_equal_to_ndarray(
        xarr_from_imgplus.data, ij_fixture.py.from_java(imgplus).data
    )
