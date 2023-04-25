import random

import numpy as np
import pytest
import scyjava as sj
import xarray as xr

import imagej.dims as dims
import imagej.images as images

# -- Image helpers --


def get_img(ij_fixture):
    # Create img
    dims = sj.jarray("j", [5])
    for i in range(len(dims)):
        dims[i] = i + 1
    img = ij_fixture.op().run("create.img", dims)

    # Populate img with random data
    cursor = img.cursor()
    while cursor.hasNext():
        val = random.random()
        cursor.next().set(val)

    return img


def get_imgplus(ij_fixture):
    """Get a 7D ImgPlus."""
    # get java resources
    Random = sj.jimport("java.util.Random")
    Axes = sj.jimport("net.imagej.axis.Axes")
    UnsignedByteType = sj.jimport("net.imglib2.type.numeric.integer.UnsignedByteType")
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


def get_nparr():
    return np.random.rand(1, 2, 3, 4, 5)


def get_xarr(option="C"):
    name: str = "test_data_array"
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
            name=name,
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
            name=name,
        )
    else:
        xarr = xr.DataArray(np.random.rand(1, 2, 3, 4, 5), name=name)

    return xarr


# -- Helpers --


def assert_xarray_coords_equal_to_rai_coords(xarr, rai):
    rai_axes = list(rai.dim_axes)
    rai_dims = list(rai.dims)
    axes_coords = dims._get_axes_coords(rai_axes, rai_dims, rai.shape)
    for dim in xarr.dims:
        xarr_dim_coords = xarr.coords[dim].to_numpy()
        rai_dim_coords = axes_coords[dims._to_ijdim(dim)]
        for i in range(len(xarr_dim_coords)):
            assert xarr_dim_coords[i] == rai_dim_coords[i]


def assert_inverted_xarr_equal_to_xarr(dataset, ij_fixture, xarr):
    # Reversing back to xarray yields original results
    invert_xarr = ij_fixture.py.from_java(dataset)
    assert (xarr.values == invert_xarr.values).all()
    assert list(xarr.dims) == list(invert_xarr.dims)
    for key in xarr.coords:
        assert (xarr.coords[key] == invert_xarr.coords[key]).all()
    if "Hello" in xarr.attrs.keys():
        assert xarr.attrs["Hello"] == invert_xarr.attrs["Hello"]
    else:
        assert xarr.attrs == invert_xarr.attrs
    assert xarr.name == invert_xarr.name


def assert_ndarray_equal_to_ndarray(narr_1, narr_2):
    assert (narr_1 == narr_2).all()


def assert_ndarray_equal_to_img(img, nparr):
    cursor = img.cursor()
    arr = sj.jarray("i", [nparr.ndim])
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


def assert_xarray_equal_to_dataset(ij_fixture, xarr, dataset):
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
    assert xarr.name == ij_fixture.py.from_java(dataset.getName())


def convert_img_and_assert_equality(ij_fixture, img):
    nparr = ij_fixture.py.from_java(img)
    assert_ndarray_equal_to_img(img, nparr)


def convert_ndarray_and_assert_equality(ij_fixture, nparr):
    img = ij_fixture.py.to_java(nparr)
    assert_ndarray_equal_to_img(img, nparr)


# -- Tests --


def test_ndarray_converts_to_img(ij_fixture):
    convert_ndarray_and_assert_equality(ij_fixture, get_nparr())


def test_img_converts_to_ndarray(ij_fixture):
    convert_img_and_assert_equality(ij_fixture, get_img(ij_fixture))


def test_cstyle_array_with_labeled_dims_converts(ij_fixture):
    xarr = get_xarr()
    assert_xarray_equal_to_dataset(ij_fixture, xarr, ij_fixture.py.to_java(xarr))


def test_fstyle_array_with_labeled_dims_converts(ij_fixture):
    xarr = get_xarr("F")
    assert_xarray_equal_to_dataset(ij_fixture, xarr, ij_fixture.py.to_java(xarr))


def test_7d_rai_to_python_permute(ij_fixture):
    assert_permuted_rai_equal_to_source_rai(get_imgplus(ij_fixture))


def test_dataset_converts_to_xarray(ij_fixture):
    xarr = get_xarr()
    dataset = ij_fixture.py.to_java(xarr)
    assert_inverted_xarr_equal_to_xarr(dataset, ij_fixture, xarr)


def test_image_metadata_conversion(ij_fixture):
    # Create a ImageMetadata
    DefaultImageMetadata = sj.jimport("io.scif.DefaultImageMetadata")
    IdentityAxis = sj.jimport("net.imagej.axis.IdentityAxis")
    metadata = DefaultImageMetadata()
    lengths = sj.jarray("j", [2])
    lengths[0] = 4
    lengths[1] = 2
    metadata.populate(
        "test",  # name
        ij_fixture.py.to_java([IdentityAxis(), IdentityAxis()]),  # axes
        lengths,
        4,  # pixelType
        8,  # bitsPerPixel
        True,  # orderCertain
        True,  # littleEndian
        False,  # indexed
        False,  # falseColor
        True,  # metadataComplete
    )
    # Some properties are computed on demand - since those computed values
    # would not be grabbed in the map, let's set them
    metadata.setThumbSizeX(metadata.getThumbSizeX())
    metadata.setThumbSizeY(metadata.getThumbSizeY())
    metadata.setInterleavedAxisCount(metadata.getInterleavedAxisCount())
    # Convert to python
    py_data = ij_fixture.py.from_java(metadata)
    # Assert equality
    assert py_data["thumbSizeX"] == metadata.getThumbSizeX()
    assert py_data["thumbSizeY"] == metadata.getThumbSizeY()
    assert py_data["pixelType"] == metadata.getPixelType()
    assert py_data["bitsPerPixel"] == metadata.getBitsPerPixel()
    assert py_data["axes"] == metadata.getAxes()
    for axis in metadata.getAxes():
        assert axis.type() in py_data["axisLengths"]
        assert py_data["axisLengths"][axis.type()] == metadata.getAxisLength(axis)
    assert py_data["orderCertain"] == metadata.isOrderCertain()
    assert py_data["littleEndian"] == metadata.isLittleEndian()
    assert py_data["indexed"] == metadata.isIndexed()
    assert py_data["interleavedAxisCount"] == metadata.getInterleavedAxisCount()
    assert py_data["falseColor"] == metadata.isFalseColor()
    assert py_data["metadataComplete"] == metadata.isMetadataComplete()
    assert py_data["thumbnail"] == metadata.isThumbnail()
    assert py_data["rois"] == metadata.getROIs()
    assert py_data["tables"] == metadata.getTables()


def test_rgb_image_maintains_correct_dim_order_on_conversion(ij_fixture):
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


def test_no_coords_or_dims_in_xarr(ij_fixture):
    xarr = get_xarr("NoDims")
    dataset = ij_fixture.py.from_java(xarr)
    assert_inverted_xarr_equal_to_xarr(dataset, ij_fixture, xarr)


dataset_conversion_parameters = [
    (
        get_img,
        "java",
        ["a", "b", "c", "d", "e"],
        ("X", "Y", "Unknown", "Unknown", "Unknown"),
        (1, 2, 3, 4, 5),
    ),
    (
        get_imgplus,
        "java",
        ["a", "b", "c", "d", "e", "f", "g"],
        ("X", "Y", "foo", "bar", "Channel", "Time", "Z"),
        (7, 8, 4, 2, 3, 5, 6),
    ),
    (
        get_nparr,
        "python",
        ["t", "pln", "row", "col", "ch"],
        ("X", "Y", "Z", "Time", "Channel"),
        (4, 3, 2, 1, 5),
    ),
    (
        get_xarr,
        "python",
        ["t", "z", "y", "x", "c"],
        ("X", "Y", "Z", "Time", "Channel"),
        (12, 6, 4, 5, 3),
    ),
]
img_conversion_parameters = [
    (get_img, "java", ["a", "b", "c", "d", "e"], (1, 2, 3, 4, 5)),
    (get_imgplus, "java", ["a", "b", "c", "d", "e", "f", "g"], (7, 8, 4, 2, 3, 5, 6)),
    (get_nparr, "python", ["t", "pln", "row", "col", "ch"], (4, 3, 2, 1, 5)),
    (get_xarr, "python", ["t", "z", "y", "x", "c"], (12, 6, 4, 5, 3)),
]
xarr_conversion_parameters = [
    (
        get_img,
        "java",
        ["a", "b", "c", "d", "e"],
        ("dim_0", "dim_1", "dim_2", "dim_3", "dim_4"),
        (5, 4, 3, 2, 1),
    ),
    (
        get_imgplus,
        "java",
        ["a", "b", "c", "d", "e"],
        ("bar", "foo", "t", "pln", "row", "col", "ch"),
        (2, 4, 5, 6, 8, 7, 3),
    ),
    (
        get_nparr,
        "python",
        ["t", "pln", "row", "col", "ch"],
        ("t", "pln", "row", "col", "ch"),
        (1, 2, 3, 4, 5),
    ),
    (
        get_xarr,
        "python",
        ["t", "z", "y", "x", "c"],
        ("t", "z", "y", "x", "c"),
        (5, 4, 6, 12, 3),
    ),
]


@pytest.mark.parametrize(
    argnames="im_req,obj_type,new_dims,exp_dims,exp_shape",
    argvalues=dataset_conversion_parameters,
)
def test_direct_to_dataset_conversions(
    ij_fixture, im_req, obj_type, new_dims, exp_dims, exp_shape
):
    # get image data
    if obj_type == "java":
        im_data = im_req(ij_fixture)
    else:
        im_data = im_req()
    # convert the image data to net.image.Dataset
    ds_out = ij_fixture.py.to_dataset(im_data, dim_order=new_dims)
    assert ds_out.dims == exp_dims
    assert ds_out.shape == exp_shape
    if hasattr(im_data, "coords") and obj_type == "python":
        assert_xarray_coords_equal_to_rai_coords(im_data, ds_out)
    if images.is_xarraylike(im_data):
        assert_xarray_equal_to_dataset(ij_fixture, im_data, ds_out)
    if (images.is_arraylike is True) and (images.is_xarraylike is False):
        assert_ndarray_equal_to_img(ds_out, im_data)


@pytest.mark.parametrize(
    argnames="im_req,obj_type,new_dims,exp_shape", argvalues=img_conversion_parameters
)
def test_direct_to_img_conversions(ij_fixture, im_req, obj_type, new_dims, exp_shape):
    # get image data
    if obj_type == "java":
        im_data = im_req(ij_fixture)
    else:
        im_data = im_req()
    # convert the image data to Img
    img_out = ij_fixture.py.to_img(im_data, dim_order=new_dims)
    assert img_out.shape == exp_shape
    if images.is_xarraylike(im_data):
        assert_ndarray_equal_to_img(
            img_out, im_data.transpose("ch", "t", "pln", "row", "col").data
        )
    if (images.is_arraylike is True) and (images.is_xarraylike is False):
        assert_ndarray_equal_to_img(img_out, im_data)


@pytest.mark.parametrize(
    argnames="im_req,obj_type,new_dims,exp_dims,exp_shape",
    argvalues=xarr_conversion_parameters,
)
def test_direct_to_xarray_conversion(
    ij_fixture, im_req, obj_type, new_dims, exp_dims, exp_shape
):
    # get image data
    if obj_type == "java":
        im_data = im_req(ij_fixture)
    else:
        im_data = im_req()
    # convert the image data to xarray
    xarr_out = ij_fixture.py.to_xarray(im_data, dim_order=new_dims)
    assert xarr_out.dims == exp_dims
    assert xarr_out.shape == exp_shape
    if hasattr(im_data, "dim_axes") and obj_type == "java":
        assert_xarray_coords_equal_to_rai_coords(xarr_out, im_data)
    if sj.isjava(im_data):
        if len(im_data.shape) <= 5:
            assert_ndarray_equal_to_img(im_data, xarr_out)
        else:
            assert_ndarray_equal_to_img(
                im_data,
                xarr_out.transpose("pln", "t", "ch", "bar", "foo", "row", "col").data,
            )
