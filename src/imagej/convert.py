"""
Utility functions for converting objects between types.
"""
import ctypes
import logging
import os
from typing import Dict, Sequence, Union

import imglyb
import numpy as np
import scyjava as sj
import xarray as xr
from jpype import JByte, JException, JFloat, JLong, JObject, JShort
from labeling import Labeling

import imagej.dims as dims
import imagej.images as images
from imagej._java import jc
from imagej._java import log_exception as _log_exception

_logger = logging.getLogger(__name__)


###############
# Java images #
###############


def java_to_dataset(ij: "jc.ImageJ", jobj) -> "jc.Dataset":
    """
    Convert the given Java image data into an ImageJ2 Dataset.

    :param ij: The ImageJ2 gateway (see imagej.init)
    :param jobj: The Java image (e.g. RandomAccessibleInterval)
    :return: The converted ImageJ2 Dataset
    """
    assert sj.isjava(jobj)
    if isinstance(jobj, jc.Dataset):
        return jobj

    # NB: This try checking is necessary because the set of ImageJ2 converters is
    # not complete. E.g., there is no way to directly go from Img to Dataset,
    # instead you need to chain the Img->ImgPlus->Dataset converters.
    try:
        if ij.convert().supports(jobj, jc.Dataset):
            return ij.convert().convert(jobj, jc.Dataset)
        if ij.convert().supports(jobj, jc.ImgPlus):
            imgplus = ij.convert().convert(jobj, jc.ImgPlus)
            return ij.dataset().create(imgplus)
        if ij.convert().supports(jobj, jc.Img):
            img = ij.convert().convert(jobj, jc.Img)
            return ij.dataset().create(jc.ImgPlus(img))
        if ij.convert().supports(jobj, jc.RandomAccessibleInterval):
            rai = ij.convert().convert(jobj, jc.RandomAccessibleInterval)
            return ij.dataset().create(rai)
    except Exception as exc:
        _log_exception(_logger, exc)
        raise exc
    raise TypeError("Cannot convert to Dataset: " + str(type(jobj)))


def java_to_img(ij: "jc.ImageJ", jobj) -> "jc.Img":
    """
    Convert the data into an ImgLib2 Img.

    :param ij: The ImageJ2 gateway (see imagej.init)
    :param jobj: The Java image (e.g. RandomAccessibleInterval)
    :return: The converted ImgLib2 Img
    """
    assert sj.isjava(jobj)
    if isinstance(jobj, jc.Img):
        return jobj

    # NB: This try checking is necessary because the set of ImageJ2
    # converters is not complete.
    try:
        if ij.convert().supports(jobj, jc.Img):
            return ij.convert().convert(jobj, jc.Img)
        if ij.convert().supports(jobj, jc.RandomAccessibleInterval):
            rai = ij.convert().convert(jobj, jc.RandomAccessibleInterval)
            return jc.ImgView.wrap(rai)
    except Exception as exc:
        _log_exception(_logger, exc)
        raise exc
    raise TypeError("Cannot convert to img: " + str(type(jobj)))


def imageplus_to_imgplus(ij: "jc.ImageJ", imp: "jc.ImagePlus") -> "jc.ImgPlus":
    """
    Convert the given ImageJ ImagePlus into an ImageJ2 ImgPlus.

    :param ij: The ImageJ2 gateway (see imagej.init)
    :param imp: The ImageJ ImagePlus
    :return: The converted ImageJ2 ImgPlus
    """
    if not jc.ImagePlus or not isinstance(imp, jc.ImagePlus):
        raise ValueError("Input is not an ImagePlus")

    ds = ij.convert().convert(imp, jc.Dataset)
    return ds.getImgPlus()


##############################
# Java image <-> NumPy array #
##############################


def ndarray_to_dataset(ij: "jc.ImageJ", narr) -> "jc.Dataset":
    """
    Convert the given NumPy ndarray into an ImageJ2 Dataset.

    :param ij: The ImageJ2 gateway (see imagej.init)
    :param narr: The NumPy ndarray
    :return: The converted ImageJ2 Dataset
    """
    assert images.is_arraylike(narr)
    rai = imglyb.to_imglib(narr)
    return java_to_dataset(ij, rai)


def ndarray_to_img(ij: "jc.ImageJ", narr) -> "jc.Img":
    """
    Convert the given NumPy ndarray into an ImgLib2 Img.

    :param ij: The ImageJ2 gateway (see imagej.init)
    :param narr: The NumPy ndarray
    :return: The converted ImgLib2 Img
    """
    assert images.is_arraylike(narr)
    rai = imglyb.to_imglib(narr)
    return java_to_img(ij, rai)


def ndarray_to_xarray(narr: np.ndarray, dim_order=None) -> xr.DataArray:
    """
    Convert the given NumPy ndarray into an xarray.DataArray. A dict with
    key 'dim_order' and a dimension order in a List[str] is required.

    :param narr: The NumPy ndarray
    :param dim_order: List of desired dimensions for the xarray.DataArray.
    :return: The converted xarray.DataArray
    """
    assert images.is_arraylike(narr)
    if dim_order:
        # check dim length
        dim_order = dims._validate_dim_order(dim_order, narr.shape)
        return xr.DataArray(narr, dims=dim_order)
    return xr.DataArray(narr)


def xarray_to_dataset(ij: "jc.ImageJ", xarr) -> "jc.Dataset":
    """
    Converts an xarray DataArray to an ImageJ2 Dataset,
    inverting C-style (slow axis first) to F-style (slow-axis last).

    :param ij: The ImageJ2 gateway (see imagej.init)
    :param xarr: The xarray DataArray
    :return: The converted ImageJ2 Dataset
    """
    assert images.is_xarraylike(xarr)
    if dims._ends_with_channel_axis(xarr):
        vals = np.moveaxis(xarr.values, -1, 0)
        dataset = ndarray_to_dataset(ij, vals)
    else:
        dataset = ndarray_to_dataset(ij, xarr.values)
    axes = dims._assign_axes(xarr)
    dataset.setAxes(axes)
    dataset.setName(xarr.name)
    _assign_dataset_metadata(dataset, xarr.attrs)

    return dataset


def xarray_to_img(ij: "jc.ImageJ", xarr) -> "jc.Img":
    """
    Converts an xarray DataArray into an ImgLib2 Img,
    inverting C-style (slow axis first) to F-style (slow axis last).

    :param ij: The ImageJ2 gateway (see imagej.init)
    :param xarr: The xarray DataArray
    :return: The converted ImgLib2 Img with inverted axes
    """
    assert images.is_xarraylike(xarr)
    if dims._ends_with_channel_axis(xarr):
        vals = np.moveaxis(xarr.values, -1, 0)
        return ndarray_to_img(ij, vals)
    else:
        return ndarray_to_img(ij, xarr.values)


def java_to_ndarray(ij: "jc.ImageJ", jobj) -> np.ndarray:
    """
    Convert a Java image to a NumPy ndarray,
    inverting F-style (slow axis last) to C-style (slow axis first).

    :param ij: The ImageJ2 gateway (see imagej.init)
    :param jobj: The Java image (e.g. RandomAccessibleInterval)
    :return: The converted NumPy ndarray with inverted axes
    """
    assert sj.isjava(jobj)
    rai = ij.convert().convert(jobj, jc.RandomAccessibleInterval)
    narr = images.create_ndarray(rai)
    images.copy_rai_into_ndarray(ij, rai, narr)
    return narr


def java_to_xarray(ij: "jc.ImageJ", jobj) -> xr.DataArray:
    """
    Convert a Java image to an xarray DataArray,
    inverting F-style (slow axis last) to C-style (slow axis first).

    Labeled dimensional axes are permuted as needed
    to conform to the scikit-image standard order; see:
    https://scikit-image.org/docs/dev/user_guide/numpy_images#coordinate-conventions

    :param ij: The ImageJ2 gateway (see imagej.init)
    :param jobj: The Java image with labeled axes (e.g. Dataset or ImgPlus)
    :return: The converted xarray DataArray with standardized axes
    """
    imgplus = ij.convert().convert(jobj, jc.ImgPlus)

    # Permute Java image dimensions to the scikit-image standard order.
    permuted_rai = _permute_rai_to_python(imgplus)

    # Create a new ndarray, and copy the Java image into it.
    narr = images.create_ndarray(permuted_rai)
    images.copy_rai_into_ndarray(ij, permuted_rai, narr)

    # Wrap ndarray into an xarray with axes matching the permuted RAI.
    assert hasattr(permuted_rai, "dim_axes")
    xr_axes = list(permuted_rai.dim_axes)
    xr_dims = list(permuted_rai.dims)
    xr_attrs = sj.to_python(permuted_rai.getProperties())
    xr_attrs = {sj.to_python(k): sj.to_python(v) for k, v in xr_attrs.items()}
    xr_attrs["imagej"] = _create_imagej_metadata(xr_axes, xr_dims)
    # reverse axes and dims to match narr
    xr_axes.reverse()
    xr_dims.reverse()
    xr_dims = dims._convert_dims(xr_dims, direction="python")
    xr_coords = dims._get_axes_coords(xr_axes, xr_dims, narr.shape)
    name = jobj.getName() if isinstance(jobj, jc.Named) else None
    return xr.DataArray(narr, dims=xr_dims, coords=xr_coords, attrs=xr_attrs, name=name)


def supports_java_to_ndarray(ij: "jc.ImageJ", obj) -> bool:
    """
    Return True iff the given object is convertible to a NumPy ndarray
    via the java_to_ndarray function.

    :param ij: The ImageJ2 gateway (see imagej.init)
    :param obj: The object to check for convertibility
    :return: True iff conversion to a NumPy ndarray is possible
    """
    try:
        return ij.convert().supports(obj, jc.RandomAccessibleInterval)
    except Exception:
        return False


def supports_java_to_xarray(ij: "jc.ImageJ", obj) -> bool:
    """
    Return True iff the given object is convertible to a NumPy ndarray
    via the java_to_xarray function.

    :param ij: The ImageJ2 gateway (see imagej.init)
    :param obj: The object to check for convertibility
    :return: True iff conversion to an xarray DataArray is possible
    """
    try:
        can_convert = ij.convert().supports(obj, jc.ImgPlus)
        has_axis = dims._has_axis(obj)
        return can_convert and has_axis
    except Exception:
        return False


######################
# ctype <-> RealType #
######################

# Dict between ctypes and equivalent RealTypes
# These types were chosen to guarantee the number of bits in each
# ctype, as the sizes of some ctypes are platform-dependent. See
# https://docs.python.org/3/library/ctypes.html#ctypes-fundamental-data-types-2
# for more information.
_ctype_map: Dict[type, str] = {
    ctypes.c_bool: "net.imglib2.type.logic.BoolType",
    ctypes.c_int8: "net.imglib2.type.numeric.integer.ByteType",
    ctypes.c_uint8: "net.imglib2.type.numeric.integer.UnsignedByteType",
    ctypes.c_int16: "net.imglib2.type.numeric.integer.ShortType",
    ctypes.c_uint16: "net.imglib2.type.numeric.integer.UnsignedShortType",
    ctypes.c_int32: "net.imglib2.type.numeric.integer.IntType",
    ctypes.c_uint32: "net.imglib2.type.numeric.integer.UnsignedIntType",
    ctypes.c_int64: "net.imglib2.type.numeric.integer.LongType",
    ctypes.c_uint64: "net.imglib2.type.numeric.integer.UnsignedLongType",
    ctypes.c_float: "net.imglib2.type.numeric.real.FloatType",
    ctypes.c_double: "net.imglib2.type.numeric.real.DoubleType",
}

# Dict of casters for realtypes that cannot directly take
# the raw conversion of ctype.value
_realtype_casters: Dict[str, type] = {
    "net.imglib2.type.numeric.integer.ByteType": JByte,
    "net.imglib2.type.numeric.integer.UnsignedIntType": JLong,
    "net.imglib2.type.numeric.integer.ShortType": JShort,
    "net.imglib2.type.numeric.integer.LongType": JLong,
    "net.imglib2.type.numeric.integer.UnsignedLongType": JLong,
    "net.imglib2.type.numeric.real.DoubleType": JFloat,
}


def ctype_to_realtype(ctype: ctypes._SimpleCData):
    """
    Convert the given Python ctype into an ImgLib2 RealType.

    :param ij: The ImageJ2 gateway (see imagej.init)
    :param ctype: The Python ctype
    :return: The converted ImgLib2 RealType
    """
    # First, convert the ctype value to java
    jtype_raw = sj.to_java(ctype.value)
    # Then, find the correct RealType
    realtype_fqcn = _ctype_map[type(ctype)]
    # jtype_raw is usually an Integer or Double.
    # We may have to cast it to fit the RealType parameter
    if realtype_fqcn in _realtype_casters:
        caster = _realtype_casters[realtype_fqcn]
        jtype_raw = caster(jtype_raw)
    # Create and return the RealType
    realtype_class = sj.jimport(realtype_fqcn)
    return realtype_class(jtype_raw)


def realtype_to_ctype(realtype: "jc.RealType"):
    """
    Convert the given ImgLib2 RealType into a Python ctype.

    :param ij: The ImageJ2 gateway (see imagej.init)
    :param realtype: The ImgLib2 RealType
    :return: The converted Python ctype
    """
    # First, convert the RealType to a Java primitive
    jtype_raw = realtype.get()
    # Then, convert to the python primitive
    converted = sj.to_python(jtype_raw)
    value = realtype.getClass().getName()
    for k, v in _ctype_map.items():
        if v == value:
            return k(converted)
    raise ValueError(f"Cannot convert RealType {value}")


def supports_ctype_to_realtype(obj):
    """
    Return True iff the given object is convertible to an ImgLib2 RealType
    via the ctype_to_realtype function.

    :param obj: The object to check for convertibility
    :return: True iff conversion to an ImgLib2 RealType is possible
    """
    return type(obj) in _ctype_map


def supports_realtype_to_ctype(obj):
    """
    Return True iff the given object is convertible to a Python ctype
    via the realtype_to_ctype function.

    :param obj: The object to check for convertibility
    :return: True iff conversion to a Python ctype is possible
    """
    if not isinstance(obj, sj.jimport("net.imglib2.type.numeric.RealType")):
        return False
    fqcn = obj.getClass().getName()
    return fqcn in _ctype_map.values()


############################
# Labeling <-> ImgLabeling #
############################


def labeling_to_imglabeling(ij: "jc.ImageJ", labeling: Labeling):
    """
    Convert a Python Labeling to an equivalent Java ImgLabeling.

    :param ij: The ImageJ2 gateway (see imagej.init)
    :param labeling: the Python Labeling
    :return: a Java ImgLabeling
    """
    labeling_io_service = ij.context().service(jc.LabelingIOService)

    # Save the image on the Python side
    tmp_pth = "./tmp"
    _delete_labeling_files(tmp_pth)
    labeling.save_result(tmp_pth)

    # Load the labeling on the Java side
    try:
        tmp_pth_json = tmp_pth + ".lbl.json"
        imglabeling = labeling_io_service.load(tmp_pth_json, JObject, JObject)
    except JException as exc:
        _delete_labeling_files(tmp_pth)
        raise exc
    _delete_labeling_files(tmp_pth)

    return imglabeling


def imglabeling_to_labeling(ij: "jc.ImageJ", imglabeling: "jc.ImgLabeling"):
    """
    Convert a Java ImgLabeling to an equivalent Python Labeling.

    :param ij: The ImageJ2 gateway (see imagej.init)
    :param imglabeling: the Java ImgLabeling
    :return: a Python Labeling
    """
    labeling_io_service = ij.context().service(jc.LabelingIOService)

    # Save the image on the Python side
    tmp_pth = os.getcwd() + "/tmp"
    tmp_pth_json = tmp_pth + ".lbl.json"
    tmp_pth_tif = tmp_pth + ".tif"
    try:
        _delete_labeling_files(tmp_pth)
        imglabeling = ij.convert().convert(imglabeling, jc.ImgLabeling)
        labeling_io_service.save(
            imglabeling, tmp_pth_tif
        )  # TODO: improve, likely utilizing the ImgLabeling's name
    except JException:
        print("Failed to save the data")

    # Load the labeling on the python side
    labeling = Labeling.from_file(tmp_pth_json)
    _delete_labeling_files(tmp_pth)
    return labeling


def supports_labeling_to_imglabeling(obj):
    """
    Return True iff the given object is convertible to an ImgLib2 ImgLabeling
    via the labeling_to_imglabeling function.

    :param ij: The ImageJ2 gateway (see imagej.init)
    :param obj: The object to check for convertibility
    :return: True iff conversion to an ImgLib2 ImgLabeling is possible
    """
    return isinstance(obj, Labeling)


def supports_imglabeling_to_labeling(obj):
    """
    Return True iff the given object is convertible to a Python Labeling
    via the imglabeling_to_labeling function.

    :param ij: The ImageJ2 gateway (see imagej.init)
    :param obj: The object to check for convertibility
    :return: True iff conversion to a Python Labeling is possible
    """
    return isinstance(obj, jc.ImgLabeling)


#######################
# Metadata converters #
#######################


def image_metadata_to_dict(ij: "jc.ImageJ", image_meta: "jc.ImageMetadata"):
    """
    Converts an io.scif.ImageMetadata to a Python dict.
    The components should be enough to create a new ImageMetadata.

    :param ij: The ImageJ2 gateway (see imagej.init)
    :param image_meta: The ImageMetadata to convert
    :return: A Python dict representing image_meta
    """

    # We import io.scif.Field here.
    # This will prevent any conflicts with java.lang.reflect.Field.
    Field = sj.jimport("io.scif.Field")

    # Convert to a dict - preserve information by copying all SCIFIO fields.
    #
    # If info is left out of this dict, make sure that
    # information is annotated with @Field upstream!
    return {
        str(f.getName()): ij.py.from_java(jc.ClassUtils.getValue(f, image_meta))
        for f in jc.ClassUtils.getAnnotatedFields(image_meta.getClass(), Field)
    }


def metadata_wrapper_to_dict(ij: "jc.ImageJ", metadata_wrapper: "jc.MetadataWrapper"):
    """
    Converts a io.scif.filters.MetadataWrapper to a Python Dict.
    The components should be enough to create a new MetadataWrapper

    :param ij: The ImageJ2 gateway (see imagej.init)
    :param metadata_wrapper: The MetadataWrapper to convert
    :return: A Python dict representing metadata_wrapper
    """

    return dict(
        impl_cls=type(metadata_wrapper),
        metadata=metadata_wrapper.unwrap(),
    )


####################
# Helper functions #
####################


def _assign_dataset_metadata(dataset: "jc.Dataset", attrs):
    """
    :param dataset: ImageJ2 Dataset
    :param attrs: Dictionary containing metadata
    """
    dataset.getProperties().putAll(sj.to_java(attrs))


def _permute_rai_to_python(rich_rai: "jc.RandomAccessibleInterval"):
    """Permute a RandomAccessibleInterval to the python reference order.

    Permute a RandomAccessibleInterval to the Python reference order of
    CXYZT (where dimensions exist). Note that this is reverse from the
    final array order of TZYXC.

    :param rich_rai: A RandomAccessibleInterval with axis labels
        (e.g. Dataset or ImgPlus).
    :return: A permuted RandomAccessibleInterval
    """
    # get input rai metadata if it exists
    try:
        rai_metadata = rich_rai.getProperties()
    except AttributeError:
        rai_metadata = None

    axis_types = [axis.type() for axis in rich_rai.dim_axes]

    # permute rai to specified order and transfer metadata
    permute_order = dims.prioritize_rai_axes_order(
        axis_types, dims._python_rai_ref_order()
    )
    permuted_rai = dims.reorganize(rich_rai, permute_order)

    # add metadata to image if it exists
    if rai_metadata is not None:
        permuted_rai.getProperties().putAll(rai_metadata)

    return permuted_rai


def _rename_xarray_dims(xarr, new_dims: Sequence[str]):
    curr_dims = xarr.dims
    if not new_dims:
        return xarr
    # check dim length
    new_dims = dims._validate_dim_order(new_dims, xarr.shape)
    dim_map = {}
    for i in range(xarr.ndim):
        dim_map[curr_dims[i]] = new_dims[i]

    return xarr.rename(dim_map)


def _create_imagej_metadata(
    axes: Sequence[Union["jc.DefaultLinearAxis", "jc.EnumeratedAxis"]],
    dim_seq: Sequence[str],
) -> dict:
    """
    Create the ImageJ metadata attribute dictionary for xarray's global attributes.
    """
    ij_metadata = {}
    assert len(axes) == len(dim_seq)
    for i in range(len(axes)):
        # get CalibratedAxis type as string (e.g. "EnumeratedAxis")
        ij_metadata[
            dims._to_ijdim(dim_seq[i]) + "_cal_axis_type"
        ] = dims._cal_axis_to_str(axes[i])
        # get scale and origin for DefaultLinearAxis
        if isinstance(axes[i], jc.DefaultLinearAxis):
            ij_metadata[dims._to_ijdim(dim_seq[i]) + "_scale"] = float(axes[i].scale())
            ij_metadata[dims._to_ijdim(dim_seq[i]) + "_origin"] = float(
                axes[i].origin()
            )

    return ij_metadata


def _delete_labeling_files(filepath):
    """
    Removes any Labeling data left over at filepath
    :param filepath: the filepath where Labeling (might have) saved data
    """
    pth_json = filepath + ".lbl.json"
    pth_tif = filepath + ".tif"
    if os.path.exists(pth_tif):
        os.remove(pth_tif)
    if os.path.exists(pth_json):
        os.remove(pth_json)


def _dim_order(hints: Dict):
    """
    Extract the dim_order from the hints kwargs.
    """
    return hints["dim_order"] if "dim_order" in hints else None
