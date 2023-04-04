"""
Utility function for creating, editing and modifying metadata.
"""
from typing import Sequence

from _jpype import JClass

import imagej.dims as dims
from imagej._java import jc


class Axis:
    """
    Utility class used to generate ImageJ2 axis metadata information.
    """

    _calibrated_axis_types = {}

    @classmethod
    def _cal_axis_to_str(cls, axis: "jc.CalibratedAxis") -> str:
        """
        Convert a CalibratedAxis class to a String.
        :param axis: CalibratedAxis type (e.g. net.imagej.axis.DefaultLinearAxis).
        :return: String of CalibratedAxis typeb(e.g. "DefaultLinearAxis").
        """
        if not cls._calibrated_axis_types:
            cls._calibrated_axis_types = cls._create_calibrated_axis_dict()

        if not isinstance(axis, JClass):
            axis = axis.__class__

        return cls._calibrated_axis_types.get(axis, "unknown")

    @classmethod
    def _str_to_cal_axis(cls, axis: str) -> "jc.CalibratedAxis":
        """
        Convert a String to CalibratedAxis class.
        :param axis: String of calibratedAxis type (e.g. "DefaultLinearAxis").
        :return: Java class of CalibratedAxis type
            (e.g. net.imagej.axis.DefaultLinearAxis).
        """
        if not cls._calibrated_axis_types:
            cls._calibrated_axis_types = cls._create_calibrated_axis_dict()

        if not isinstance(axis, str):
            raise TypeError(f"Axis is not type string: {type(axis)}.")

        for k, v in cls._calibrated_axis_types.items():
            if axis == v:
                return k

        return None

    @classmethod
    def _create_calibrated_axis_dict(cls):
        """
        Create the CalibratedAxis dictionary on demand.
        """
        axis_types = {
            jc.ChapmanRichardsAxis: "ChapmanRichardsAxis",
            jc.DefaultLinearAxis: "DefaultLinearAxis",
            jc.EnumeratedAxis: "EnumeratedAxis",
            jc.ExponentialAxis: "ExponentialAxis",
            jc.ExponentialRecoveryAxis: "ExponentialRecoveryAxis",
            jc.GammaVariateAxis: "GammaVariateAxis",
            jc.GaussianAxis: "GaussianAxis",
            jc.IdentityAxis: "IdentityAxis",
            jc.InverseRodbardAxis: "InverseRodbardAxis",
            jc.LogLinearAxis: "LogLinearAxis",
            jc.PolynomialAxis: "PolynomialAxis",
            jc.PowerAxis: "PowerAxis",
            jc.RodbardAxis: "RodbardAxis",
        }

        return axis_types


class ImageMetadata:
    """
    Utility class used to create and update image metadata.
    """

    @classmethod
    def create_imagej_metadata(
        cls, axes: Sequence["jc.CalibratedAxis"], dim_seq: Sequence[str]
    ) -> dict:
        """
        Create the ImageJ metadata attribute dictionary for xarray's global attributes.
        :param axes: A list or tuple of ImageJ2 axis objects
            (e.g. net.imagej.axis.DefaultLinearAxis).
        :param dim_seq: A list or tuple of the dimension order (e.g. ['X', 'Y', 'C']).
        :return: Dict of image metadata.
        """
        ij_metadata = {}
        if len(axes) != len(dim_seq):
            raise ValueError(
                f"Axes length ({len(axes)}) does not match \
                    dimension length ({len(dim_seq)})."
            )

        for i in range(len(axes)):
            # get CalibratedAxis type as string (e.g. "EnumeratedAxis")
            ij_metadata[
                dims._to_ijdim(dim_seq[i]) + "_cal_axis_type"
            ] = Axis._cal_axis_to_str(axes[i])
            # get scale and origin for DefaultLinearAxis
            if isinstance(axes[i], jc.DefaultLinearAxis):
                ij_metadata[dims._to_ijdim(dim_seq[i]) + "_scale"] = float(
                    axes[i].scale()
                )
                ij_metadata[dims._to_ijdim(dim_seq[i]) + "_origin"] = float(
                    axes[i].origin()
                )

        return ij_metadata
