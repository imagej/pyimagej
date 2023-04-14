from _jpype import JClass

from imagej._java import jc

_calibrated_axes = [
    "net.imagej.axis.ChapmanRichardsAxis",
    "net.imagej.axis.DefaultLinearAxis",
    "net.imagej.axis.EnumeratedAxis",
    "net.imagej.axis.ExponentialAxis",
    "net.imagej.axis.ExponentialRecoveryAxis",
    "net.imagej.axis.GammaVariateAxis",
    "net.imagej.axis.GaussianAxis",
    "net.imagej.axis.IdentityAxis",
    "net.imagej.axis.InverseRodbardAxis",
    "net.imagej.axis.LogLinearAxis",
    "net.imagej.axis.PolynomialAxis",
    "net.imagej.axis.PowerAxis",
    "net.imagej.axis.RodbardAxis",
]


def calibrated_axis_to_str(axis: "jc.CalibratedAxis") -> str:
    """
    Convert a CalibratedAxis class to a String.
    :param axis: CalibratedAxis type (e.g. net.imagej.axis.DefaultLinearAxis).
    :return: String of CalibratedAxis typeb(e.g. "DefaultLinearAxis").
    """
    if not isinstance(axis, JClass):
        axis = axis.__class__

    return str(axis).split("'")[1]


def str_to_calibrated_axis(axis: str) -> "jc.CalibratedAxis":
    """
    Convert a String to CalibratedAxis class.
    :param axis: String of calibratedAxis type (e.g. "DefaultLinearAxis").
    :return: Java class of CalibratedAxis type
        (e.g. net.imagej.axis.DefaultLinearAxis).
    """
    if not isinstance(axis, str):
        raise TypeError(f"Axis {type(axis)} is not a String.")

    if axis in _calibrated_axes:
        return getattr(jc, axis.split(".")[3])
    else:
        return None
