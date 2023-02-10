"""
Internal utility functions for working with Java objects.
These are not intended for external use in PyImageJ-based scripts!
"""
import logging
from functools import lru_cache

from jpype import JArray, JObject
from scyjava import JavaClasses, jstacktrace


def log_exception(logger: logging.Logger, exc: "jc.Throwable") -> None:
    if logger.isEnabledFor(logging.DEBUG):
        jtrace = jstacktrace(exc)
        if jtrace:
            logger.debug(jtrace)


# Import Java resources on demand.


@lru_cache(maxsize=None)
def JObjectArray():
    return JArray(JObject)


class MyJavaClasses(JavaClasses):
    """
    Utility class used to make importing frequently-used Java classes
    significantly easier and more readable.
    """

    @JavaClasses.java_import
    def Double(self):
        return "java.lang.Double"

    @JavaClasses.java_import
    def Throwable(self):
        return "java.lang.Throwable"

    @JavaClasses.java_import
    def ImagePlus(self):
        return "ij.ImagePlus"

    @JavaClasses.java_import
    def ImageMetadata(self):
        return "io.scif.ImageMetadata"

    @JavaClasses.java_import
    def MetadataWrapper(self):
        return "io.scif.filters.MetadataWrapper"

    @JavaClasses.java_import
    def LabelingIOService(self):
        return "io.scif.labeling.LabelingIOService"

    @JavaClasses.java_import
    def ChapmanRichardsAxis(self):
        return "net.imagej.axis.ChapmanRichardsAxis"

    @JavaClasses.java_import
    def DefaultLinearAxis(self):
        return "net.imagej.axis.DefaultLinearAxis"

    @JavaClasses.java_import
    def EnumeratedAxis(self):
        return "net.imagej.axis.EnumeratedAxis"

    @JavaClasses.java_import
    def ExponentialAxis(self):
        return "net.imagej.axis.ExponentialAxis"

    @JavaClasses.java_import
    def ExponentialRecoveryAxis(self):
        return "net.imagej.axis.ExponentialRecoveryAxis"

    @JavaClasses.java_import
    def GammaVariateAxis(self):
        return "net.imagej.axis.GammaVariateAxis"

    @JavaClasses.java_import
    def GaussianAxis(self):
        return "net.imagej.axis.GaussianAxis"

    @JavaClasses.java_import
    def IdentityAxis(self):
        return "net.imagej.axis.IdentityAxis"

    @JavaClasses.java_import
    def InverseRodbardAxis(self):
        return "net.imagej.axis.InverseRodbardAxis"

    @JavaClasses.java_import
    def LogLinearAxis(self):
        return "net.imagej.axis.LogLinearAxis"

    @JavaClasses.java_import
    def PolynomialAxis(self):
        return "net.imagej.axis.PolynomialAxis"

    @JavaClasses.java_import
    def PowerAxis(self):
        return "net.imagej.axis.PowerAxis"

    @JavaClasses.java_import
    def RodbardAxis(self):
        return "net.imagej.axis.RodbardAxis"

    @JavaClasses.java_import
    def VariableAxis(self):
        return "net.imagej.axis.VariableAxis"

    @JavaClasses.java_import
    def Dataset(self):
        return "net.imagej.Dataset"

    @JavaClasses.java_import
    def ImageJ(self):
        return "net.imagej.ImageJ"

    @JavaClasses.java_import
    def ImgPlus(self):
        return "net.imagej.ImgPlus"

    @JavaClasses.java_import
    def Axes(self):
        return "net.imagej.axis.Axes"

    @JavaClasses.java_import
    def Axis(self):
        return "net.imagej.axis.Axis"

    @JavaClasses.java_import
    def AxisType(self):
        return "net.imagej.axis.AxisType"

    @JavaClasses.java_import
    def CalibratedAxis(self):
        return "net.imagej.axis.CalibratedAxis"

    @JavaClasses.java_import
    def ClassUtils(self):
        return "org.scijava.util.ClassUtils"

    @JavaClasses.java_import
    def Dimensions(self):
        return "net.imglib2.Dimensions"

    @JavaClasses.java_import
    def RandomAccessibleInterval(self):
        return "net.imglib2.RandomAccessibleInterval"

    @JavaClasses.java_import
    def ImgMath(self):
        return "net.imglib2.algorithm.math.ImgMath"

    @JavaClasses.java_import
    def Img(self):
        return "net.imglib2.img.Img"

    @JavaClasses.java_import
    def ImgView(self):
        return "net.imglib2.img.ImgView"

    @JavaClasses.java_import
    def ImgLabeling(self):
        return "net.imglib2.roi.labeling.ImgLabeling"

    @JavaClasses.java_import
    def Named(self):
        return "org.scijava.Named"

    @JavaClasses.java_import
    def Util(self):
        return "net.imglib2.util.Util"

    @JavaClasses.java_import
    def Views(self):
        return "net.imglib2.view.Views"


jc = MyJavaClasses()
