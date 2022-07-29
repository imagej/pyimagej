Blob detection (headless)
============================

This example demonstrates a mixed workflow using PyImageJ to convert a Java image into an :code:`xarray.DataArray` (*i.e.* a metadata wrapped NumPyArray),
run scikit-image's Lapacian of Gaussian (LoG) blob detection (:code:`skimage.feature.blob_log()`) to identify puncta, convert the blob LoG detections
into ImageJ regions of interest and perform measurements. This example is intended to run with `X virtual framebuffer`_ in order to achieve a
headless environment. Xvfb is needed in this example, in order to make use of the ImageJ ROI Manager, which is dependent on a graphical
user interface.

.. literalinclude:: blob_detection_xvfb.py
    :language: python

.. _X virtual framebuffer: https://www.x.org/releases/X11R7.7/doc/man/man1/Xvfb.1.xhtml