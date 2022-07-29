Blob detection (interactive)
============================

This example demonstrates a mixed workflow using PyImageJ to convert a Java image into an :code:`xarray.DataArray` (*i.e.* a metadata wrapped NumPyArray),
run scikit-image's Lapacian of Gaussian (LoG) blob detection (:code:`skimage.feature.blob_log()`) to identify puncta and then convert the blob LoG detections
into ImageJ regions of interest. The image with the blob LoG detections are displayed in ImageJ's image viewer, :code:`matplotlib.pyplot` and napari.

.. literalinclude:: blob_detection_interactive.py
    :language: python