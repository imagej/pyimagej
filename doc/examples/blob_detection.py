import imagej
import scyjava as sj
import napari
import xarray as xr
import numpy as np

from math import sqrt
from skimage.feature import blob_log
from matplotlib import pyplot as plt


def find_blobs(image: xr.DataArray, min_sigma: float, max_sigma: float, num_sigma: int, threshold=0.1, show=False) -> np.ndarray:
    """
    Find blobs with Laplacian of Gaussian (LoG).
    """
    # detect blobs in image
    blobs = blob_log(image, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma, threshold=threshold)
    blobs[:, 2] = blobs[:, 2] * sqrt(2)

    return blobs


def detections_to_pyplot(image: xr.DataArray, detections: np.ndarray):
    """
    Display image with detections in matplotlib.pyplot.
    """
    fig, ax = plt.subplots(1, 2, figsize=(10,8), sharex=True, sharey=True)
    ax[0].imshow(image, interpolation='nearest')
    ax[1].imshow(image, interpolation='nearest')
    for blob in detections:
        y, x, r = blob
        c = plt.Circle((x, y), r, color='white', linewidth=1, fill=False)
        ax[1].add_patch(c)

    plt.tight_layout
    plt.show()


def detections_to_imagej(dataset, detections: np.ndarray, add_to_roi_manager=False):
    """
    Convert blob detections to ImageJ oval ROIs.
    Optionally add the ROIs to the RoiManager.
    """
    # get ImageJ resources
    ImagePlus = sj.jimport('ij.ImagePlus')
    OvalRoi = sj.jimport('ij.gui.OvalRoi')
    Overlay = sj.jimport('ij.gui.Overlay')
    ov = Overlay()

    # convert Dataset to ImagePlus
    imp = ij.convert().convert(dataset, ImagePlus)

    if add_to_roi_manager:
        RoiManager = sj.jimport('ij.plugin.frame.RoiManager')()
        rm = RoiManager.getRoiManager()

    for i in range(len(detections)):
        values = detections[i].tolist()
        y = values[0]
        x = values[1]
        r = values[2]
        d = r * 2
        roi = OvalRoi(x - r, y - r, d, d)
        imp.setRoi(roi)
        ov.add(roi)
        if add_to_roi_manager:
            rm.addRoi(roi)

    imp.setOverlay(ov)
    imp.show()


def detections_to_napari(image_array: xr.DataArray, detections: np.ndarray):
    """
    Convert blob detections to Napari oval ROIs.
    """
    ovals = []
    for i in range(len(detections)):
        values = detections[i].tolist()
        y = values[0]
        x = values[1]
        r = values[2]
        pos_1 = [y - r, x - r] # top left
        pos_2 = [y - r, x + r] # top right
        pos_3 = [y + r, x + r] # bottom right
        pos_4 = [y + r, x - r] # bottom left
        ovals.append([pos_1, pos_2, pos_3, pos_4])

    napari_detections = np.asarray(ovals)
    
    viewer = napari.Viewer()
    viewer.add_image(image_array)
    shapes_layer = viewer.add_shapes()
    shapes_layer.add(
        napari_detections,
        shape_type='ellipse',
        edge_width=1,
        edge_color='yellow',
        face_color='royalblue',
    )

    napari.run()


def detections_to_bdv(image, detections:np.ndarray):
    """
    Display image and ROIs in BigDataViewer.
    """
    # get useful classes
    BdvFunctions = sj.jimport('bdv.util.BdvFunctions')
    BdvOptions = sj.jimport('bdv.util.BdvOptions')

    # show image
    BdvFunctions.show(image, 'test', BdvOptions.options().is2D())


if __name__ == "__main__":
    # initialize imagej
    ij = imagej.init(mode='interactive')
    print(f"ImageJ version: {ij.getVersion()}")

    # load some sample data
    img = ij.io().open('../sample-data/test_image.tif')
    img_xr = ij.py.from_java(img)
    detected_blobs = find_blobs(img_xr, min_sigma=0.5, max_sigma=3, num_sigma=10, threshold=0.0075)
    detections_to_pyplot(img_xr, detected_blobs)
    detections_to_imagej(img, detected_blobs, True)
    detections_to_bdv(img, detected_blobs)
    detections_to_napari(img_xr, detected_blobs)