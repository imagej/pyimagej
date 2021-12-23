import imagej
import napari
import xarray as xr
import numpy as np
import code
import scyjava as sj
from math import sqrt
from skimage import exposure
from skimage.feature import blob_log
from skimage.color import rgb2gray
from matplotlib import pyplot as plt

def find_blobs(image: xr.DataArray, min_sigma: float, max_sigma: float, num_sigma: int, threshold=0.1, show=False) -> np.ndarray:
    if isinstance(image, xr.DataArray):
        image = image.squeeze()
        image = image.data

    # detect blobs
    image_gray_scale = rgb2gray(image)
    blobs = blob_log(image_gray_scale, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma, threshold=threshold)
    blobs[:, 2] = blobs[:, 2] * sqrt(2)

    if show:
        fig, ax = plt.subplots(1, 2, figsize=(10,8), sharex=True, sharey=True)
        ax[0].imshow(image, interpolation='nearest')
        ax[1].imshow(image, interpolation='nearest')
        for blob in blobs:
            y, x, r = blob
            c = plt.Circle((x, y), r, color='white', linewidth=1, fill=False)
            ax[1].add_patch(c)

        plt.tight_layout()
        plt.show()

    return blobs

def array_to_oval_roi(image, circles_array: np.ndarray, add_to_roi_manager=False):
    # get ImageJ resources
    ImagePlus = sj.jimport('ij.ImagePlus')
    OvalRoi = sj.jimport('ij.gui.OvalRoi')
    Overlay = sj.jimport('ij.gui.Overlay')
    ov = Overlay()

    # convert to imp
    imp = ij.convert().convert(image, ImagePlus)

    if add_to_roi_manager:
        RoiManager = sj.jimport('ij.plugin.frame.RoiManager')()
        rm = RoiManager.getRoiManager()

    for i in range(len(circles_array)):
        values = circles_array[i].tolist()
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

    return None

def array_to_napari(image_array: xr.DataArray, circles_array: np.ndarray):
    circles = []
    for i in range(len(circles_array)):
        values = circles_array[i].tolist()
        y = values[0]
        x = values[1]
        r = values[2]
        pos_1 = [y - r, x - r] # top left
        pos_2 = [y - r, x + r] # top right
        pos_3 = [y + r, x + r] # bottom right
        pos_4 = [y + r, x - r] # bottom left
        circles.append([pos_1, pos_2, pos_3, pos_4])

    napari_circles_array = np.asarray(circles)
    
    viewer = napari.Viewer()
    viewer.add_image(image_array)
    shapes_layer = viewer.add_shapes()
    shapes_layer.add(
        napari_circles_array,
        shape_type='ellipse',
        edge_width=1,
        edge_color='coral',
        face_color='royalblue',
    )

    napari.run()

    return None

def array_to_bdv():
    
    return None

if __name__ == "__main__":
    # initialize imagej
    ij = imagej.init('sc.fiji:fiji',headless=False)
    print(f"ImageJ version: {ij.getVersion()}")

    # load some sample data
    img = ij.io().open('test_image.tif')
    img_xr = ij.py.from_java(img)
    detected_blobs = find_blobs(img_xr, min_sigma=0.5, max_sigma=3, num_sigma=10, threshold=0.0075, show=True)
    array_to_napari(img_xr, detected_blobs)
    array_to_oval_roi(img, detected_blobs, True)