import imagej
import scyjava as sj
import xarray as xr
import numpy as np

from math import sqrt
from skimage.feature import blob_log


def find_blobs(image: xr.DataArray, min_sigma: float, max_sigma: float, num_sigma: int, threshold=0.1) -> np.ndarray:
    """
    Find blobs with Laplacian of Gaussian (LoG).
    """
    # detect blobs in image
    blobs = blob_log(image, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma, threshold=threshold)
    blobs[:, 2] = blobs[:, 2] * sqrt(2)

    return blobs


def process(image, oval_array: np.ndarray, add_to_roi_manager=True, multimeasure=True):
    """
    Process the blob rois.
    """
    # get ImageJ resoures
    ImagePlus = sj.jimport('ij.ImagePlus')
    OvalRoi = sj.jimport('ij.gui.OvalRoi')
    Overlay = sj.jimport('ij.gui.Overlay')
    ov = Overlay()

    # convert image to imp
    imp = ij.convert().convert(image, ImagePlus)

    if add_to_roi_manager:
        rm = ij.RoiManager().getRoiManager()

    for i in range(len(oval_array)):
        values = oval_array[i].tolist()
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

    if rm != None and multimeasure:
        rm.runCommand(imp, "Measure")
        return ij.ResultsTable.getResultsTable()
    
    return None


def get_dataframe(table):
    """
    Convert results table to pandas Dataframe.
    """
    Table = sj.jimport('org.scijava.table.Table')
    sci_table = ij.convert().convert(table, Table)

    return ij.py.from_java(sci_table)


if __name__ == "__main__":
    # initialize imagej
    ij = imagej.init(mode='interactive')
    print(f"ImageJ version: {ij.getVersion()}")

    # load sample data and run blob detection
    img = ij.io().open('../sample-data/test_image.tif')
    img_xr = ij.py.from_java(img)
    detected_blobs = find_blobs(img_xr, min_sigma=0.5, max_sigma=3, num_sigma=10, threshold=0.0075)
    results_table = process(img, detected_blobs)
    df = get_dataframe(results_table)
    print(f"Output: \n{df}")