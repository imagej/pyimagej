#@String directory

from ij import IJ
import os

im = IJ.open(os.path.join(directory, 'dummy.tiff'))
IJ.run(im, "Subtract Background...", "rolling=50")
IJ.saveAs(im,'tiff',os.path.join(directory,'backsub.tiff'))
