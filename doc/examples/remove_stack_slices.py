import imagej
import imagej.stack as ijs
import xarray as xr

# start imagej
print("Starting ImageJ...")
ij = imagej.init(headless=False)
print(f"ImageJ Version: {ij.getVersion()}")

# get xarray from dataset
img = ij.io().open('/home/eevans/Downloads/test_stack.tif')
img_xr = ij.pyfrom_java(img)

# extract first channel
channel_1 = ijs.extract_channel(img_xr, 1)