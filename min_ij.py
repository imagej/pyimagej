import imagej
import numpy as np
import jpype
import jpype.imports
import xarray as xr

# start ij
ij = imagej.init()

# load classes
dataset = jpype.JClass('net.imagej.Dataset')
rai = jpype.JClass('net.imglib2.RandomAccessibleInterval')

# load image and convert to array
img = ij.io().open('https://samples.fiji.sc/new-lenna.jpg')
data = ij.convert().convert(img, dataset)
print("DATATATATA: {}".format(type(data)))

# replicate _dataset_to_xarray
attrs = ij.py.from_java(data.getProperties())
print("****************")
print("[DEBUG] attrs: {}".format(attrs))

print("****************")
axes = [(jpype.JObject(data.axis(idx), jpype.JClass('net.imagej.axis.CalibratedAxis'))) for idx in range(data.numDimensions())]
print("[DEBUG] axes: {}".format(axes))

print("****************")
dims = [ij.py._ijdim_to_pydim(axes[idx].type().getLabel()) for idx in range(len(axes))]
print("[DEBUG] dims: {}".format(dims))

print("****************")
values = ij.py.rai_to_numpy(data)
print("[DEBUG] values type: {}".format(type(values)))
print("[DEBUG] values falgs:\n{}".format(values.flags))

print("****************")
coords = ij.py._get_axes_coords(axes, dims, np.shape(np.transpose(values)))
print("[DEBUG] coords type: {}".format(type(coords)))

print("****************")
if dims[len(dims)-1].lower() in ['c', 'channel']:
    xarr_dims = ij.py._invert_except_last_element(dims)
    print("[DEBUG] value type: {}".format(type(values)))
    print("[DEBUG] value flags before moveaxis:\n{}".format(values.flags))
    values = np.moveaxis(values, 0, -1)
    print("[DEBUG] value type: {}".format(type(values)))
    print("[DEBUG] value flags after moveaxis:\n{}".format(values.flags))
else:
    xarr_dims = list(reversed(dims))

xarr = xr.DataArray(values, dims=xarr_dims, coords=coords, attrs=attrs)

print("****************")
data_8bit = xarr.astype(int)
print("[DEBUG] data_8bit type: {}".format(type(data_8bit)))
print("[DEBUG] data_8bit shape: {}".format(np.shape(data_8bit.data)))
print(data_8bit)

print("****************")
data_rgb = np.moveaxis(data_8bit.data, 0, -1)
print("[DEBUG] data_rgb type: {}".format(type(data_rgb)))
print("[DEBUG] data_rgb shape: {}".format(np.shape(data_rgb.data)))
print(data_rgb)

print("****************")
data_rgb_rs = np.reshape(data_rgb, (1279, 853, 3), order='C')
print("[DEBUG] data_rgb_rs type: {}".format(type(data_rgb_rs)))
print("[DEBUG] data_rgb_rs shape: {}".format(np.shape(data_rgb_rs)))
print(data_rgb_rs)

print("****************")
# convert image directly to numpy, no xarray
np_data = ij.convert().convert(data, rai)
np_data = ij.py.rai_to_numpy(np_data)
np_data = np_data.astype(int)
np_data = np.moveaxis(np_data, 0, -1)
#ij.py.show(data_8bit)
ij.py.show(data_8bit.data)


