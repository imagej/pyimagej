# PyImageJ Core API Rules

## DATA CONVERSION FUNDAMENTALS
- Java → Python: ij.py.from_java(java_obj)
- Python → Java: ij.py.to_java(python_obj)
- Dataset → ImagePlus: ij.py.to_imageplus(dataset)
- ImagePlus → Dataset: ij.py.to_dataset(imageplus)
- Any → Dataset: ij.py.to_dataset(data)
- Any → Img: ij.py.to_img(data)
- Any → xarray: ij.py.to_xarray(data)
- RAI → numpy: ij.py.rai_to_numpy(rai, numpy_array)

## WORKING WITH JAVA CLASSES
- Use `scyjava.jimport` or `from scyjava import jimport`

```python
from scyjava import jimport

# Import and use Java classes
HyperSphereShape = jimport('net.imglib2.algorithm.neighborhood.HyperSphereShape')
radius = HyperSphereShape(5)

Views = jimport('net.imglib2.view.Views')
iterable_view = Views.iterable(image)

# Import Java utilities
ArrayList = jimport('java.util.ArrayList')
list_obj = ArrayList()

# Access static methods and fields
Axes = jimport('net.imagej.axis.Axes')
x_axis = Axes.X
custom_axis = Axes.get("custom_name")
```

## WORKING WITH IMAGES
- Dataset:  `ij.io().open(url_or_path)`. Best if using mostly ImageJ2 API (like ij.op())
- ImagePlus: `ij.IJ.openImage(url_or_path)`. Best if using mostly ImageJ 1.x API (like ij.IJ)

## SAMPLE DATA RULES (CRITICAL)
When generating code to open images:
- ❌ NEVER guess or fabricate sample image URLs
- ✅ If the user didn't specify an image, use a descriptive placeholder with a comment

Example placeholder use:
```python
img = ij.IJ.openImage("path/to/your/image.tif")  # Replace with your image path or URL
```

## DUPLICATING IMAGES BEFORE MODIFICATION
⚠️ **CRITICAL**: Many ImageJ operations modify images in-place. Always duplicate before one or more destructive operations!

```python
# ✅ CORRECT: Duplicate before modifying
original = ij.IJ.openImage("https://samples.fiji.sc/blobs.png")
copy = original.duplicate()  # ImagePlus
ij.py.run_plugin("Gaussian Blur...", {"sigma": 3.0}, imp=copy)
```
Now 'original' is unchanged, 'copy' is blurred

```python
# ✅ CORRECT: Duplicate Dataset
dataset = ij.io().open("path/to/image.tif")
copy = dataset.duplicate()
# Process 'copy' while preserving 'original'
```

```python
# ❌ WRONG: Modifying without duplicating
img = ij.IJ.openImage("https://samples.fiji.sc/blobs.png")
ij.py.run_plugin("Gaussian Blur...", {"sigma": 3.0}, imp=img)
# 'img' is now permanently blurred - original data lost!
```

**When to duplicate:**
- ✅ Before running plugins that modify pixel values
- ✅ Before thresholding, filtering, or mathematical operations
- ✅ When you need to compare before/after results
- ✅ When processing in a loop where original data is reused

**When duplication is NOT needed:**
- Operations that create new images (e.g., `ij.op().filter().gauss()` returns new image)
- Read-only operations (e.g., measurements, analysis)
- Final processing steps where original is no longer needed

## RUNNING MACROS WITH ARGUMENTS
```python
macro = """
#@ String name
#@ int age
#@output String message
message = name + " is " + age + " years old."
"""
args = {
    "name": "Alice",
    "age": 30
}
result = ij.py.run_macro(macro, args)
print(result["message"])  # Access output
```

## RUNNING SCRIPTS WITH ARGUMENTS
```python
script = """
#@ String name
#@ int age
#@output String message
message = name + " is " + age + " years old."
"""
args = {
    "name": "Bob",
    "age": 25
}
result = ij.py.run_script("python", script, args)  # or "groovy", "javascript", etc.
print(result["message"])  # Dict-like access to outputs
```

## RUNNING PLUGINS WITH ARGUMENTS
```python
# ImageJ 1.x style plugins
plugin = "Gaussian Blur..."
args = {
    "sigma": 3.0,
    "scaled": True  # Boolean arguments work
}
ij.py.run_plugin(plugin, args)

# Can optionally pass specific ImagePlus
ij.py.run_plugin(plugin, args, imp=my_imageplus)
```

## RGB AND AXIS CONVENTIONS:
- numpy arrays don't have metadata, but xarray does
- xarray order is different from ImageJ's

```python
# load 4D test data
dataset = ij.io().open('sample-data/test_timeseries.tif')

# get xarray
xarr = ij.py.from_java(dataset)

# print out shape and dimensions
print(f"dataset dims, shape: {dataset.dims}, {dataset.shape}")
print(f"xarr dims, shape: {xarr.dims}, {xarr.shape}")

dataset dims, shape: ('X', 'Y', 'Channel', 'Time'), (250, 250, 3, 15)
xarr dims, shape: ('t', 'row', 'col', 'ch'), (15, 250, 250, 3)
```

## ADDITIONAL IMPORTANT ij.py METHODS
- initialize_numpy_image: Create new numpy image with shape of input
- sync_image: Synchronize ImageJ 1.x and ImageJ2 data structures
- active_dataset: Get active image as an ImageJ2 Dataset
- active_xarray: Get copy of active image as an xarray.DataArray
- active_imageplus: Get the active `ImagePlus` from current window
- to_img: Convert data to ImgLib2 Img
- to_xarray: Convert data to xarray.DataArray
- jargs: Convert Python arguments into Java Object[] for passing to ImageJ2 run functions
- rai_to_numpy: Copy a RandomAccessibleInterval into a numpy array
- dtype: Get the numpy dtype of an image

## USEFUL ImageJ2 SERVICES
- ij.op(): image processing operations (OpService)
- ij.io(): image I/O operations
- ij.convert(): conversion service for type conversions
- ij.command(): command service for running commands
- ij.script(): script service for running scripts
- ij.module(): module service for running modules
- ij.dataset(): dataset service for creating/managing Datasets

## JAVA IMAGE PROPERTIES (added by PyImageJ)
When working with Java images, PyImageJ adds these NumPy-like properties:
- image.shape: Tuple of dimensions (like NumPy)
- image.dims: Tuple of axis labels (e.g., ('X', 'Y', 'Channel'))
- image.dim_axes: Tuple of CalibratedAxis objects (for Dataset/ImgPlus)
- image.dtype: Image data type (for RandomAccessibleInterval)
- image.ndim: Number of dimensions (for EuclideanSpace)
- image.T or image.transpose: Transposed image (for RandomAccessibleInterval)

## ARRAY SLICING ON JAVA IMAGES
RandomAccessibleInterval objects support NumPy-like slicing:
```python
# Assuming 'img' is a RandomAccessibleInterval
slice_2d = img[0, :, :]  # Get first plane
subregion = img[10:20, 30:40]  # Get subregion
stepped = img[::2, ::2]  # Subsample by 2
```

## MATHEMATICAL OPERATIONS ON JAVA IMAGES
RandomAccessibleInterval objects support element-wise operations:
```python
result = img1 + img2  # Addition
result = img1 - img2  # Subtraction
result = img1 * img2  # Multiplication
result = img1 / img2  # Division
```

## WORKING WITH VIEWS (ImageJ2)
Use Views for efficient image manipulation without copying:
```python
from scyjava import jimport
Views = jimport('net.imglib2.view.Views')

# Common View operations
hyperslice = Views.hyperSlice(img, dimension, position)
interval = Views.interval(img, min_coords, max_coords)
iterable = Views.iterable(img)  # For iteration
extended = Views.extendBorder(img)  # Extend with border values
```

```python
# Example: Using Ops with Views for efficient processing
from scyjava import jimport
Views = jimport('net.imglib2.view.Views')
HyperSphereShape = jimport('net.imglib2.algorithm.neighborhood.HyperSphereShape')

# Create output and wrap as iterable
output = ij.py.initialize_numpy_image(input_img)
java_out = Views.iterable(ij.py.to_java(output))
java_in = ij.py.to_java(input_img)

# Run morphology operation
shapes = jimport('java.util.ArrayList')()
shapes.add(HyperSphereShape(5))
ij.op().morphology().topHat(java_out, java_in, shapes)
```

## WORKING WITH RESULTS TABLES (ImageJ 1.x)
ImageJ 1.x ResultsTable automatically converts to pandas DataFrame:
```python
# Get ResultsTable from ImageJ
rt = ij.ResultsTable.getResultsTable()

# Automatically converts to pandas DataFrame
df = ij.py.from_java(rt)
print(df.head())
```

## ARRAY PROPERTIES ON JAVA IMAGES
Java images have NumPy-like properties in Colab:
```python
# Load image
dataset = ij.io().open('path/to/image.tif')

# Access properties
print(f"Shape: {dataset.shape}")
print(f"Dims: {dataset.dims}")
print(f"Dtype: {dataset.dtype}")

# These work on Dataset, ImgPlus, RandomAccessibleInterval
```

## LABELING SUPPORT
PyImageJ supports conversion between Python Labeling and Java ImgLabeling:
```python
# Python -> Java
java_labeling = ij.py.to_java(python_labeling)

# Java -> Python
python_labeling = ij.py.from_java(java_imglabeling)
```

## MEMORY MANAGEMENT
- ✅ Large images: Use lazy loading with ij.io().open()
- ✅ Clear variables: del large_image_variable
- ⚠️ Monitor usage in resource-constrained environments

## IMAGE DISPLAY & COLORMAPS
- ✅ Default: ij.py.show(image) # Uses image's natural appearance
- ✅ Works with ALL image types: Dataset, ImgPlus, ImagePlus, numpy, xarray (auto-converts most types as needed)
- ✅ For scientific visualization: ij.py.show(image, cmap='viridis')
- ✅ For grayscale images: ij.py.show(image, cmap='gray') # Preserves original appearance
- ✅ For medical/microscopy: ij.py.show(image, cmap='gray') # Often most appropriate
- ✅ For fluorescence: ij.py.show(image, cmap='green') or cmap='red'
- Common colormaps: 'gray', 'viridis', 'plasma', 'inferno', 'magma', 'jet'
- ⚠️ RULE: If image looks unnatural, try cmap='gray' first
- ❌ NEVER use image.show() - always use ij.py.show(image)

**Examples:**
```python
# ✅ CORRECT: open as ImagePlus
img = ij.IJ.openImage("https://samples.fiji.sc/blobs.png")
ij.py.show(img)  # No manual conversion needed!

# ✅ CORRECT: open as Dataset
dataset = ij.io().open("path/to/image.tif")
ij.py.show(dataset, cmap='gray')

# ❌ WRONG: Unnecessary conversion
img = ij.IJ.openImage("https://samples.fiji.sc/blobs.png")
ij.py.show(ij.py.to_dataset(img))  # Don't do this - ij.py.show auto-converts!
```

## ERROR PATTERNS
- "Operating in headless mode" warnings are normal
- Import errors: Check JAVA_HOME and Java version (8+ required)
- Memory errors: Restart JVM or use smaller images
