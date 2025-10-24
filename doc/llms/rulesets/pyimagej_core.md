# PyImageJ Core API Rules

Your context contains two essential pieces of text regarding the code that you write. The starts and stops of these API rules are clearly delimited to help you find them. It is very important that you remember these guidelines.

1. **API rules and patterns** - The basic PyImageJ syntax, methods, and best practices you must follow
2. **Runtime environment** - The user has indicated the environment where they want to run their code. It **may or may not** be the same environment they are communicating with you! They may be writing code that **cannot** run in your environment, but your role is to help them write effective and accurate code in that environment.

**When Generating Code:**
- ALWAYS follow the API patterns and syntax specified in these rules
- Apply these rules consistently across all code examples, demonstrations, and challenges
- If you're unsure about syntax, refer back to the patterns shown in these rules

## SAMPLE DATA RULES (CRITICAL)
**NEVER guess or fabricate sample image URLs.** If you don't know the specific image the user wants to open:
- Use a placeholder comment like `# TODO: Replace with your image path`
- Direct users to official sample data sources (in order of preference):
  1. https://samples.fiji.sc/ - Classic ImageJ/Fiji sample images
  2. https://scif.io/images/index.html - Additional file formats and test images
  3. https://www.ebi.ac.uk/bioimage-archive/ - Biological imaging datasets
  4. https://cellprofiler.org/examples/ - CellProfiler example datasets
  5. https://www.allencell.org/ - Allen Cell catalog
  6. https://www.cellimagelibrary.org/home - Cell Image Library datasets

**Example of correct placeholder usage:**
```python
# TODO: Replace with your image URL or local path
# For sample images, see: https://samples.fiji.sc/
image_path = "https://samples.fiji.sc/blobs.png"  # Example
dataset = ij.io().open(image_path)
```

## DATA CONVERSION FUNDAMENTALS
- Java → Python: ij.py.from_java(java_obj)
- Python → Java: ij.py.to_java(python_obj)
- Dataset → ImagePlus: ij.py.to_imageplus(dataset)
- ImagePlus → Dataset: ij.py.to_dataset(imageplus)
- Any → Dataset: ij.py.to_dataset(data)
- Any → Img: ij.py.to_img(data)
- Any → xarray: ij.py.to_xarray(data)
- RAI → numpy: ij.py.rai_to_numpy(rai, numpy_array)

## DUAL API AWARENESS
- ImageJ2 (preferred): Dataset, ImgPlus, ij.io()
- ImageJ 1.x (legacy): ImagePlus, ij.IJ, ij.WindowManager

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
- Dataset:  ij.io().open(path)
- ImagePlus: ij.IJ.openImage(url_or_path)
- BioFormats (opens an ImagePlus[]):
```python
import scyjava as sj

# get the BioFormats Importer
BF = sj.jimport('loci.plugins.BF')
options = sj.jimport('loci.plugins.in.ImporterOptions')() # import and initialize ImporterOptions
options.setOpenAllSeries(True)
options.setVirtual(True)
options.setId("/path/to/series.ext")

# open the series
imps = BF.openImagePlus(options)
```

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
- run_macro: Run an ImageJ macro
- run_script: Run a SciJava (ImageJ2/Fiji) script
- run_plugin: Run ImageJ 1.x or ImageJ2/Fiji plugins/commands
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

## IMAGEJ OPS
- ij.op().help(): provide information on requested Op
- ij.op().help("op.name"): get help for specific op
- blurred = ij.op().filter().gauss(image, sigma=2.0)
- threshold = ij.op().threshold().otsu(image)
- result = ij.op().math().add(img1, img2)
- output = ij.op().run("op.name", input1, input2, ...)  # Generic op execution
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

## CORE ImageJ 1.x OBJECT ACCESS
- Check if active: ij.legacy.isActive()
- ij.IJ: Access to ImageJ 1.x IJ class
- ij.ResultsTable: Access to ImageJ 1.x ResultsTable (converts to pandas DataFrame)
- ij.RoiManager: Access to ImageJ 1.x ROI Manager
- ij.WindowManager: Access to ImageJ 1.x Window Manager

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

## WORKING WITH VIEWS (ImgLib2)
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

## WORKING WITH RESULTS TABLES
ImageJ 1.x ResultsTable automatically converts to pandas DataFrame:
```python
# Get ResultsTable from ImageJ
rt = ij.ResultsTable.getResultsTable()

# Automatically converts to pandas DataFrame
df = ij.py.from_java(rt)
print(df.head())
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
- Large images: Use lazy loading with ij.io().open()
- Clear variables: del large_image_variable
- Monitor usage in resource-constrained environments

## IMAGE DISPLAY & COLORMAPS
- Default: ij.py.show(image) # Uses image's natural appearance
- For scientific visualization: ij.py.show(image, cmap='viridis')
- For grayscale images: ij.py.show(image, cmap='gray') # Preserves original appearance
- For medical/microscopy: ij.py.show(image, cmap='gray') # Often most appropriate
- For fluorescence: ij.py.show(image, cmap='green') or cmap='red'
- Common colormaps: 'gray', 'viridis', 'plasma', 'inferno', 'magma', 'jet'
- RULE: If image looks unnatural, try cmap='gray' first

## ERROR PATTERNS
- "Operating in headless mode" warnings are normal
- Import errors: Check JAVA_HOME and Java version (8+ required)
- Memory errors: Restart JVM or use smaller images
