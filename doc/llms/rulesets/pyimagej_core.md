# PyImageJ Core API Rules

Your context contains two essential pieces of text regarding the code that you write. The starts and stops of these API rules are clearly delimited to help you find them. It is very important that you remember these guidelines.

1. **API rules and patterns** - The basic PyImageJ syntax, methods, and best practices you must follow
2. **Runtime environment** - The user has indicated the environment where they want to run their code. It **may or may not** be the same environment they are communicating with you! They may be writing code that **cannot** run in your environment, but your role is to help them write effective and accurate code in that environment.

**When Generating Code:**
- ALWAYS follow the API patterns and syntax specified in these rules
- Apply these rules consistently across all code examples, demonstrations, and challenges
- If you're unsure about syntax, refer back to the patterns shown in these rules

## DATA CONVERSION FUNDAMENTALS
- Java → Python: ij.py.from_java(java_obj)
- Python → Java: ij.py.to_java(python_obj)
- Dataset → ImagePlus: ij.py.to_imageplus(dataset)
- ImagePlus → Dataset: ij.py.to_dataset(dataset)

## DUAL API AWARENESS
- ImageJ2 (preferred): Dataset, ImgPlus, ij.io()
- ImageJ 1.x (legacy): ImagePlus, ij.IJ, ij.WindowManager

## WORKING WITH JAVA CLASSES
- Use `scyjava.jimport`

```python
from scyjava import jimport
# import HyperSphereShape and create radius of 5
HyperSphereShape = jimport('net.imglib2.algorithm.neighborhood.HyperSphereShape')
radius = HyperSphereShape(5)
```

## WORKING WITH IMAGES
- Dataset:  ij.io().open
- ImagePlus: ij.IJ.openImage
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

## USEFUL ImageJ2 SERVICES
- ij.op(): image processing operations

## IMAGEJ OPS
- ij.op().help(): provide information on requested Op
- blurred = ij.op().filter().gauss(image, sigma=2.0)
- threshold = ij.op().threshold().otsu(image)
- result = ij.op().math().add(img1, img2)

## CORE ImageJ 1.x OBJECT ACCESS
- Check if active: ij.legacy.isActive()
- ij.IJ
- ij.ResultsTable
- ij.RoiManager
- ij.WindowManager

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
