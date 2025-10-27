# ImageJ 1.x API Supplement
ImageJ 1.x presents a substantial API for working with ImagePlus

## CORE ImageJ 1.x OBJECT ACCESS
- Check if active: `ij.legacy.isActive()`
- `ij.IJ`: Access to ImageJ 1.x IJ class (static utility methods)
- `ij.ResultsTable`: Access to ImageJ 1.x ResultsTable (auto-converts to pandas)
- `ij.RoiManager`: Access to ImageJ 1.x ROI Manager
- `ij.WindowManager`: Access to ImageJ 1.x Window Manager

## CRITICAL IMAGEJ 1.x INDEXING

⚠️ **ImageJ 1.x uses 1-BASED indexing** for stacks (slices, channels, frames)
- Slice 1 is the first slice (NOT slice 0)
- Python is 0-based, ImageJ 1.x is 1-based - be careful!

## THRESHOLDING

ImageJ 1.x provides automatic and manual thresholding via `ij.IJ` methods.

```python
# Auto threshold (applies to ImagePlus, modifies display LUT)
ij.IJ.setAutoThreshold(img, "Default dark")  # Dark background
ij.IJ.setAutoThreshold(img, "Li")            # Bright background (default)
ij.IJ.setAutoThreshold(img, "Otsu dark")     # Otsu with dark background

# Available methods: Default, Huang, Intermodes, IsoData, Li, 
#                    MaxEntropy, Mean, MinError, Minimum, Moments,
#                    Otsu, Percentile, RenyiEntropy, Shanbhag,
#                    Triangle, Yen

# Reset threshold
ij.IJ.resetThreshold(img)

# Manual threshold (min and max pixel values)
img.getProcessor().setThreshold(lower, upper, ij.ImagePlus.RED_LUT)
# LUT options: NO_LUT_UPDATE, RED_LUT, BLACK_AND_WHITE_LUT, OVER_UNDER_LUT

# Get threshold values
ip = img.getProcessor()
lower = ip.getMinThreshold()
upper = ip.getMaxThreshold()

# Check if thresholded
if ip.getMinThreshold() != ij.ImagePlus.NO_THRESHOLD:
    print(f"Image is thresholded: {lower} - {upper}")

# Convert to binary mask (creates new 8-bit image)
# ⚠️ Make sure image is thresholded first!
ij.IJ.run(img, "Convert to Mask", "")  # In place, destructive!

# Or threshold and convert in one step
binary = img.duplicate()  # Make copy first
ij.IJ.setAutoThreshold(binary, "Li dark")
ij.IJ.run(binary, "Convert to Mask", "")

# Stack thresholding (all slices)
ij.IJ.run(img, "Auto Threshold", "method=Li white stack")
# Options: white (bright background) or dark (dark background)
#          stack (apply to all slices)
```

### Thresholding Workflow Example
```python
# Load image
img = ij.IJ.openImage("https://imagej.net/images/blobs.gif")

# Apply auto threshold
ij.IJ.setAutoThreshold(img, "Default dark")

# Get threshold values that were chosen
ip = img.getProcessor()
lower = ip.getMinThreshold()
upper = ip.getMaxThreshold()
print(f"Threshold range: {lower} - {upper}")

# Create binary mask
binary = img.duplicate()
binary.setTitle("Binary Mask")
ij.IJ.setAutoThreshold(binary, "Default dark")
ij.IJ.run(binary, "Convert to Mask", "")

# Now binary is a true binary image (0 or 255 pixels)
# Can use for measurements, particle analysis, etc.
```


## IMAGEPLUS STACK NAVIGATION

```python
# Stack/hyperstack position (1-indexed!)
img.setSlice(5)  # Navigate to slice 5 (NOT slice 4!)
current = img.getCurrentSlice()  # Returns 1-based index

# Hyperstack navigation (all 1-indexed)
img.setPosition(channel=1, slice=5, frame=3)  
channel, slice_num, frame = img.getPosition()  # All 1-based

# Get processor for current slice
ip = img.getProcessor()  # 2D pixel data for current position
```

## ROI MANAGER WORKFLOWS

```python
# Get ROI Manager (works in Colab virtual display)
from scyjava import jimport
RoiManager = jimport('ij.plugin.frame.RoiManager')
rm = RoiManager.getRoiManager()

# Add ROIs
roi = ij.gui.Roi(50, 50, 100, 100)
rm.addRoi(roi)

# Get ROIs (0-indexed, unlike slices!)
count = rm.getCount()
roi = rm.getRoi(index)  # 0-indexed
all_rois = rm.getRoisAsArray()

# Measure all ROIs (populates ResultsTable)
rm.runCommand(img, "Measure")
rt = ij.ResultsTable.getResultsTable()
df = ij.py.from_java(rt)  # Auto-converts to pandas

# Save/Load ROI sets
rm.runCommand("Save", "/path/to/rois.zip")
rm.runCommand("Open", "/path/to/rois.zip")
```

## CREATING ROIS PROGRAMMATICALLY

```python
from scyjava import jimport

# Basic shapes (already in core, but coordinates matter)
roi = ij.gui.Roi(x, y, width, height)  # Rectangle
oval = ij.gui.OvalRoi(x, y, width, height)
line = ij.gui.Line(x1, y1, x2, y2)

# Point ROIs
point = ij.gui.PointRoi(x, y)  # Single point
# Multiple points
x_coords = [10, 20, 30]
y_coords = [15, 25, 35]
points = ij.gui.PointRoi(x_coords, y_coords, len(x_coords))

# Polygon/Polyline ROIs
PolygonRoi = jimport('ij.gui.PolygonRoi')
Roi = jimport('ij.gui.Roi')
x_points = [10, 50, 50, 10]
y_points = [10, 10, 50, 50]
polygon = PolygonRoi(x_points, y_points, len(x_points), Roi.POLYGON)
polyline = PolygonRoi(x_points, y_points, len(x_points), Roi.POLYLINE)
freehand = PolygonRoi(x_points, y_points, len(x_points), Roi.FREEROI)

# ROI properties
roi.setName("My ROI")
roi.setStrokeColor(jimport('java.awt.Color').RED)
roi.setStrokeWidth(2.0)
roi.setPosition(channel, slice_num, frame)  # For hyperstacks (1-indexed)

# Measure single ROI
img.setRoi(roi)
stats = roi.getStatistics()  # Returns ImageStatistics
area = stats.area
mean = stats.mean
```

## IMAGEPROCESSOR PIXEL OPERATIONS

Already covered: Conversion basics

```python
# Get processor for current slice
ip = img.getProcessor()

# Direct pixel access (0-indexed coordinates, unlike slices!)
value = ip.getPixel(x, y)  # Returns int
ip.putPixel(x, y, value)

# Type-safe pixel access (works for all image types)
value = ip.getPixelValue(x, y)  # Returns float
ip.putPixelValue(x, y, value)

# Get pixel array (1D array, row-major order)
pixels = ip.getPixels()  # byte[], short[], float[], or int[] for RGB

# Statistical measurements on ROI
ip.setRoi(roi)
stats = ip.getStatistics()
mean = stats.mean
stdDev = stats.stdDev

# Common in-place operations (⚠️ DESTRUCTIVE!)
ip.smooth()  # 3x3 smoothing
ip.sharpen()  # Unsharp mask
ip.findEdges()  # Edge detection
ip.invert()  # Invert intensities

# Math operations (in place)
ip.add(10)  # Add constant
ip.multiply(2.0)  # Multiply
ip.gamma(0.5)  # Gamma correction
ip.abs()  # Absolute value
ip.sqr()  # Square
ip.sqrt()  # Square root
ip.log()  # Natural log

# Operations that return NEW processor
new_ip = ip.resize(new_width, new_height)
ip.setRoi(x, y, width, height)
cropped_ip = ip.crop()
copy_ip = ip.duplicate()
```

## IMAGESTACK OPERATIONS

```python
# Get stack (1-indexed operations!)
stack = img.getStack()
size = stack.size()

# Access slices (1-indexed!)
ip = stack.getProcessor(slice_num)  # slice_num from 1 to size

# Slice labels (1-indexed!)
label = stack.getSliceLabel(slice_num)
stack.setSliceLabel("Label", slice_num)

# Add/delete slices (1-indexed!)
stack.addSlice("Label", processor)  # Adds to end
stack.addSlice("Label", processor, index)  # Insert at index (1-indexed)
stack.deleteSlice(slice_num)

# Create new stack
from scyjava import jimport
ImageStack = jimport('ij.ImageStack')
new_stack = ImageStack(width, height)
new_stack.addSlice("Slice 1", processor1)
new_img = ij.ImagePlus("Stack Title", new_stack)
```

## CALIBRATION ACCESS

```python
# Get/set calibration
cal = img.getCalibration()
pixel_width = cal.pixelWidth
pixel_height = cal.pixelHeight  
pixel_depth = cal.pixelDepth
unit = cal.getUnit()

# Modify calibration
cal.pixelWidth = 0.5  # microns per pixel
cal.pixelHeight = 0.5
cal.setUnit("micron")
img.setCalibration(cal)

# Convert pixel coordinates to calibrated units
x_cal = cal.getX(x_pixel)
y_cal = cal.getY(y_pixel)
```

## PRACTICAL WORKFLOWS

### Stack Processing (Slice-by-Slice)
```python
# Process each slice (remember 1-indexing!)
img = ij.IJ.openImage("path/to/stack.tif")
stack = img.getStack()

for i in range(1, stack.size() + 1):  # 1 to size (inclusive)
    ip = stack.getProcessor(i)
    # Process slice (in-place)
    ip.smooth()
    ip.gamma(1.2)

img.updateAndDraw()  # Update display if needed
```

### ROI Manager Measurement Pipeline
```python
# Complete measurement workflow
img = ij.IJ.openImage("https://imagej.net/images/blobs.gif")

# Create/get ROI Manager
from scyjava import jimport
RoiManager = jimport('ij.plugin.frame.RoiManager')
rm = RoiManager.getRoiManager()

# Add ROIs (your segmentation logic here)
for i in range(5):
    roi = ij.gui.Roi(i*50, i*50, 40, 40)
    rm.addRoi(roi)

# Measure all
rm.runCommand(img, "Measure")

# Get results as pandas DataFrame
rt = ij.ResultsTable.getResultsTable()
df = ij.py.from_java(rt)

# Now use pandas for analysis
mean_area = df['Area'].mean()
```

## KEY DIFFERENCES FROM IMAGEJ2/OPS

- **1-based indexing**: Slices, channels, frames in ImageJ 1.x are 1-indexed (vs 0-based in ImgLib2)
- **In-place operations**: Most ImageProcessor methods modify pixels destructively (vs Ops creating new images)
- **Mutable stacks**: ImageStack operations modify in place (vs immutable Views)
- **ROI Manager**: 0-indexed for ROI access, but ROI positions are 1-indexed for stacks
- **ImagePlus vs Dataset**: ImagePlus has direct pixel access; Dataset requires converters
- **Synchronous**: All operations complete immediately (vs lazy Views in ImgLib2)