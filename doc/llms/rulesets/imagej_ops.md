# ImageJ Ops API Guide for PyImageJ

ImageJ Ops provides a library of image processing operations accessed through `ij.op()`. This guide focuses on practical usage patterns for image analysis workflows.

## OP TYPES

Ops are classified by how they handle inputs and outputs:

### Function
- **Creates new output** object and returns it
- Input is **not modified**
- Call: `result = op.calculate(input)` or `result = ij.op().namespace().opname(input)`
- Example: `blurred = ij.op().filter().gauss(image, sigma=2.0)` returns NEW image

### Computer  
- **Writes to pre-allocated output** (you provide the output object)
- Input is **not modified**, output must be **different object** than input
- Call: `op.compute(input, output)` or `ij.op().namespace().opname(output, input, ...)`
- Example: `ij.op().filter().gauss(output, image, sigma=2.0)` writes to existing output

### Inplace
- **Mutates the input** directly (in-place modification)
- Input and output **must be same object**
- Call: `op.mutate(arg)` or `ij.op().namespace().opname(arg, arg, ...)`
- Example: `ij.op().math().add(image, image, 10.0)` modifies image in-place

### Hybrid
- **Flexible**: can work as Function, Computer, or Inplace
- Behavior depends on arguments provided:
  - `result = op(input)` → acts as Function (creates new output)
  - `op(output, input)` → acts as Computer (fills output)
  - `op(arg, arg)` → acts as Inplace (mutates arg)
- Most common type for general-purpose ops
- Example: `ij.op().filter().gauss()` works in all three modes

**Practical Usage:**
```python
# Function mode: creates new image (most common)
blurred = ij.op().filter().gauss(image, sigma=2.0)

# Computer mode: fill pre-allocated output
output = ij.py.initialize_numpy_image(image)
ij.op().filter().gauss(output, image, sigma=2.0)

# Inplace mode: modify image directly (rare for filters, common for math ops)
ij.op().math().add(image, image, 10.0)  # Add 10 to image in-place
```

## DISCOVERING OPS

```python
# Get help on available ops
ij.op().help()  # List all op namespaces

# Get help for specific namespace
ij.op().help("filter")
ij.op().help("threshold")
ij.op().help("math")

# Get help for specific op
ij.op().help("filter.gauss")
ij.op().help("threshold.otsu")
```

## FILTER OPERATIONS

### Gaussian Filtering
```python
# Single sigma (isotropic)
blurred = ij.op().filter().gauss(image, sigma=2.0)

# Per-dimension sigmas (anisotropic)
sigmas = [2.0, 2.0, 1.0]  # x, y, z
blurred = ij.op().filter().gauss(image, sigmas)

# With output image
output = ij.py.initialize_numpy_image(image)
ij.op().filter().gauss(output, image, sigma=2.0)
```

### Other Filters
```python
# Median filter
from scyjava import jimport
RectangleShape = jimport('net.imglib2.algorithm.neighborhood.RectangleShape')
shape = RectangleShape(radius=2, skip_center=False)
filtered = ij.op().filter().median(image, shape)

# Bilateral filter (edge-preserving)
bilateral = ij.op().filter().bilateral(image, sigma_r=10.0, sigma_s=5.0, radius=3)

# Variance filter
variance = ij.op().filter().variance(image, shape)

# Sobel edge detection
edges = ij.op().filter().sobel(image)

# Derivative of Gaussian
deriv = ij.op().filter().derivativeGauss(image, dimension=0, sigma=2.0)

# Tubeness (for detecting tubular structures)
tubeness = ij.op().filter().tubeness(image, sigma=2.0)

# Frangi vesselness (for vessel/filament detection)
vesselness = ij.op().filter().frangi(image, scale=2.0)
```

### Convolution
```python
# Convolve with kernel
kernel = ij.op().create().img([5, 5])  # Create 5x5 kernel
# ... populate kernel ...
convolved = ij.op().filter().convolve(image, kernel)

# FFT-based convolution (faster for large kernels)
convolved = ij.op().filter().convolve(image, kernel)  # Auto-selects method
```

## THRESHOLD OPERATIONS

### Global Thresholding
```python
# Otsu thresholding (auto-threshold)
threshold_value = ij.op().threshold().otsu(image)

# Apply threshold to create binary image
binary = ij.op().threshold().apply(image, threshold_value)

# Common threshold methods (all return threshold value)
threshold_value = ij.op().threshold().huang(image)
threshold_value = ij.op().threshold().ij1(image)  # ImageJ 1.x default
threshold_value = ij.op().threshold().isoData(image)
threshold_value = ij.op().threshold().li(image)
threshold_value = ij.op().threshold().maxEntropy(image)
threshold_value = ij.op().threshold().mean(image)
threshold_value = ij.op().threshold().minError(image)
threshold_value = ij.op().threshold().minimum(image)
threshold_value = ij.op().threshold().moments(image)
threshold_value = ij.op().threshold().otsu(image)
threshold_value = ij.op().threshold().percentile(image)
threshold_value = ij.op().threshold().renyiEntropy(image)
threshold_value = ij.op().threshold().shanbhag(image)
threshold_value = ij.op().threshold().triangle(image)
threshold_value = ij.op().threshold().yen(image)
```

### Local (Adaptive) Thresholding
```python
from scyjava import jimport
RectangleShape = jimport('net.imglib2.algorithm.neighborhood.RectangleShape')
shape = RectangleShape(radius=15, skip_center=False)

# Local thresholding methods
binary = ij.op().threshold().localMean(image, shape, c=0.0)
binary = ij.op().threshold().localMedian(image, shape, c=0.0)
binary = ij.op().threshold().localMidGrey(image, shape, c=0.0)
binary = ij.op().threshold().localNiblack(image, shape, c=0.0, k=0.5)
binary = ij.op().threshold().localSauvola(image, shape, c=0.0, k=0.5, r=0.5)
binary = ij.op().threshold().localPhansalkar(image, shape, c=0.0, k=0.25, r=0.5)
binary = ij.op().threshold().localBernsen(image, shape, c=0.0, contrast_threshold=15.0)
binary = ij.op().threshold().localContrast(image, shape)

# Local versions of global methods
binary = ij.op().threshold().huang(image, shape)
binary = ij.op().threshold().ij1(image, shape)
binary = ij.op().threshold().otsu(image, shape)
# ... and others
```

## MORPHOLOGY OPERATIONS

```python
from scyjava import jimport

# Define structuring element shape
HyperSphereShape = jimport('net.imglib2.algorithm.neighborhood.HyperSphereShape')
RectangleShape = jimport('net.imglib2.algorithm.neighborhood.RectangleShape')
DiamondShape = jimport('net.imglib2.algorithm.neighborhood.DiamondShape')

# Create shapes
sphere = HyperSphereShape(radius=3)
rectangle = RectangleShape(radius=2, skip_center=False)
diamond = DiamondShape(radius=2)

# Basic morphological operations
eroded = ij.op().morphology().erode(image, sphere)
dilated = ij.op().morphology().dilate(image, sphere)
opened = ij.op().morphology().open(image, sphere)
closed = ij.op().morphology().close(image, sphere)

# Top-hat and black top-hat
# Create list of shapes for operations
ArrayList = jimport('java.util.ArrayList')
shapes = ArrayList()
shapes.add(sphere)

tophat = ij.op().morphology().topHat(image, shapes)
blacktophat = ij.op().morphology().blackTopHat(image, shapes)

# Thinning operations (skeletonization)
skeleton = ij.op().morphology().thinZhangSuen(binary_image)
skeleton = ij.op().morphology().thinGuoHall(binary_image)
skeleton = ij.op().morphology().thinHilditch(binary_image)
skeleton = ij.op().morphology().thinMorphological(binary_image)

# Fill holes in binary image
filled = ij.op().morphology().fillHoles(binary_image)
```

## MATH OPERATIONS

### Element-wise Operations
```python
# Arithmetic on images
result = ij.op().math().add(img1, img2)
result = ij.op().math().subtract(img1, img2)
result = ij.op().math().multiply(img1, img2)
result = ij.op().math().divide(img1, img2)

# With scalars
result = ij.op().math().add(image, 10.0)
result = ij.op().math().multiply(image, 2.5)

# Unary operations
result = ij.op().math().abs(image)
result = ij.op().math().sqr(image)  # Square
result = ij.op().math().sqrt(image)
result = ij.op().math().log(image)
result = ij.op().math().exp(image)
result = ij.op().math().invert(image)
result = ij.op().math().reciprocal(image)

# Trigonometric
result = ij.op().math().sin(image)
result = ij.op().math().cos(image)
result = ij.op().math().tan(image)

# Logical operations (for binary images)
result = ij.op().math().and(binary1, binary2)
result = ij.op().math().or(binary1, binary2)
result = ij.op().math().xor(binary1, binary2)
result = ij.op().math().not(binary_image)
```

## STATISTICS OPERATIONS

```python
# Basic statistics
mean_val = ij.op().stats().mean(image)
std_val = ij.op().stats().stdDev(image)
var_val = ij.op().stats().variance(image)
min_val = ij.op().stats().min(image)
max_val = ij.op().stats().max(image)
sum_val = ij.op().stats().sum(image)
size = ij.op().stats().size(image)

# Advanced statistics
median_val = ij.op().stats().median(image)
geometric_mean = ij.op().stats().geometricMean(image)
harmonic_mean = ij.op().stats().harmonicMean(image)
kurtosis = ij.op().stats().kurtosis(image)
skewness = ij.op().stats().skewness(image)

# Quantiles and percentiles
percentile_val = ij.op().stats().percentile(image, 95.0)  # 95th percentile
quantile_val = ij.op().stats().quantile(image, 0.95)  # Same as percentile

# Min/max together
from scyjava import jimport
Pair = jimport('net.imglib2.util.Pair')
minmax = ij.op().stats().minMax(image)
min_val = minmax.getA()
max_val = minmax.getB()
```

## IMAGE CREATION

```python
# Create image matching dimensions
output = ij.op().create().img(image)  # Same type and size

# Create with specific dimensions
output = ij.op().create().img([512, 512])  # 512x512

# Create with specific type
from scyjava import jimport
FloatType = jimport('net.imglib2.type.numeric.real.FloatType')
output = ij.op().create().img([512, 512], FloatType())

# Create kernel (for convolution)
kernel = ij.op().create().kernelGauss([5, 5], [1.5, 1.5])
kernel = ij.op().create().kernelLog([5, 5], [2.0, 2.0])
kernel = ij.op().create().kernelSobel()
```

## TRANSFORM OPERATIONS

```python
# FFT and inverse FFT
from scyjava import jimport
ComplexFloatType = jimport('net.imglib2.type.numeric.complex.ComplexFloatType')

fft_result = ij.op().filter().fft(image)
inverse = ij.op().filter().ifft(fft_result)

# Rotation (2D)
# Note: Rotation creates new image
rotated = ij.op().transform().rotate(image, angle_radians)

# Scale/Resize
scaled = ij.op().transform().scale(image, [2.0, 2.0])  # 2x scaling

# Subsample
subsampled = ij.op().transform().subsampleView(image, 2)  # Every 2nd pixel
```

## WORKING WITH NEIGHBORHOODS

Shapes define neighborhoods for local operations:

```python
from scyjava import jimport

# Rectangle/Box
RectangleShape = jimport('net.imglib2.algorithm.neighborhood.RectangleShape')
rect = RectangleShape(radius=3, skip_center=False)  # 7x7 box

# Sphere/Circle
HyperSphereShape = jimport('net.imglib2.algorithm.neighborhood.HyperSphereShape')
sphere = HyperSphereShape(radius=5)

# Diamond
DiamondShape = jimport('net.imglib2.algorithm.neighborhood.DiamondShape')
diamond = DiamondShape(radius=3)

# Use with local operations
median_filtered = ij.op().filter().median(image, rect)
local_threshold = ij.op().threshold().localMean(image, rect, c=0.0)
```

## USEFUL PATTERNS

### In-place vs New Image
```python
# Many ops create NEW images by default
blurred = ij.op().filter().gauss(image, sigma=2.0)  # New image

# For in-place, provide output
output = image.copy()  # Or use duplicate()
ij.op().filter().gauss(output, image, sigma=2.0)  # Modifies output
```

### Type Conversion for Ops
```python
# Some ops require specific types
# Convert if needed
from scyjava import jimport
FloatType = jimport('net.imglib2.type.numeric.real.FloatType')

# Convert to float for processing
float_img = ij.op().convert().float32(image)
processed = ij.op().filter().gauss(float_img, sigma=2.0)

# Convert back if needed
result = ij.op().convert().uint8(processed)
```

### Chaining Operations
```python
# Process pipeline
blurred = ij.op().filter().gauss(image, sigma=2.0)
threshold = ij.op().threshold().otsu(blurred)
binary = ij.op().threshold().apply(blurred, threshold)
closed = ij.op().morphology().close(binary, sphere)
```

### Views for Efficiency
```python
# Use Views to avoid copying large data
from scyjava import jimport
Views = jimport('net.imglib2.view.Views')

# Extend image for boundary handling
extended = Views.extendMirrorSingle(image)

# Process only a region
Views = jimport('net.imglib2.view.Views')
subregion = Views.interval(image, [50, 50], [150, 150])
processed = ij.op().filter().gauss(subregion, sigma=2.0)
```

## COMMON WORKFLOWS

### Segmentation Pipeline
```python
from scyjava import jimport

# 1. Smooth image
smoothed = ij.op().filter().gauss(image, sigma=2.0)

# 2. Threshold
threshold_val = ij.op().threshold().otsu(smoothed)
binary = ij.op().threshold().apply(smoothed, threshold_val)

# 3. Morphological cleanup
HyperSphereShape = jimport('net.imglib2.algorithm.neighborhood.HyperSphereShape')
shape = HyperSphereShape(radius=2)

# Remove small noise
opened = ij.op().morphology().open(binary, shape)

# Fill holes
closed = ij.op().morphology().close(opened, shape)

# 4. Label connected components
labeled = ij.op().labeling().cca(closed)
```

### Background Subtraction
```python
# Create background estimate with large Gaussian
background = ij.op().filter().gauss(image, sigma=50.0)

# Subtract background
corrected = ij.op().math().subtract(image, background)
```

### Edge Detection
```python
# Method 1: Sobel
edges = ij.op().filter().sobel(image)

# Method 2: Derivative of Gaussian
dog_x = ij.op().filter().derivativeGauss(image, dimension=0, sigma=2.0)
dog_y = ij.op().filter().derivativeGauss(image, dimension=1, sigma=2.0)

# Combine gradients
gradient_mag = ij.op().math().add(
    ij.op().math().sqr(dog_x),
    ij.op().math().sqr(dog_y)
)
gradient_mag = ij.op().math().sqrt(gradient_mag)
```

## KEY DIFFERENCES FROM IMAGEJ 1.x

- **0-based indexing**: ImgLib2 uses 0-based indexing (vs ImageJ 1.x 1-based for slices)
- **New images by default**: Most ops return new images (vs ImageJ 1.x in-place ImageProcessor operations)
- **Immutable Views**: Views provide virtual transformations without copying (vs mutable ImageStack)
- **Type-generic**: Works with any pixel type through ImgLib2 (vs separate ByteProcessor, ShortProcessor, etc.)
- **Shape objects**: Local operations use Shape objects for neighborhoods (vs fixed kernels)
- **No ROI Manager**: Use labeling/segmentation ops instead of ROI Manager workflows
