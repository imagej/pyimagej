# PyImageJ Beginner Activities

When suggesting PyImageJ activities for beginners, focus on foundational concepts and building confidence. Use the explain→demonstrate→challenge pattern consistently.

## Key Learning Goals
- Understanding what PyImageJ is and how it connects Python to ImageJ
- Basic initialization and setup
- Loading and displaying simple images
- Understanding Python-Java data conversion fundamentals
- Understanding different image types (Dataset vs ImagePlus)
- Basic image operations and conversions

## Suggested Activity Topics

### First Steps with PyImageJ
**Concept**: Basic PyImageJ usage
**Activities**:
- Explore available methods of ij gateway
- Check PyImageJ version and available memory

### Loading Your First Images
**Concept**: Basic image I/O operations
**Activities**:
- Load images from local files using `ij.io().open()`
- Load images from web URLs
- Display images using `ij.py.show()`

### Translating Data Between Python and Java
**Concept**: Understanding how PyImageJ bridges Python and Java data
**Activities**:
- Understand the Python-Java conversion concept
- Practice with `ij.py.to_java()` and `ij.py.from_java()` using basic examples
- Convert simple variables to Java and back

### Understanding Image Types
**Concept**: Applying conversion knowledge to common image data types
**Activities**:
- Understand when ImageJ2 (Dataset) vs ImageJ1 (ImagePlus) is needed
- Compare `ij.io().open()` vs `ij.IJ.openImage()` return types
- Convert between numpy arrays and ImageJ formats (Dataset, ImagePlus)

### Basic Image Information
**Concept**: Inspecting image properties and metadata
**Activities**:
- Get image dimensions, pixel types, and calibration
- Print basic image statistics
- Explore image metadata

### Simple Image Processing
**Concept**: Basic image operations without complex algorithms
**Activities**:
- Run basic ImageJ processes (blur, sharpen, threshold)
- Adjust brightness and contrast
- Crop and resize images
- Save processed images

## Activity Delivery Guidelines

**For each activity, follow this pattern:**

1. **Explain** (Markdown cell): Provide clear context about what the user will learn and why it's important
2. **Demonstrate** (Code cell): Show a complete, working example with detailed comments
3. **Challenge** (Code cell): Provide a partially complete code template for the user to fill in

**Example structure:**
```markdown
### Activity: Loading Your First Image

We'll learn how to load an image using PyImageJ. This is the foundation of all image analysis workflows.

Image loading in PyImageJ uses the `ij.io().open()` method, which can handle many file formats and even web URLs.
```

```python
# Demonstration: Load and display an image
import imagej

# Initialize PyImageJ (assuming setup is already done)
# Load a sample image from the web
image = ij.io().open('https://imagej.net/images/blobs.gif')

# Display the image
ij.py.show(image, cmap='viridis')

# Check image properties
print(f"Image type: {type(image)}")
print(f"Image shape: {ij.py.from_java(image).shape}")
```

```python
# Challenge: Load and analyze your own image
# TODO: Load an image from this URL: 'https://imagej.net/images/boats.gif'
image = # Your code here

# TODO: Display the image with a 'plasma' colormap
# Your code here

# TODO: Print the image dimensions and data type
# Your code here

# TODO: Convert to numpy array and print min/max pixel values
array = # Your code here
print(f"Min value: {array.min()}, Max value: {array.max()}")
```

## Beginner-Friendly Error Handling
When beginners encounter errors, guide them through common issues:
- Java memory errors → suggest increasing heap size
- Import errors → verify PyImageJ installation
- Display issues in Colab → explain headless mode and `ij.py.show()`
- File not found → check file paths and URLs

## Building Confidence
- Start with very simple, guaranteed-to-work examples
- Use familiar sample images (ImageJ's built-in samples)
- Celebrate small wins and explain why each step is important
- Connect each activity to real research applications
