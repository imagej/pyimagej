# Coding Beginner Activities

For users new to programming, focus on Python fundamentals that are essential for effective PyImageJ usage. Emphasize practical concepts over abstract programming theory.

## Key Learning Goals
- Basic Python syntax and data types
- Working with numpy arrays (fundamental for image data)
- Understanding functions and basic control flow
- File handling and data import/export
- Basic debugging and error interpretation
- Understand error messages in plain language
- Building programming intuition through visual learning and familiar analogies
- Developing confidence with guaranteed success examples

## Suggested Activity Topics

### Python Basics for Image Analysis
**Concept**: Core Python concepts needed for scientific computing
**Activities**:
- Variables, numbers, and strings
- Lists and dictionaries for organizing data
- Understanding Python imports and packages
- Using print statements for debugging and exploration

### Working with Numbers and Arrays
**Concept**: Numpy fundamentals for image data
**Activities**:
- Creating arrays with numpy
- Understanding array shapes and dimensions
- Basic array operations (slicing, indexing)
- Mathematical operations on arrays (+, -, *, /)

### Functions and Code Organization
**Concept**: Writing reusable code for image analysis workflows
**Activities**:
- Writing simple functions with parameters
- Understanding return values
- Organizing analysis steps into functions
- Using descriptive variable and function names

### Loading and Saving Data
**Concept**: File I/O essential for image analysis
**Activities**:
- Reading files from disk and URLs
- Understanding file paths (absolute vs relative)
- Saving results to files (CSV, text)
- Working with different data formats

### Basic Control Flow
**Concept**: Making decisions and repeating operations
**Activities**:
- Using if/else for conditional processing
- Simple for loops for batch processing
- Understanding when and how to use loops with images
- Basic error handling with try/except

### Understanding Data Types
**Concept**: Different ways to represent image data
**Activities**:
- Numbers vs. strings vs. lists
- Understanding when data conversion is needed
- Working with boolean values for image masks
- Type checking and debugging type errors

## Activity Delivery Guidelines

**For beginners, emphasize:**

1. **Explain** why each concept matters for image analysis specifically
2. **Demonstrate** with very concrete, visual examples
3. **Challenge** with small, achievable tasks that build confidence

**Example structure:**
```markdown
### Activity: Your First Numpy Array

Arrays are how we represent images in Python. Think of an array like a grid of numbers, where each number represents a pixel's brightness.

This is fundamental because all image processing works by changing these numbers in clever ways.
```

```python
# Demonstration: Creating and exploring arrays
import numpy as np

# Create a simple 2D array (like a tiny grayscale image)
small_image = np.array([[100, 150, 200],
                       [120, 160, 210],
                       [110, 140, 190]])

print("Our tiny image:")
print(small_image)
print(f"Shape: {small_image.shape}")  # (3, 3) means 3 rows, 3 columns
print(f"Brightest pixel: {small_image.max()}")
print(f"Darkest pixel: {small_image.min()}")

# Create a larger random image
random_image = np.random.randint(0, 255, size=(100, 100))
print(f"Random image shape: {random_image.shape}")
```

```python
# Challenge: Create and explore your own array
# TODO: Create a 5x5 array with all values equal to 128 (medium gray)
gray_square = # Your code here

# TODO: Print the array and its shape
# Your code here

# TODO: Make the center pixel brighter (value 255)
# Hint: arrays use [row, column] indexing, center of 5x5 is [2, 2]
# Your code here

# TODO: Print the modified array
# Your code here
```