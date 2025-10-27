# Google Colab Environment Ruleset

## PRIMER NOTEBOOK SETUP
- Change bundle in `🏝️ Setup Fiji Bundle` cell if different Fiji version needed
- Change commit in `📦 Download LLM Rules` cell if rules files seem outdated
- Bundles include a JDK, Maven and Jgo caches to avoid hitting SciJava servers

## PRESUMED INITIALIZATION STATE
- ✅ `ij` variable already initialized
- ❌ Do not call imagej.init() again
- PyImageJ initialized in 'interactive' mode with Xvfb virtual display, wrapping Fiji
- Legacy mode available: check with ij.legacy.isActive()

## DISPLAY ENVIRONMENT
- Interactive mode with virtual display (Xvfb) - NOT true headless
- GUI objects can be used (RoiManager, WindowManager) but no visual display
- Can use APIs that require graphical environment
- No actual GUI windows visible to user
- ❌ Do NOT call ij.ui().showUI() - display not visible to user
- ✅ Access WindowManager.getCurrentImage() - returns ImagePlus
- ✅ Use RoiManager.getRoiManager() - functional but not visually displayed

## IMAGE DISPLAY PATTERNS
- ✅ 2D images: ij.py.show(image) with optional color map parameter
- ✅ >2D data: Use ipywidgets for interactive visualization
- ❌ NEVER: ij.ui().show() > Works but displays nothing useful
- ❌ NEVER: image.show() > Use ij.py.show(image) instead
- ⚠️ If you see `TypeError: Invalid shape (...) for image data` > Image is 3D+ ➡️ use ipywidgets pattern below

## PACKAGE INSTALLATION
- ✅ ALWAYS: %pip install package_name (preferred in notebooks)
- ✅ ALTERNATIVE: !pip install package_name (works but less integrated)
- ❌ DISCOURAGE: conda (not reliable in Colab)
- IMPORTANT: %pip ensures packages install to the correct kernel
- Common packages: numpy, matplotlib, scikit-image, pandas

## CONNECTING YOUR DATA (REPRODUCIBLE METHODS)
- ❌ AVOID: Files sidebar local uploads (runtime-specific, not reproducible)
- ⚠️ DISCOURAGE: Mounting Google Drive (user-specific, not shareable, but could reduce initialization time)
- ✅ RECOMMENDED: Host data on the web and download to runtime

**Option 1: Host on web**
- ij.io().open() and ij.IJ.openImage() should both be able to handle most URLs (but not Google Drive or GitHub)

**Option 2: Public Google Drive file with gdown**
Google Drive files need to be downloaded specially

```python
# Install gdown if needed
%pip install gdown

# Download from public Google Drive link
import gdown
file_id = "1ABC...XYZ"  # Extract from share link
gdown.download(f"https://drive.google.com/uc?id={file_id}", "data.tif", quiet=False)
```

**Option 3: Clone from GitHub repository**
GitHub is not ideal for hosting images, but also should be cloned at the repository-level.

```python
# Clone repo containing data
!git clone https://github.com/username/dataset-repo.git
# Access data
dataset = ij.io().open("dataset-repo/images/sample.tif")
```

## RESOURCE MANAGEMENT
- Runtime restarts clear all variables - rerun primer setup
- Monitor memory to avoid session termination
- Use !commands for shell, %commands for magic

## N-DIMENSIONAL DATA HANDLING
- For 3D+ data, we need to create interactive widgets instead of using `ij.py.show`
- ⚠️ Use this to solve `TypeError: Invalid shape (...) for image data` 
```python
import ipywidgets as widgets
from IPython.display import display

def show_slice(data, slice_idx=0):
    if len(data.shape) == 2:
        ij.py.show(data, cmap='gray')
    else:
        # Show specific slice
        slice_2d = data[:, :, slice_idx]
        ij.py.show(slice_2d, cmap='gray')

# Create slider for 3D data
slider = widgets.IntSlider(min=0, max=data.shape[2]-1, step=1, value=0)
widgets.interact(show_slice, data=widgets.fixed(data), slice_idx=slider)
```

## WORKING WITH LEGACY IMAGEJ IN COLAB
- Legacy ImageJ functions work in virtual display environment
- RoiManager, WindowManager accessible but not visually displayed
- Macros and plugins execute normally
- ResultsTable data can be extracted to pandas:
```python
# Run analysis that populates ResultsTable
ij.py.run_plugin("Analyze Particles...", args)

# Extract results to pandas
rt = ij.ResultsTable.getResultsTable()
df = ij.py.from_java(rt)
df.head()
```

## WINDOWMANAGER USAGE PATTERNS

**Usually you will just use the image variable**
```python
# ❌ WRONG: There is no ij.getImage method
ij.IJ.setAutoThreshold(ij.getImage(), "Li dark")

# ✅ CORRECT: pass the image as a paramter
ij.IJ.setAutoThreshold(preprocessed_image, "Li dark")
```

**Most plugins modify images IN PLACE - no WindowManager needed:**
```python
# ✅ CORRECT: Plugin modifies image in place
img = ij.IJ.openImage("https://samples.fiji.sc/blobs.png")
ij.py.run_plugin("Gaussian Blur...", args={"sigma": 3}, imp=img)
# img is now blurred - show it directly (auto-converts ImagePlus)
ij.py.show(img)
```

**Only use WindowManager when plugins CREATE new images:**
```python
# ✅ CORRECT: Stitching creates a new image
tile1 = ij.IJ.createImage("Tile1", "8-bit random", 512, 512, 1)
tile2 = ij.IJ.createImage("Tile2", "8-bit random", 512, 512, 1)
args = {"first_image": tile1.getTitle(), "second_image": tile2.getTitle()}
ij.py.run_plugin("Pairwise stitching", args)
# New image was created, retrieve it
result = ij.WindowManager.getCurrentImage()
ij.py.show(result)  # Auto-converts ImagePlus to display
```

**Don't use WindowManager unnecessarily**
```python
# ❌ WRONG: Don't need WindowManager for in-place modifications
img = ij.IJ.openImage("https://samples.fiji.sc/blobs.png")
ij.py.run_plugin("Gaussian Blur...", args={"sigma": 3}, imp=img)
result = ij.WindowManager.getCurrentImage()  # Unnecessary!
```