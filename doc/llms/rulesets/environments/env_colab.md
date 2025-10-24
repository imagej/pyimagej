# Google Colab Environment Ruleset

## PRIMER NOTEBOOK SETUP
- Assume the notebook has been run and `ij` variable exists
- If `ij` is not available instruct user to build from pyimagej-primer.ipynb
- PyImageJ initialized in 'interactive' mode with Xvfb virtual display
- Pre-configured bundles from https://github.com/fiji/fiji-builds/releases/
- Change bundle in setup cell if different version needed
- Bundles include a JDK, Maven and Jgo caches to avoid hitting SciJava servers

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

## PACKAGE INSTALLATION
- ✅ ALWAYS: %pip install package_name (preferred in notebooks)
- ✅ ALTERNATIVE: !pip install package_name (works but less integrated)
- ❌ DISCOURAGE: conda (not reliable in Colab)
- IMPORTANT: %pip ensures packages install to the correct kernel
- Common packages: numpy, matplotlib, scikit-image, pandas

## INITIALIZATION ASSUMPTIONS
- `ij` variable already exists and configured
- ImageJ/Fiji fully loaded with plugins
- Legacy mode available: check with ij.legacy.isActive()
- Don't call imagej.init() - already done in setup

## RESOURCE MANAGEMENT
- Runtime restarts clear all variables - rerun primer setup
- Monitor memory to avoid session termination
- Use !commands for shell, %commands for magic

## N-DIMENSIONAL DATA HANDLING
- For 3D+ data, create interactive widgets:
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