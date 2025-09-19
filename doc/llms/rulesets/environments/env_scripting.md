t# Fiji Script Editor Environment Ruleset

## FIJI PYTHON SCRIPT MODE
- Scripts run inside Fiji's "Python mode" (Edit > Options > Python...)
- Python interpreter is embedded within ImageJ/Fiji
- NEVER call imagej.init() - ImageJ is already running
- ALWAYS start scripts with: `#@ ImageJ ij`

## MANDATORY SCRIPT HEADER
```python
#@ ImageJ ij
```
Additional parameters as needed (see Script Parameters section)

## SCRIPT PARAMETERS (CRITICAL)
- Use ImageJ's parameter system for user input: https://imagej.net/scripting/parameters
- Common parameter types:
  ```python
  #@ String name
  #@ int value
  #@ File input_file
  #@ ImagePlus image
  #@ Dataset dataset
  #@ OpService ops
  ```
- Parameters are automatically injected as variables
- NO input() or GUI dialogs - use script parameters instead

## DISPLAY RESTRICTIONS
- ❌ NEVER use: matplotlib, plt.show(), ij.py.show()
- ❌ NEVER use: Python-specific display methods
- ✅ ALWAYS use: ImageJ's native display methods
- ✅ For images: ij.ui().show(image) or image.show()
- ✅ For results: ij.log().info("message") or print()

## IMAGEPLUS vs DATASET
- Legacy plugins expect ImagePlus objects
- Modern ImageJ2 uses Dataset objects
- Convert as needed: ij.py.to_imageplus() or ij.py.to_dataset()
- Check what your target plugin/operation expects

## SCRIPT OUTPUT
- Results appear in ImageJ's Log window
- Images open in ImageJ's display windows
- Use ij.log().info() for structured output
- Return values are handled by ImageJ's script framework

## PLUGIN INTEGRATION
- Full access to ImageJ/Fiji plugins via menu commands
- Use ij.command().run() for plugin execution
- Legacy macro functions available via ij.IJ.run()

