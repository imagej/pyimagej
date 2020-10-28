pyimagej (on JPype) bug list
===

| issue | expected result | actual result | line |
| :---: | :---: | :---: | :---: |
| Use specific ij version | Load specified version | `TypeError: Class net.imagej.axis.EnumeratedAxis is not found` | `pyimagej/imagej/imagej.py`, line 166 |
| No `getTitle` attribute found | return image title | WindowManager does not have `getTitle` attribute -- does not appear in the API (old?) | `pyimagej/test/test_imagej.py` , line 93 |
| np.shae output does not match expected output | (3, 1279, 853) | (1279, 853, 3) | `pyimagej tutorials/manual axis`  |
| Unable to fix axis order | transpose axis | `TypeError: unhashable type: 'list'` | `pyimagej tutorials/manual axis`  |
| Macro error, `plugin = 'Mean'` | Use the `mean` plugin | Macro error: `Unrecognized command: "Mean" in line 1` | `pyimagej tutorials/run_plugin` |
| Cannot extract axis, no attribute `type` | return image axes | `AttributeError: object has no attribute 'type'` | `pyimagej tutorials/CalibratedAxis` |
| Macro error, `Pairwise stitching` | Use the `Pairwise stitching` | Macro error: `Unrecognized command: "Pairwise stitching"` | `pyimagej/test/test_imagej.py`, line 80 |
| Ambigous overloads | select the correct method to run | `TypeError: Ambiguous overloads found` | `pyimagej/test/test_pyimagej`, line 22 |
