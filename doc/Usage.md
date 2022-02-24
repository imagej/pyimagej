# Using PyImageJ

This file covers some common use cases of PyImageJ.  If you still have
questions, [the Scientific Community Image Forum](https://forum.image.sc) is
the best place to get general help, ImageJ advice, and any other image
processing tasks.  Bugs can be reported to the PyImageJ GitHub [issue
tracker](issues).

## The ImageJ2 gateway

The ImageJ2 gateway is the object interface that lets you use ImageJ-related
features (see [Initialization.md](Initialization.md)).  This gateway contains
all of the regular ImageJ2 Java functions. PyImageJ also adds a module of
convenience functions under `ij.py`. For example, converting a numpy array to
an ImageJ2 dataset:
```python
import imagej
import numpy

array = numpy.random.rand([5, 4, 3])
ij = imagej.init()
dataset = ij.py.to_java(array)
```

## Converting between Java and Python

Converting between Java and Python is done using the `ij.py.to_java()` and
`ij.py.from_java()` functions.  A table of common data types and their
converted values is listed below.

| Python object                   | Java Object                                                    |
|---------------------------------|----------------------------------------------------------------|
| `numpy.ndarray`                 | `net.imglib2.python.ReferenceGuardingRandomAccessibleInterval` |
| `xarray.DataArray`              | `net.imagej.Dataset`                                           |
| `str`                           | `java.lang.String`                                             |
| `int`                           | `java.lang.Integer`                                            |
| `float`                         | `java.lang.Float`                                              |
| `list`                          | `java.util.ArrayList`                                          |
| `dict`                          | `java.util.LinkedHashMap`                                      |
| `tuple`                         | `java.util.ArrayList`                                          |

#### Numpy and xarrays are linked to Java equivalents 

The function `to_java` is capable of converting common Python and numpy data
types into their Java/ImageJ/ImageJ2 equivalent. There is one important nuance;
converting a `numpy.ndarray` or `xarray.DataArray` to Java creates a Java
object that points to the numpy array. **This means that changing the Java
object also changes the numpy array.**

Let's take a look at lists as an example. Lists are not linked.

```python
import imagej
ij = imagej.init()
ex_list = [1, 2, 3, 4]
java_list = ij.py.to_java(ex_list)
ex_list[0] = 4
print(java_list[0]) #  This will still equal 1!
```

By contrast, ops can operate on numpy arrays and change them,
though you need to wrap the arrays in `to_java` first:

```python
import numpy as np
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7, 8]])
arr_output = ij.py.new_numpy_image(arr1)

ij.op().run('multiply', ij.py.to_java(arr_output), ij.py.to_java(arr1), ij.py.to_java(arr2))
print(arr_output) # This output will be [[5, 12], [21, 32]]!  
```

## Working with ops

Working with ImageJ2 ops may require casting your data to different
data structures in order to send it through to the op.
See the [tutorial notebook series](README.md)
for several examples of op usage and how to troubleshoot.

## Using the original ImageJ

In order to use [original ImageJ](https://imagej.net/software/imagej) macros,
plugins, or other code you must initiate the environment with legacy supported.
If in doubt, you can check the `ij.legacy().isActive()` function to see if your
initialization worked properly.  See [Initialization.md](Initialization.md) for
a how-to on starting up PyImageJ with legacy support.

### Manipulating windows

In order to use a graphical user interface, you must also initialize PyImageJ
with `mode='gui'` or `mode='interactive'`. To work with windows, you can:

* Use ImageJ2's
  [`WindowService`](https://javadoc.scijava.org/ImageJ/net/imagej/display/WindowService.html)
  through `ij.window()`.

* Use the original ImageJ's
[`WindowManager`](https://javadoc.scijava.org/ImageJ1/index.html?ij/WindowManager.html)
using the `ij.WindowManager` property.

#### Convenience functions

Current convenience functions include `active_image_plus` to get the
`ij.ImagePlus` from the current window and `active_xarray` to convert the
current window into an `xarray.DataArray`.

You can also use `synchronize_ij1_to_ij2` to synchronize the current data
structures. See [Troubleshooting.md](Troubleshooting.md) for an explanation of
when this is needed.

#### WindowService
You can get a list of active ImageJ2 windows with the following command
```python
ij.window().getOpenWindows()
```

You can close any ImageJ2 windows through the following command.
```python
ij.window().clear()
```

#### Using original ImageJ macros, scripts, and plugins

Running an original ImageJ macro is as simple as providing the macro code in a
string and the arguments in a dictionary to `run_macro`. Modify the following
code to print your name, age, and city.

```python
import imagej
ij = imagej.init(['net.imagej:imagej', 'net.imagej:imagej-legacy'])

macro = """
#@ String name
#@ int age
#@ String city
#@output Object greeting
greeting = "Hello " + name + ". You are " + age + " years old, and live in " + city + "."
"""
args = {
    'name': 'Chuckles',
    'age': 13,
    'city': 'Nowhere'
}
result = ij.py.run_macro(macro, args)
print(result.getOutput('greeting'))  # Prints: Hello Chuckles.  You are 13 years old, and live in Nowhere.
```

Running scripts in other languages is similar, but you also have to specify the
file extension for the scripting language it is written in.

```python
import imagej
ij = imagej.init(['net.imagej:imagej', 'net.imagej.:imagej-legacy'])
language_extension = 'ijm'
result_script = ij.py.run_script(language_extension, macro, args)
print(result_script.getOutput('greeting'))
```

Finally, running plugins works in the same manner as macros. You simply enter
the plugin name as a string and the arguments in a dict. For the few plugins
that use ImageJ2-style macros (i.e., explicit booleans in the recorder), set
the optional variable `ij1_style=False`.

Here is an example of using the "Mean" plugin to blur an image:
  
```python
import imagej
ij = imagej.init('sc.fiji:fiji:2.0.0-pre-10')
ij.py.run_macro("""run("Blobs (25K)");""")
blobs = ij.py.active_image_plus()
ij.py.show(blobs)

plugin = 'Mean'
args = {
    'block_radius_x': 10,
    'block_radius_y': 10            
}
ij.py.run_plugin(plugin, args)
imp = ij.py.active_image_plus()
ij.py.show(imp)
```

  
  
## Other convenience methods of `ij.py`

#### ij.py.dims

This can be used to determine the dimensions of a numpy **or** ImageJ image:

```python
import imagej
import numpy as np
ij = imagej.init()

# numpy image
arr = np.zeros([10, 10])
print(ij.py.dims(arr)) # (10, 10)

# imagej image
img = ij.py.to_java(arr)
print(ij.py.dims(img)) # [10, 10]
```

#### `ij.py.new_numpy_image`

Takes a single image argument, which can either be a numpy image
or an imagej image.

```python
# create a new numpy image from a numpy image
arr2 = ij.py.new_numpy_image(arr)
print(type(arr2)) # <class `numpy.ndarray`>

# create a new numpy image from an imagej image
arr3 = ij.py.new_numpy_image(img) 
print(type(arr3)) # <class `numpy.ndarra`>
```
