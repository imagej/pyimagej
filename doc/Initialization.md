# How to initialize ImageJ

The PyImageJ plugin works by setting up a gateway interface into ImageJ. This gateway interface
is activated using the `imagej.init()` function and yields a wrapped ImageJ Java class.
This interface has access to all of the Java based functions, and also has convenience functions
for translating back and forth between python in `imagej.py`.  

Setting up this gateway consists of two steps.  The first step is to set Java options.
This step is optional, but must be done first because they cannot be changed after ImageJ
is initialized.  The second step is to specify what version of ImageJ the gateway will
represent.

## Quick start

If all you need is an ImageJ2 gateway, with no additional memory or changed options, then
you may enter the following line to get the most recent version of ImageJ from Maven.

```python
import imagej
ij = imagej.init()
```


## Setting JVM options
The ImageJ gateway is initialized through a Java Virtual Machine (JVM).  You must set
the [JVM options](https://docs.oracle.com/javase/8/docs/technotes/tools/unix/java.html) 
for the JVM through `scyjava.config` before calling ImageJ.

#### Example: Increasing Memory
The JVM has a "max heap" value limiting how much
memory it can use. You can increase it:

```python
import scyjava.config
scyjava.config.add_options('-Xmx6g')
import imagej
ij = imagej.init()
```

Replace `6g` with the amount of memory Java should have. You can also pass


## Ways to initialize

PyImageJ can be initialized to call different versions of ImageJ, with the ability to include
legacy support for ImageJ1, a GUI, Fiji plugins, or specific versions of component libraries.
Complex Maven endpoints can be entered as a single string, or can be a list of valid Maven endpoints.


| Requirement                                   | Code<sup>1</sup>                                                     | Reproducible?<sup>2</sup> |
|:----------------------------------------------|:---------------------------------------------------------------------|:-------------------------:|
| Newest available version of ImageJ            | `ij = imagej.init()`                                                 | NO                        |
| Specific version of ImageJ                    | `ij = imagej.init('net.imagej:imagej:2.0.0-rc-71')` <br> or <br>  ```ij=imagej.init('2.0.0-rc-71'')```                  | YES                       |
| With a GUI (newest version)                   | `ij = imagej.init(headless=False)`                                   | NO                        |
| With a GUI (specific version)                 | `ij = imagej.init('net.imagej:imageJ:2.0.0-rc-71', headless=False)`  | YES                       |
| With support for ImageJ 1.x (newest versions) | `endpoints = ['net.imagej:imagej',` <br> &emsp; `'net.imagej:imagej-legacy']` <br> `ij = imagej.init(endpoints)`     | NO                        |
| With Fiji plugins (newest version)            | `ij = imagej.init('sc.fiji:fiji')`                                   | NO                        |
| With Fiji plugins (specific version)          | `ij = imagej.init('sc.fiji:fiji:2.0.0-pre-10')`                      | YES                       |
| From a local installation                     | `ij = imagej.init('/Applications/Fiji.app')`                         | DEPENDS                   |
| With a specific plugin                        | `endpoints = ['net.imagej:imagej',` <br> &emsp; `'net.preibisch:BigStitcher']` <br> `ij = imagej.init(endpoints)`    | NO
| With a specific plugin version                | `endpoints = ['net.imagej.imagej:2.0.0-rc-71',` <br> &emsp; `'net.preibisch:BigStitcher:0.4.1']` <br> `ij = imagej.init(endpoints)` | YES      |

#### Newest available version


If you want to launch the newest available release version of ImageJ:

```python
import imagej
ij = imagej.init()
```

This invocation will automatically download and cache the newest release of
[net.imagej:imagej](http://maven.imagej.net/#nexus-search;gav~net.imagej~imagej~~~).

#### Explicitly specified version

You can specify a particular version, to facilitate reproducibility:

```python
import imagej
ij = imagej.init('2.0.0-rc-68')
ij.getVersion()
```

#### With graphical capabilities

If you want to have support for the graphical user interface:

```python
import imagej
ij = imagej.init(headless=False)
ij.ui().showUI()
```

Note there are issues with Java AWT via Python on macOS; see
[this article](https://github.com/imglib/imglyb#awt-through-pyjnius-on-osx)
for a workaround.

#### Including ImageJ 1.x support

By default, the ImageJ gateway will not include the
[legacy layer](https://imagej.net/Legacy) for backwards compatibility with
[ImageJ 1.x](https://imagej.net/ImageJ1).  The legacy layer is necessary for macros and any
ImageJ1 based plugins.
You can enable the legacy layer as follows:

```python
import imagej
ij = imagej.init(['net.imagej:imagej', 'net.imagej:imagej-legacy'])
```

#### Including Fiji plugins

By default, the ImageJ gateway will include base ImageJ2 functionality only,
without additional plugins such as those that ship with the
[Fiji](https://fiji.sc/) distribution of ImageJ.

You can create an ImageJ gateway including Fiji plugins as follows:

```python
import imagej
ij = imagej.init('sc.fiji:fiji')
```

#### From a local installation

If you have an installation of [ImageJ2](https://imagej.net/ImageJ2)
such as [Fiji](https://fiji.sc/), you can wrap an ImageJ gateway around it:

```python
import imagej
ij = imagej.init('/Applications/Fiji.app')
```

Replace `/Applications/Fiji.app` with the actual location of your installation.

#### With more memory available to Java

Java's virtual machine (the JVM) has a "max heap" value limiting how much
memory it can use. You can increase the value as follows:

```python
import scyjava.config
scyjava.config.add_options('-Xmx6g')
import imagej
ij = imagej.init()
```

Replace `6g` with the amount of memory Java should have. You can also pass
[other JVM arguments](https://docs.oracle.com/javase/8/docs/technotes/tools/unix/java.html).

#### With a specific plugin
For plugins that have Maven endpoints, you can specify them in the initialization call. 
```python
import imagej
ij = imagej.init(['net.imagej:imagej', 'net.preibisch:BigStitcher'])
```
This can be done for the latest version as above, or for a specific version, as below.
```python
import imagej
ij=imagej.init(['net.imagej:imagej:2.0.0-rc-71', 'net.preibisch:BigStitcher:0.4.1'])
```

#### Plugins without Maven endpoints
For plugins that are published to a Maven repository, it is preferred to simply add them to the endpoint,
rather than using the below approaches.

If you wish to use plugins that do not have Maven artifacts, you have a few main options.  

* Use a local installation of ImageJ that has the plugins, as described above.
* Specify a remote version of ImageJ, but point to a local directory to discover plugins.
```python
import imagej
import scyjava.config
plugins_dir = 'Path/To/Your/Plugins'
scyjava.config.add_options(f'-Dplugins.dir={plugins_dir}')
ij = imagej.init('net.imagej:imagej:2.0.0-rc-71')
```
