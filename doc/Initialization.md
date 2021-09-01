# How to initialize ImageJ

The PyImageJ plugin works by setting up a gateway interface into ImageJ. This
gateway interface is activated using the `imagej.init()` function and yields a
wrapped ImageJ Java class. This interface has access to all of the Java based
functions, and also has convenience functions for translating back and forth
between python in `imagej.py`.

Setting up this gateway consists of two steps. The first step is to set Java
options. This step is optional, but must be done first because they cannot be
changed after ImageJ is initialized. The second step is to specify what version
of ImageJ the gateway will represent.

## Quick start

If all you want is the newest version of [ImageJ2](https://imagej.net/ImageJ2),
with no custom configuration (e.g., extra memory allocated to Java), use this:

```python
import imagej
ij = imagej.init()
```

It will download and and cache ImageJ, then spin up a gateway for you.

### Configuring the JVM

The ImageJ gateway is initialized through a Java Virtual Machine (JVM).
If you want to [configure the
JVM](https://docs.oracle.com/javase/8/docs/technotes/tools/unix/java.html),
it must be done _before_ calling initializing an ImageJ gateway. E.g.:

```python
import imagej
import scyjava
scyjava.config.add_option('-Xmx6g')
ij = imagej.init()
```
See "With more memory available to Java" below for further details.

## Ways to initialize

PyImageJ can be initialized to call different versions of ImageJ, with the
ability to include legacy support for ImageJ1, a GUI, Fiji plugins, or specific
versions of component libraries. Complex Maven endpoints can be entered as a
single string, or can be a list of valid Maven endpoints.

| Requirement                                   | Code                                                                               | Reproducible? |
|:----------------------------------------------|:----------------------------------------------------------------------------------:|:-------------:|
| Newest available version of ImageJ            | `ij = imagej.init()`                                                               | NO            |
| Specific version of ImageJ                    | `ij = imagej.init('2.1.0')`                                                        | YES           |
| With a GUI (newest version)                   | `ij = imagej.init(headless=False)`                                                 | NO            |
| With a GUI (specific version)                 | `ij = imagej.init('net.imagej:imagej:2.1.0', headless=False)`                      | YES           |
| With support for ImageJ 1.x (newest versions) | `ij = imagej.init(['net.imagej:imagej', 'net.imagej:imagej-legacy'])`              | NO            |
| With Fiji plugins (newest version)            | `ij = imagej.init('sc.fiji:fiji')`                                                 | NO            |
| With Fiji plugins (specific version)          | `ij = imagej.init('sc.fiji:fiji:2.1.1')`                                           | YES           |
| From a local installation                     | `ij = imagej.init('/Applications/Fiji.app')`                                       | DEPENDS       |
| With a specific plugin                        | `ij = imagej.init(['net.imagej:imagej', 'net.preibisch:BigStitcher'])`             | NO            |
| With a specific plugin version                | `ij = imagej.init(['net.imagej:imagej:2.1.0', 'net.preibisch:BigStitcher:0.4.1'])` | YES           |


#### WARNING: Transitive dependency versions

When specifying a particular version, e.g. `ij = imagej.init('sc.fiji:fiji:2.1.1')`, please note that **you will not** obtain the (transitive) dependencies as specified at that version. So if, for example, you wanted to use the `TrakEM2` plugin at the version that shipped with `Fiji 2.1.1`, you would need to find that version and include it in your initializion string:

```python
import imagej
ij = imagej.init(['sc.fiji:fiji:2.1.1', 'sc.fiji:TrakEM2_:1.3.3'])
```

Note also that, while technically possible, it is not advised to explicitly specifiy versions for components in the `net.imglib2` or `org.scijava` domains as this could lead to instability of the `imglib2-imglyb` translation layer.

Please see [this discussion](https://github.com/scijava/scyjava/issues/23#issuecomment-888532488) for more technical information on this limitation.

#### Newest available version

If you want to launch the newest available release version of ImageJ:

```python
import imagej
ij = imagej.init()
```

This invocation will automatically download and cache the newest release of
[net.imagej:imagej](https://maven.scijava.org/#nexus-search;gav~net.imagej~imagej~~~).

#### Explicitly specified version

You can specify a particular version, to facilitate reproducibility:

```python
import imagej
ij = imagej.init('2.1.0')
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
[this article](https://github.com/imglib/imglyb#awt-on-macos)
for a workaround.

#### Including ImageJ 1.x support

By default, the ImageJ gateway will not include the
[legacy layer](https://imagej.net/Legacy) for backwards compatibility with
[ImageJ 1.x](https://imagej.net/ImageJ1). The legacy layer is necessary for
macros and any ImageJ1-based plugins. You can enable it as follows:

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

or at a reproducible version:

```python
import imagej
ij = imagej.init('sc.fiji:fiji:2.1.1')
```

#### From a local installation

If you have a local installation of [ImageJ2](https://imagej.net/ImageJ2)
such as [Fiji](https://fiji.sc/), you can wrap an ImageJ gateway around it:

```python
import imagej
ij = imagej.init('/Applications/Fiji.app')
```

Replace `/Applications/Fiji.app` with the path to your installation.

#### With more memory available to Java

Java's virtual machine (the JVM) has a "max heap" value limiting how much
memory it can use. You can increase the value as follows:

```python
import imagej
import scyjava
scyjava.config.add_option('-Xmx6g')
ij = imagej.init()
```

Replace `6g` with the amount of memory Java should have. Save some
memory for your core operating system and other programs, though; a good
rule of thumb is to give Java no more than 80% of your physical RAM.

You can also pass
[other JVM arguments](https://docs.oracle.com/javase/8/docs/technotes/tools/unix/java.html).

#### With a specific plugin

For plugins available via Maven, you can specify them in the init call. E.g.:

```python
import imagej
ij = imagej.init(['net.imagej:imagej', 'net.preibisch:BigStitcher'])
```

This can be done for the latest versions as above, or at fixed versions like:

```python
import imagej
ij=imagej.init(['net.imagej:imagej:2.1.0', 'net.preibisch:BigStitcher:0.4.1'])
```

#### Plugins without Maven endpoints

For plugins that are published to a Maven repository, it is preferred to
simply add them to the endpoint, rather than using the below approaches.

If you wish to use plugins which are not available as Maven artifacts,
you have a couple of options:

1. Use a local installation of ImageJ with the plugins, as described above.

2. Specify a remote version of ImageJ, but set `plugins.dir` to point to a
   local directory to discover the plugins from there. For example:

   ```python
   import imagej
   import scyjava
   plugins_dir = '/Applications/Fiji.app/plugins'
   scyjava.config.add_option(f'-Dplugins.dir={plugins_dir}')
   ij = imagej.init(['net.imagej:imagej', 'net.imagej:imagej-legacy'])
   ```

   Where `plugins_dir` is a path to a folder full of ImageJ plugins.
