## Tutorial Notebooks

To learn PyImageJ, follow this numbered sequence of tutorial notebooks:

1. [Starting PyImageJ](1-Starting-PyImageJ.ipynb)
2. [Opening and Displaying Images](2-Opening-and-Displaying-Images.ipynb)
3. [Sending Data to Java](3-Sending-Data-to-Java.ipynb)
4. [Retrieving Data from Java](4-Retrieving-Data-from-Java.ipynb)
5. [Convenience methods of PyImageJ](5-Convenience-methods-of-PyImageJ.ipynb)
6. [Working with Images](6-Working-with-Images.ipynb)
7. [Running Macros, Scripts, and Plugins](7-Running-Macros-Scripts-and-Plugins.ipynb)
8. [Discover and run ImageJ commands](8-Discover-and-run-ImageJ-commands.ipynb)
9. [Working with Large Images](9-Working-with-Large-Images.ipynb)
10. [Troubleshooting](10-Troubleshooting.ipynb)

## Reference Documentation

Reference guides exist detailing the following topics:

* [Installation](Install.md) - how to install PyImageJ to your system
* [Initialization](Initialization.md) - how to start PyImageJ from your Python program
* [Headless](Headless.md), [Xvfb](Xvfb.md) - ways to use PyImageJ when there is no computer screen
* [Troubleshooting](Troubleshooting.md) - common problems and their potential solutions
* [Development](Development.md) - how to develop the PyImageJ codebase

## Real World Examples

Here are links to some examples and use cases:

* The [examples](examples) in this repository demonstrate some workflows.
* The [cellprofiler](cellprofiler) folder in this repository contains some
  sample [CellProfiler](https://cellprofiler.org/) workflows demonstrating
  integration with ImageJ via PyImageJ.
* The [pyimagej-dextr](https://github.com/imagej/pyimagej-dextr) repository
  uses Deep Extreme Cut (DEXTR) to generate ImageJ ROIs via PyImageJ.
* [PoreSpy](https://github.com/PMEAL/porespy), a collection of image analysis
  tools used to extract information from 3D images of porous materials,
  uses some ImageJ filters via PyImageJ.
* Some projects at [LOCI](https://imagej.net/orgs/loci) use PyImageJ to invoke
  the [Image Stitching](https://imagej.net/plugins/image-stitching) and
  [SIFT-based image registration](https://imagej.net/plugins/linear-stack-alignment-with-sift)
  plugins of [Fiji](https://fiji.sc/), together with deep learning for
  multimodal image registration:
  * [Image registration based on SIFT feature matching](https://github.com/uw-loci/automatic-histology-registration-pyimagej/blob/8ad405170ec46dccbdc1c20fbbeb6eaff47b8b76/ij_sift_registration.ipynb)
  * [Generate pseudo modality via CoMIR for multimodal image registration](https://github.com/uw-loci/automatic-histology-registration-pyimagej/blob/8ad405170ec46dccbdc1c20fbbeb6eaff47b8b76/pseudo_modality.ipynb)
  * [Non-disruptive collagen characterization in clinical histopathology using cross-modality image synthesis](https://github.com/uw-loci/he_shg_synth_workflow/blob/v1.0.0/main.py#L167-L175)
  * [WSISR: Single image super-resolution for whole slide image using convolutional neural networks and self-supervised color normalization](https://github.com/uw-loci/demo_wsi_superres/blob/38283031eee4823d332fae1b6b32b5da33fb957f/train_compress.py#L162-L169)

To add your usage of PyImageJ to this list, please submit a pull request!
