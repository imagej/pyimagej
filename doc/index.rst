.. PyImageJ documentation master file, created by
   sphinx-quickstart on Fri Apr  8 08:47:48 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PyImageJ's documentation!
====================================

PyImageJ provides a set of wrapper functions for integration between ImageJ2 and Python.
It also supports the original ImageJ API and data structures.

A major advantage of this approach is the ability to combine ImageJ and ImageJ2 with
other tools available from the Python software ecosystem, including NumPy, SciPy, 
scikit-image, CellProfiler, OpenCV, ITK and many more.

.. toctree::
   :maxdepth: 3
   :caption: 🚀 Getting Started

   Install
   Initialization

.. toctree::
   :maxdepth: 3
   :caption: 🪄 How-to guides

   notebooks
   Headless
   Troubleshooting

.. toctree::
   :maxdepth: 3
   :caption: 🔬 Use cases

   Blob detection (interactive) <examples/blob_interactive.rst>
   Blob detection (headless) <examples/blob_xvfb.rst>
   CellProfiler <cellprofiler/README>
   Classic Segmentation <Classic-Segmentation>
   Puncta Segmentation <Puncta-Segmentation>
   other_use_cases

.. toctree::
   :maxdepth: 3
   :caption: 🛠️ Development

   Development

.. toctree::
   :maxdepth: 3
   :caption: 📚 Reference

   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
