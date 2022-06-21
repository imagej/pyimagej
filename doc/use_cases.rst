Use cases
=========

Below are use cases utitlizing PyImageJ and other packages such as :code:`scikit-image`.
These examples are available in the `PyImageJ repository`_.

.. toctree::
    :maxdepth: 3

    Blob detection (interactive) <examples/blob_interactive.rst>
    Blob detection (headless) <examples/blob_xvfb.rst>
    CellProfiler <cellprofiler/README>
    Puncta Segmentation <Puncta-Segmentation>

------------------------------------------------------------------------

.. rubric:: More use cases

Here are additional links to some other examples using PyImageJ
in other projects.

* The `pyimagej-dextr`_ repository uses Deep Extreme Cut (DEXTR)
  to generate ImageJ ROIs via PyImageJ.
* `PoreSpy`_, a collection of image analysis
  tools used to extract information from 3D images of porous materials,
  uses some ImageJ filters via PyImageJ.
* Some projects at `LOCI`_ use PyImageJ to invoke
  the `Image Stitching`_ and `SIFT-based image registration`_
  plugins of `Fiji`_, together with deep learning for
  multimodal image registration:

  * `Image registration based on SIFT feature matching`_

  * `Generate pseudo modality via CoMIR for multimodal image registration`_

  * `Non-disruptive collagen characterization in clinical histopathology using cross-modality image synthesis`_

  * `WSISR - Single image super-resolution for whole slide image using convolutional neural networks and self-supervised color normalization`_

To add your usage of PyImageJ to this list, please submit a pull request!

.. _PyImageJ repository: https://github.com/imagej/pyimagej/tree/master/doc
.. _pyimagej-dextr: https://github.com/imagej/pyimagej-dextr
.. _PoreSpy: https://github.com/PMEAL/porespy
.. _LOCI: https://imagej.net/orgs/loci
.. _Image Stitching: https://imagej.net/plugins/image-stitching
.. _Fiji: https://fiji.sc/
.. _SIFT-based image registration: https://imagej.net/plugins/linear-stack-alignment-with-sift
.. _Image registration based on SIFT feature matching: https://github.com/uw-loci/automatic-histology-registration-pyimagej/blob/8ad405170ec46dccbdc1c20fbbeb6eaff47b8b76/ij_sift_registration.ipynb
.. _Generate pseudo modality via CoMIR for multimodal image registration: https://github.com/uw-loci/automatic-histology-registration-pyimagej/blob/8ad405170ec46dccbdc1c20fbbeb6eaff47b8b76/pseudo_modality.ipynb
.. _Non-disruptive collagen characterization in clinical histopathology using cross-modality image synthesis: https://github.com/uw-loci/he_shg_synth_workflow/blob/v1.0.0/main.py#L167-L175
.. _WSISR - Single image super-resolution for whole slide image using convolutional neural networks and self-supervised color normalization: https://github.com/uw-loci/demo_wsi_superres/blob/38283031eee4823d332fae1b6b32b5da33fb957f/train_compress.py#L162-L169
