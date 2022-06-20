# CellProfiler and PyImageJ - _RunImageJScript_

One initial goal in the development of PyImageJ was to improve integration of
ImageJ with [CellProfiler](https://cellprofiler.org/), a Python-based
open-source tool for creating modular image analysis pipelines.

While CellProfiler has had an existing integration with Java for many years
[[1], [2]], more recently the JPype library has become very robust and now
serves as an ideal library for accessing Java resources from Python. PyImageJ
offers a new opportunity to better bridge these applications in a lightweight
manner without requiring fundamental structural changes to either platform,
yielding a connection that is simpler, more powerful and more performant. We
have accomplished this through the [RunImageJScript] CellProfiler module.
RunImageJScript functions like any other CellProfiler module, allowing the
user to execute an ImageJ2 script as part of their workflow. PyImageJ enables
translation between structures in each domain, mapping script inputs and
outputs to CellProfiler workflow settings. This allows CellProfiler to benefit
from image-analysis algorithms and functionality exclusive to ImageJ without
the speed penalty incurred by previous bridge attempts [[3]].

As an example use case, we demonstrated a >3-fold increase in performance of a
CellProfiler workflow that identifies and measures cells (see Figure below)
using ImageJ's [Trainable Weka Segmentation] plugin (TWS) via PyImageJ. We used
the Broad Biomage Benchmark Collection (BBBC) image set [[4]] BBBC030, a set of
DIC images of Chinese Hamster Ovary cells [[5]]; in order to assess
segmentation performance as well as cell merges and splits, the ground truth
outlines were converted to label matrices in CellProfiler and lightly edited
with the original authors' permission. A single image from the set of 60 was
trained with the TWS plugin to distinguish background from cell centers and
cell boundaries, and the classifier model was exported. RunImageJScript was
then used from inside CellProfiler, invoking TWS to apply the classifier onto
each of the 60 images and first segment cell centers and then cell boundaries;
we have previously found17 that separate prediction of cell interiors versus
boundaries decreases the numbers of cells accidentally split into more than one
object or merged with a neighboring cell. This approach outperformed two
CellProfiler-only classical image processing methods of identifying the cells
(Panel B), one using the EnhanceOrSuppressFeatures module to distinguish the
cells from background by their texture, and the other using the EnhanceEdges
module to distinguish the cells from background by looking for edges in the
image, especially at the task of minimizing incorrect merges and splits (Panel
C). As none of the CellProfiler workflows attempted to identify cells touching
the edges of the image, segmentation accuracy was assessed both for the entire
set of ground truth cells and for only the ground truth cells not touching the
cell border, yielding similar results.

------------------------------------------------------------------------------

![](figure.svg)

Figure: RunImageJScript, built on PyImageJ, allows running CellProfiler
workflows on the BBBC030 data set. CellProfiler 4.1 introduced the
RunImageJMacro module which relies upon writing images to disk in order to
exchange data with ImageJ.

**A**. Runtime of a pipeline using either an internal CellProfiler module (Smooth)
vs rolling ball background subtraction in ImageJ via RunImageJMacro or
RunImageJScript. Comparing execution time indicates that RunImageJScript is
more than 3x faster than RunImageJMacro and has comparable performance to
pipelines that do not use an ImageJ plugin.

**B**. Accuracy of three segmentation approaches on BBBC030 in CellProfiler. At all
Intersection-over-Union (IoU) thresholds, using RunImageJScript to call
ImageJ’s Trainable Weka Segmentation equals or outperforms CellProfiler-only
approaches. Solid lines are performance on all cells; dashed lines are
performance on non-edge-touching cells, which CellProfiler did not try to
segment.

**C**. Across all cells, RunImageJScript misses the fewest cells at IoU 0.7, as
well as better balancing errors in splitting and merging.

**D**. Outlines of segmented classified cells. Cells outlined in teal are
classified as touching at least one other cell, while cells outlined in orange
are classified as isolated. High performance at this task requires minimizing
incorrect merges and splits.

[1]: https://doi.org/10.4308/hjb.20.4.151
[2]: https://doi.org/10.1002/cpz1.89
[3]: https://doi.org/10.1186/s12859-021-04344-9
[4]: https://doi.org/10.1038/nmeth.2083
[5]: https://doi.org/10.1038/srep30420
[RunImageJScript]: https://imagej.net/plugins/runimagejscript
[Trainable Weka Segmentation]: https://imagej.net/plugins/tws
