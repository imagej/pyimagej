import pytest
import scyjava as sj

# -- Tests --


def test_plugins_load_using_pairwise_stitching(ij):
    try:
        sj.jimport("plugin.Stitching_Pairwise")
    except TypeError:
        pytest.skip("No Pairwise Stitching plugin available. Skipping test.")

    if not ij.legacy:
        pytest.skip("No original ImageJ. Skipping test.")
    if ij.ui().isHeadless():
        pytest.skip("No GUI. Skipping test.")

    tile1 = ij.IJ.createImage("Tile1", "8-bit random", 512, 512, 1)
    tile2 = ij.IJ.createImage("Tile2", "8-bit random", 512, 512, 1)
    args = {"first_image": tile1.getTitle(), "second_image": tile2.getTitle()}
    ij.py.run_plugin("Pairwise stitching", args)
    result_name = ij.WindowManager.getCurrentImage().getTitle()

    ij.IJ.run("Close All", "")

    assert result_name == "Tile1<->Tile2"
