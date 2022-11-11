import pytest
import scyjava as sj

# -- Tests --


def test_plugins_load_using_pairwise_stitching(ij_fixture):
    try:
        sj.jimport("plugin.Stitching_Pairwise")
    except TypeError:
        pytest.skip("No Pairwise Stitching plugin available. Skipping test.")

    if not ij_fixture.legacy:
        pytest.skip("No original ImageJ. Skipping test.")
    if ij_fixture.ui().isHeadless():
        pytest.skip("No GUI. Skipping test.")

    tile1 = ij_fixture.IJ.createImage("Tile1", "8-bit random", 512, 512, 1)
    tile2 = ij_fixture.IJ.createImage("Tile2", "8-bit random", 512, 512, 1)
    args = {"first_image": tile1.getTitle(), "second_image": tile2.getTitle()}
    ij_fixture.py.run_plugin("Pairwise stitching", args)
    result_name = ij_fixture.WindowManager.getCurrentImage().getTitle()

    ij_fixture.IJ.run("Close All", "")

    assert result_name == "Tile1<->Tile2"
