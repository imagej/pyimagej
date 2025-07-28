import scyjava as sj
import pytest

# -- Helpers --


def ensure_legacy_enabled(ij):
    if not ij.legacy or not ij.legacy.isActive():
        pytest.skip("No original ImageJ. Skipping test.")


def get_img(ij):
    # a simple blank 512x512 image
    dims = sj.jarray("j", [2])
    dims[0] = 512
    dims[1] = 512
    return ij.op().run("create.img", dims)


def get_script() -> str:
    return "\n#@ Img img\n#@output Integer out\nd = img.dimensionsAsLongArray()\nout = d[0] + d[1]\n"


def get_macro() -> str:
    return "\n#@output Integer out\ngetDimensions(w, h, c, s, f);\nout = w + h;\n"


# -- Tests --


def test_groovy_script(ij):
    args = {"img": get_img(ij)}
    script = get_script()
    output = ij.py.run_script("Groovy", script, args)
    assert output["out"] == 1024


def test_imagej_macro(ij):
    ensure_legacy_enabled(ij)
    _imp = ij.py.to_imageplus(get_img(ij)).show()
    macro = get_macro()
    output = ij.py.run_macro(macro)
    assert output["out"] == 1024
