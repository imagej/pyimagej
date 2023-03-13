def test_when_imagej_starts(ij):
    """
    The ImageJ2 gateway test fixture registers a callback function via
    when_imagej_starts, which injects a small piece of data into the gateway
    object. We check for that data here to make sure the callback happened.
    """
    assert "success" == getattr(ij, "_when_imagej_starts_result", None)
