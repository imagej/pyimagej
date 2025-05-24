from scyjava import jimport

def test_when_imagej_starts(ij):
    """
    The ImageJ2 gateway test fixture registers a callback function via
    when_imagej_starts, which injects a small piece of data into the gateway
    object. We check for that data here to make sure the callback happened.
    """
    System = jimport("java.lang.System")
    version = str(System.getProperty("java.version"))
    digits = version.split(".")
    major = digits[1] if digits[0] == "1" else digits[0]
    assert major == getattr(ij, "_java_version", None)
