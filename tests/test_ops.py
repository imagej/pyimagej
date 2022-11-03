import numpy as np
import scyjava as sj

# -- Tests --


def test_frangi(ij_fixture):
    input_array = np.array(
        [[1000, 1000, 1000, 2000, 3000], [5000, 8000, 13000, 21000, 34000]]
    )
    result = np.zeros(input_array.shape)
    ij_fixture.op().filter().frangiVesselness(
        ij_fixture.py.to_java(result), ij_fixture.py.to_java(input_array), [1, 1], 4
    )
    correct_result = np.array(
        [[0, 0, 0, 0.94282, 0.94283], [0, 0, 0, 0.94283, 0.94283]]
    )
    result = np.ndarray.round(result, decimals=5)
    assert (result == correct_result).all()


def test_gaussian(ij_fixture):
    input_array = np.array(
        [[1000, 1000, 1000, 2000, 3000], [5000, 8000, 13000, 21000, 34000]]
    )
    sigmas = [10.0] * 2
    output_array = (
        ij_fixture.op().filter().gauss(ij_fixture.py.to_java(input_array), sigmas)
    )
    result = []
    correct_result = [8435, 8435, 8435, 8435]
    ra = output_array.randomAccess()
    for x in [0, 1]:
        for y in [0, 1]:
            ra.setPosition(x, y)
            result.append(ra.get().get())
    assert result == correct_result


def test_top_hat(ij_fixture):
    ArrayList = sj.jimport("java.util.ArrayList")
    HyperSphereShape = sj.jimport("net.imglib2.algorithm.neighborhood.HyperSphereShape")
    Views = sj.jimport("net.imglib2.view.Views")

    result = []
    correct_result = [0, 0, 0, 1000, 2000, 4000, 7000, 12000, 20000, 33000]

    input_array = np.array(
        [[1000, 1000, 1000, 2000, 3000], [5000, 8000, 13000, 21000, 34000]]
    )
    output_array = np.zeros(input_array.shape)
    java_out = Views.iterable(ij_fixture.py.to_java(output_array))
    java_in = ij_fixture.py.to_java(input_array)
    shapes = ArrayList()
    shapes.add(HyperSphereShape(5))

    ij_fixture.op().morphology().topHat(java_out, java_in, shapes)
    itr = java_out.iterator()
    while itr.hasNext():
        result.append(itr.next().get())

    assert result == correct_result


def test_image_math(ij_fixture):
    Views = sj.jimport("net.imglib2.view.Views")

    input_array = np.array([[1, 1, 2], [3, 5, 8]])
    result = []
    correct_result = [192, 198, 205, 192, 198, 204]
    java_in = Views.iterable(ij_fixture.py.to_java(input_array))
    java_out = (
        ij_fixture.op()
        .image()
        .equation(java_in, "64 * (Math.sin(0.1 * p[0]) + Math.cos(0.1 * p[1])) + 128")
    )

    itr = java_out.iterator()
    while itr.hasNext():
        result.append(itr.next().get())
    assert result == correct_result
