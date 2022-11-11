import numpy as np
import pytest
import scyjava as sj

# -- Fixtures --


@pytest.fixture
def img():
    # Create img
    ArrayImgs = sj.jimport("net.imglib2.img.array.ArrayImgs")
    img = ArrayImgs.bytes(2, 3, 4)
    # Insert a different value into each index
    tmp_val = 1
    cursor = img.cursor()
    while cursor.hasNext():
        cursor.next().set(tmp_val)
        tmp_val += 1
    # Return the new img
    return img


# -- Tests --


def test_slice_index(img):
    assert img[0, 0, 0].get() == 1


def test_slice_index_negative(img):
    assert img[-1, -1, -1].get() == 24


def test_slice_2d(img):
    Views = sj.jimport("net.imglib2.view.Views")
    expected = Views.hyperSlice(img, 0, 0)
    actual = img[0, :, :]
    for i in range(3):
        for j in range(4):
            assert expected[i, j] == actual[i, j]


def test_slice_2d_negative(img):
    Views = sj.jimport("net.imglib2.view.Views")
    expected = Views.hyperSlice(img, 0, 1)
    actual = img[-1, :, :]
    for i in range(3):
        for j in range(4):
            assert expected[i, j] == actual[i, j]


def test_slice_1d(img):
    Views = sj.jimport("net.imglib2.view.Views")
    expected = Views.hyperSlice(Views.hyperSlice(img, 0, 0), 0, 0)
    actual = img[0, 0, :]
    for i in range(4):
        assert expected[i] == actual[i]


def test_slice_1d_negative(img):
    Views = sj.jimport("net.imglib2.view.Views")
    expected = Views.hyperSlice(Views.hyperSlice(img, 0, 1), 0, 1)
    actual = img[-1, -2, :]
    for i in range(4):
        assert expected[i] == actual[i]


def test_slice_int(img):
    Views = sj.jimport("net.imglib2.view.Views")
    expected = Views.hyperSlice(img, 0, 0)
    actual = img[0]
    for i in range(3):
        for j in range(4):
            assert expected[i, j] == actual[i, j]


def test_slice_not_enough_dims(img):
    Views = sj.jimport("net.imglib2.view.Views")
    expected = Views.hyperSlice(Views.hyperSlice(img, 0, 0), 0, 0)
    actual = img[0, 0]
    for i in range(4):
        assert expected[i] == actual[i]


def test_step(img):
    # Create a stepped img via Views
    Views = sj.jimport("net.imglib2.view.Views")
    steps = sj.jarray("j", 3)
    steps[0] = 1
    steps[1] = 1
    steps[2] = 2
    expected = Views.subsample(img, steps)
    # Create a stepped img via slicing notation
    actual = img[:, :, ::2]
    for i in range(2):
        for j in range(3):
            for k in range(2):
                assert expected[i, j, k] == actual[i, j, k]


def test_step_not_enough_dims(img):
    # Create a stepped img via Views
    Views = sj.jimport("net.imglib2.view.Views")
    steps = sj.jarray("j", 3)
    steps[0] = 2
    steps[1] = 1
    steps[2] = 1
    expected = Views.subsample(img, steps)
    expected = Views.dropSingletonDimensions(expected)
    # Create a stepped img via slicing notation
    actual = img[::2]
    for i in range(3):
        for j in range(4):
            assert expected[i, j] == actual[i, j]


def test_slice_and_step(img):
    # Create a stepped img via Views
    Views = sj.jimport("net.imglib2.view.Views")
    intervaled = Views.hyperSlice(img, 0, 0)
    steps = sj.jarray("j", 2)
    steps[0] = 1
    steps[1] = 2
    expected = Views.subsample(intervaled, steps)
    # Create a stepped img via slicing notation
    actual = img[:1, :, ::2]
    for i in range(3):
        for j in range(2):
            assert expected[i, j] == actual[i, j]


def test_shape(img):
    assert hasattr(img, "shape")
    assert img.shape == (2, 3, 4)


def test_dtype(img):
    assert hasattr(img, "dtype")
    ByteType = sj.jimport("net.imglib2.type.numeric.integer.ByteType")
    assert img.dtype == ByteType


def test_ndim(img):
    assert hasattr(img, "ndim")
    assert img.ndim == 3


def test_transpose1d(img):
    img = img[0, 0]
    transpose = img.T
    for i in range(2):
        assert transpose[i] == img[i]


def test_transpose2d(img):
    img = img[0]
    transpose = img.T
    for i in range(3):
        for j in range(2):
            assert transpose[i, j] == img[j, i]


def test_transpose3d(img):
    transpose = img.T
    for i in range(4):
        for j in range(3):
            for k in range(2):
                assert transpose[i, j, k] == img[k, j, i]


def test_addition(img):
    actual = img + img
    expected = np.multiply(img, 2)
    for i in range(2):
        for j in range(3):
            for k in range(4):
                assert expected[i, j, k] == actual[i, j, k]


def test_subtraction(img):
    actual = img - img
    expected = np.multiply(img, 0)
    for i in range(2):
        for j in range(3):
            for k in range(4):
                assert expected[i, j, k] == actual[i, j, k]


def test_multiplication(img):
    actual = img * img
    expected = np.multiply(img, img)
    for i in range(2):
        for j in range(3):
            for k in range(4):
                assert expected[i, j, k] == actual[i, j, k]


def test_division(img):
    actual = img / img
    expected = np.divide(img, img)
    for i in range(2):
        for j in range(3):
            for k in range(4):
                assert expected[i, j, k] == actual[i, j, k]
