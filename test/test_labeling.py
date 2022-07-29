from typing import Any, Dict, List

import pytest
import scyjava as sj
import numpy as np

from jpype import JObject, JException, JArray, JInt, JLong


@pytest.fixture(scope="module")
def py_labeling():
    import labeling as lb

    a = np.zeros((4, 4), np.int32)
    a[:2] = 1
    example1_images = []
    example1_images.append(a)
    b = a.copy()
    b[:2] = 2
    example1_images.append(np.flip(b.transpose()))
    c = a.copy()
    c[:2] = 3
    example1_images.append(np.flip(c))
    d = a.copy()
    d[:2] = 4
    example1_images.append(d.transpose())

    merger = lb.Labeling.fromValues(np.zeros((4, 4), np.int32))
    merger.iterate_over_images(example1_images, source_ids=["a", "b", "c", "d"])
    return merger


@pytest.fixture(scope="module")
def java_labeling(ij_fixture):

    img = np.zeros((4, 4), dtype=np.int32)
    img[:2, :2] = 6
    img[:2, 2:] = 3
    img[2:, :2] = 7
    img[2:, 2:] = 4
    img_java = ij_fixture.py.to_java(img)
    sets = [[], [1], [2], [1, 2], [2, 3], [3], [1, 4], [3, 4]]
    sets = [set(l) for l in sets]
    sets_java = ij_fixture.py.to_java(sets)

    ImgLabeling = sj.jimport("net.imglib2.roi.labeling.ImgLabeling")
    return ImgLabeling.fromImageAndLabelSets(img_java, sets_java)


def assert_labels_equality(
    exp: Dict[str, Any], act: Dict[str, Any], ignored_keys: List[str]
):
    for key in exp.keys():
        if key in ignored_keys:
            continue
        assert exp[key] == act[key]


class TestImgLabelingConversions(object):
    def test_py_to_java(self, ij_fixture, py_labeling, java_labeling):
        j_convert = ij_fixture.py.to_java(py_labeling)
        # Assert indexImg equality
        expected_img = ij_fixture.py.from_java(java_labeling.getIndexImg())
        actual_img = ij_fixture.py.from_java(j_convert.getIndexImg())
        assert np.array_equal(expected_img, actual_img)
        # Assert label sets equality
        expected_labels = ij_fixture.py.from_java(
            java_labeling.getMapping().getLabelSets()
        )
        actual_labels = ij_fixture.py.from_java(j_convert.getMapping().getLabelSets())
        assert expected_labels == actual_labels

    def test_java_to_py(self, ij_fixture, py_labeling, java_labeling):
        # Convert
        p_convert = ij_fixture.py.from_java(java_labeling)
        # Assert indexImg equality
        exp_img, exp_labels = py_labeling.get_result()
        act_img, act_labels = p_convert.get_result()
        assert np.array_equal(exp_img, act_img)
        # Assert (APPLICABLE) metadata equality
        # Skipping numSources - ImgLabeling doesn't have this
        # Skipping indexImg - py_labeling wasn't loaded from file
        assert_labels_equality(
            vars(exp_labels), vars(act_labels), ["numSources", "indexImg"]
        )

    def test_py_java_py(self, ij_fixture, py_labeling):
        # Convert
        to_java = ij_fixture.py.to_java(py_labeling)
        back_to_py = ij_fixture.py.from_java(to_java)
        print(py_labeling.label_sets)
        print(back_to_py.label_sets)
        # Assert indexImg equality
        exp_img, exp_labels = py_labeling.get_result()
        act_img, act_labels = back_to_py.get_result()
        assert np.array_equal(exp_img, act_img)
        # Assert (APPLICABLE) metadata equality
        # Skipping numSources - ImgLabeling doesn't have this
        # Skipping indexImg - py_labeling wasn't loaded from file
        assert_labels_equality(
            vars(exp_labels), vars(act_labels), ["numSources", "indexImg"]
        )
