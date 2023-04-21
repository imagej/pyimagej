from typing import List

import numpy as np


class ROIDendron:
    """
    Parent class for containing children (ROIs).
    """

    def __init__(self):
        self.rois = []
        self.count = 0

    def add_roi(self, roi: "ROI"):
        """
        Add an instance of the ROI class to the ROIDendron.

        :param roi: An instance of the ROI class.
        """
        if roi is not None:
            roi.set_dendron(self)
            self.rois.append(roi)
            self._update_roi_count()
        else:
            return None

    def remove_roi(self, index: int):
        """
        Remove an instance of the ROI class from the ROIDendron.

        :param index: Index of the ROI to remove from the ROIDendron.
        """
        roi = self.rois[index]
        roi.set_dendron(None)
        self.rois.remove(roi)
        self._update_roi_count()

    def _update_roi_count(self):
        self.count = len(self.rois)


class ROI:
    """
    Base class for a Region of Interest (ROI).
    """

    def __init__(self):
        self.dendron = None
        self._data = None

    def set_dendron(self, dendron: "ROIDendron"):
        """
        Set the parent ROIDendron of the ROI.

        :param dendron: An instance of the ROIDendron class.
        """
        self.dendron = dendron


class Ellipsoid(ROI):
    """
    Initializes an Ellipsoid ROI instance with the provided data.

    :param data: A 2D numpy array, typically with shape (2, 2),
        where row index 0 defines the center position and row
        index 1 defines the semi axis length (e.g. radii).
    """

    def __init__(self, data: np.ndarray):
        super().__init__()
        self._data = data
        self._mv = memoryview(self._data)
        self.ndim = data.shape[1]
        self._center = self._mv[:1]
        self._semi_axis_length = self._mv[1:]

    def get_center(self) -> List[float]:
        """
        Get center position of the ellipsoid.

        :return: List of center position: [x, y].
        """
        return self._center.tolist()[0]

    def get_semi_axis_length(self) -> List[float]:
        """
        Get semi axis length (i.e. radii) of the ellipsoid.

        :return: List of semi axis length: [x, y].
        """
        return self._semi_axis_length.tolist()[0]


class Line(ROI):
    """
    Initializes a Line ROI instance with the provided data.

    :param data: A 2D numpy array, typically with shape (2, 2),
        where row index 0 defines endpoint one of the line and
        row index 1 defines endpoint two of the line.
    """

    def __init__(self, data: np.ndarray):
        super().__init__()
        self._data = data
        self.ndim = data.shape[1]
        self._mv = memoryview(self._data)
        self._endpoint_one = self._mv[:1]
        self._endpoint_two = self._mv[1:]

    def get_endpoint_one(self) -> List[float]:
        """
        Get endpoint one of the line.

        :return: List of endpoint one coordinates.
        """
        return self._endpoint_one.tolist()[0]

    def get_endpoint_two(self) -> List[float]:
        """
        Get endpoint two of the line.

        :return: List of endpoint two coordinates.
        """
        return self._endpoint_two.tolist()[0]


class Rectangle(ROI):
    """
    Initializes a Rectangle ROI instance with the provided data.

    :param data: A 2D numpy array, typically with shape (2, 2),
        where row index 0 defines the minimum dimension values and
        row index 1 defines the maximum dimension value of the rectangle.
    """

    def __init__(self, data: np.ndarray):
        super().__init__()
        self._data = data
        self._mv = memoryview(self._data)
        self.ndim = data.shape[1]
        self._min = self._mv[:1]
        self._max = self._mv[1:]

    def get_min_values(self) -> List[float]:
        """
        Get the minimum dimension values of the rectangle.

        :return: List of minimum dimension values.
        """
        return self._min.tolist()[0]

    def get_max_values(self) -> List[float]:
        """
        Get the maximum dimension values of the rectangle.

        :return: List of maximum dimension values.
        """
        return self._max.tolist()[0]


class Polygon(ROI):
    """
    Initializes a Polygon ROI instance with the provided data.

    :param data: A numpy array with shape [1, D], where D
        are the number of dimensions.
    """

    def __init__(self, data: np.ndarray):
        super().__init__()
        self._data = data
        self.ndim = data.shape[1]

    def get_vertices(self) -> np.ndarray:
        """
        Get the vertices of the polygon.

        :return: [1, D] numpy array containing coordinates.
        """
        return self._data
