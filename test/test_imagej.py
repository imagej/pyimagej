import argparse
import sys
import unittest
import imagej

if "--ij" in sys.argv:
    parser = argparse.ArgumentParser()
    parser.add_argument('--ij', default='/Applications/Fiji.app', help="set ij_dir")
    args = parser.parse_args()
    print("set ij_dir to " + args.ij)
    ij_dir = args.ij
    ij = imagej.init(ij_dir)
    sys.argv = sys.argv[2:]
else:
    ij_dir = None # Use newest release version, downloaded from Maven.
    ij = imagej.init(ij_dir)


from jnius import autoclass, cast
import numpy as np
import xarray as xr

class TestImageJ(unittest.TestCase):

    def testFrangi(self):
        input_array = np.array([[1000, 1000, 1000, 2000, 3000], [5000, 8000, 13000, 21000, 34000]])
        result = np.zeros(input_array.shape)
        ij.op().filter().frangiVesselness(ij.py.to_java(result), ij.py.to_java(input_array), [1, 1], 4)
        correct_result = np.array([[0, 0, 0, 0.94282, 0.94283], [0, 0, 0, 0.94283, 0.94283]])
        result = np.ndarray.round(result, decimals=5)
        self.assertTrue((result == correct_result).all())

    def testGaussian(self):
        input_array = np.array([[1000, 1000, 1000, 2000, 3000], [5000, 8000, 13000, 21000, 34000]])

        output_array = ij.op().filter().gauss(ij.py.to_java(input_array), 10)
        result = []
        correct_result = [8440, 8440, 8439, 8444]
        ra = output_array.randomAccess()
        for x in [0, 1]:
            for y in [0, 1]:
                ra.setPosition(x, y)
                result.append(ra.get().get())
        self.assertEqual(result, correct_result)

    """
    def testTopHat(self):
        ArrayList = autoclass('java.util.ArrayList')
        HyperSphereShape = autoclass('net.imglib2.algorithm.neighborhood.HyperSphereShape')
        Views = autoclass('net.imglib2.view.Views')

        result = []
        correct_result = [0, 0, 0, 1000, 2000, 4000, 7000, 12000, 20000, 33000]

        input_array = np.array([[1000, 1000, 1000, 2000, 3000], [5000, 8000, 13000, 21000, 34000]])
        output_array = np.zeros(input_array.shape)
        java_out = Views.iterable(ij.py.to_java(output_array))
        java_in = ij.py.to_java(input_array)
        shapes = ArrayList()
        shapes.add(HyperSphereShape(5))

        ij.op().morphology().topHat(java_out, java_in, shapes)
        itr = java_out.iterator()
        while itr.hasNext():
            result.append(itr.next().get())

        self.assertEqual(result, correct_result)
    """

    def testImageMath(self):
        Views = autoclass('net.imglib2.view.Views')

        input_array = np.array([[1, 1, 2], [3, 5, 8]])
        result = []
        correct_result = [192, 198, 205, 192, 198, 204]
        java_in = Views.iterable(ij.py.to_java(input_array))
        java_out = ij.op().image().equation(java_in, "64 * (Math.sin(0.1 * p[0]) + Math.cos(0.1 * p[1])) + 128")

        itr = java_out.iterator()
        while itr.hasNext():
            result.append(itr.next().get())
        self.assertEqual(result, correct_result)

    def testPluginsLoadUsingPairwiseStitching(self):
        if ij_dir is None:
            # HACK: Skip test if not testing with a local Fiji.app.
            return

        macro = """
        newImage("Tile1", "8-bit random", 512, 512, 1);
        newImage("Tile2", "8-bit random", 512, 512, 1);
        """
        plugin = 'Pairwise stitching'
        args = {'first_image': 'Tile1', 'second_image': 'Tile2'}

        ij.script().run('macro.ijm', macro, True).get()
        ij.py.run_plugin(plugin, args)
        WindowManager = autoclass('ij.WindowManager')
        result_name = WindowManager.getCurrentImage().getTitle()

        ij.script().run('macro.ijm', 'run("Close All");', True).get()

        self.assertEqual(result_name, 'Tile1<->Tile2')


    def main(self):
        unittest.main()


class TestXarrayConversion(unittest.TestCase):
    def testCstyleArrayWithLabeledDimsConverts(self):
        xarr = xr.DataArray(np.random.rand(5, 4, 3, 6, 12), dims=['T', 'Z', 'C', 'Y', 'X'],
                             coords={'X': range(0, 12), 'Y': np.arange(0, 12, 2), 'C': ['R', 'G', 'B'],
                                     'Z': np.arange(10, 50, 10), 'T': np.arange(0, 0.05, 0.01)},
                             attrs={'Hello': 'Wrld'})

        dataset = ij.py.to_java(xarr)
        axes = [cast('net.imagej.axis.DefaultLinearAxis', dataset.axis(axnum)) for axnum in range(5)]
        labels = [axis.type().getLabel() for axis in axes]
        origins = [axis.origin() for axis in axes]
        scales = [axis.scale() for axis in axes]

        self.assertListEqual(origins, [0, 0, 0, 10, 0])
        self.assertListEqual(scales, [1, 2, 1, 10, 0.01])

        self.assertListEqual(list(reversed(xarr.dims)), labels)

        self.assertEqual(xarr.attrs, ij.py.from_java(dataset.getProperties()))

    def testFstyleArrayWiathLabeledDimsConverts(self):
        xarr = xr.DataArray(np.ndarray([5, 4, 3, 6, 12], order='F'), dims=['t', 'z', 'c', 'y', 'x'],
                            coords={'x': range(0, 12), 'y': np.arange(0, 12, 2),
                                    'z': np.arange(10, 50, 10), 't': np.arange(0, 0.05, 0.01)},
                            attrs={'Hello': 'Wrld'})

        dataset = ij.py.to_java(xarr)
        axes = [cast('net.imagej.axis.DefaultLinearAxis', dataset.axis(axnum)) for axnum in range(5)]
        labels = [axis.type().getLabel() for axis in axes]
        origins = [axis.origin() for axis in axes]
        scales = [axis.scale() for axis in axes]

        self.assertListEqual(origins, [0, 10, 0, 0, 0])
        self.assertListEqual(scales, [0.01, 10, 1, 2, 1])

        self.assertListEqual([dim.upper() for dim in xarr.dims], labels)
        self.assertEqual(xarr.attrs, ij.py.from_java(dataset.getProperties()))

    def testDatasetConvertsToXarray(self):
        xarr = xr.DataArray(np.random.rand(5, 4, 3, 6, 12), dims=['t', 'z', 'c', 'y', 'x'],
                             coords={'x': list(range(0, 12)), 'y': list(np.arange(0, 12, 2)), 'c': [0, 1, 2],
                                     'z': list(np.arange(10, 50, 10)), 't': list(np.arange(0, 0.05, 0.01))},
                             attrs={'Hello': 'Wrld'})

        dataset = ij.py.to_java(xarr)

        invert_xarr = ij.py.from_java(dataset)
        self.assertTrue((xarr.values == invert_xarr.values).all())

        self.assertEqual(list(xarr.dims), list(invert_xarr.dims))
        for key in xarr.coords:
            self.assertTrue((xarr.coords[key] == invert_xarr.coords[key]).all())
        self.assertEqual(xarr.attrs, invert_xarr.attrs)



if __name__ == '__main__':
    unittest.main()


