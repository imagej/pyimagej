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


from jnius import autoclass
import numpy as np


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


if __name__ == '__main__':
    unittest.main()


