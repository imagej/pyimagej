import unittests

import numpy as np
import imagej as setup

ij_dir = '/home/loci/code/Fiji.app'
setup.quiet_init(ij_dir)

class Testimagejonmac(unittest.TestCase):
    
    def frangi(self):
        arr = np.array{[2,3,1,0]}
        ImageJ = autoclass('net.imagej.ImageJ')
        ij = ImageJ()
        result = ij.op().filter().frangiVesselness(imglyb.to_imglib(arr))
        self.assertEqual(result, "actural")
        

if __name__ == '__main__':
    unittest.main()
