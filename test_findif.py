import unittest
import numpy

from findif import grad

class NewTest(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_scalar(self):
        def f(x):
            return x**2
        self.assertAlmostEqual(grad(f)(3), 6)

    def test_2d(self):
        def f(x, y):
            return x**2 + y**2
        gradf = grad(f)
        numpy.testing.assert_allclose(gradf(3, 4), (6, 8))

    def test_2_array(self):
        def f(x_arr):
            return numpy.dot(x_arr, x_arr)

        x = numpy.array((3., 4.))
        numpy.testing.assert_allclose(grad(f)(x), (6, 8))
            
        

if __name__ == "__main__":
    unittest.main()
