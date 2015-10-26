import unittest
import numpy

from findif import grad, ndgrad

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
        numpy.testing.assert_allclose(ndgrad(f)(x), (6, 8))

    def test_2_matrix(self):
        def f(x):
            return x[0, 0]**2 + x[1, 1]**2 - x[0, 1]*x[1, 0]
        x = numpy.array([[1., 2.], [3., 4.]])
        numpy.testing.assert_allclose(ndgrad(f)(x), [[2, -3], [-2, 8]])

    def test_diff_class_method(self):
        class A(object):
            def __init__(self, data):
                self.x = data

            def exe(self):
                return self.x[0, 0]**2 + self.x[1, 1]**2 - self.x[0, 1]*self.x[1, 0]

        a_instance = A(numpy.array([[1., 2.], [3., 4.]]))
                
        def f(x):
            a = A(x)
            return a.exe()

        x = numpy.array([[1., 2.], [3., 4.]])
        numpy.testing.assert_allclose(ndgrad(f)(x), [[2, -3], [-2, 8]])

            
        

if __name__ == "__main__":
    unittest.main()
