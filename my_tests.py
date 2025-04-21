import unittest
import numpy as np

from nn import ANNClassification, ANNRegression

class test_ANN(unittest.TestCase):

    def setUp(self):
        self.X = np.array([[1, 1, 1],
                           [1, 0, 1],
                           [1, 0, 0],
                           [0, 0, 0],
                           [0, 1, 1]])
        self.y = np.array([1, 1, 0, 1, 1])

        self.X_short = np.array([[1, 1],
                                 [0,1]])
        self.y_short = np.array([1,0])

    def test_build(self):
        fitter = ANNClassification(units=[7, 21, 25])
        fitter.build(self.X, self.y)

        self.assertEqual(fitter.weights[0].shape, (7, 3))
        self.assertEqual(fitter.weights[1].shape, (21, 7))
        self.assertEqual(fitter.weights[2].shape, (25, 21))
        self.assertEqual(fitter.weights[3].shape, (2, 25))

        self.assertEqual(fitter.biases[0].shape[0], 7)
        self.assertEqual(fitter.biases[1].shape[0], 21)
        self.assertEqual(fitter.biases[2].shape[0], 25)
        self.assertEqual(fitter.biases[3].shape[0], 2)

    def test_forward_pass(self):
        fitter = ANNClassification(units=[2])
        fitter.build(self.X_short, self.y_short)

        preds = fitter.predict(self.X_short)

        np.testing.assert_allclose(preds, np.array([[0.5, 0.5],
                                         [0.5, 0.5]]), atol=0.001)
        
        print(fitter.activations)


#([[0.9481, 0.9481],[0.9405, 0.9405]])
if __name__ == "__main__":
    import unittest
    unittest.main()