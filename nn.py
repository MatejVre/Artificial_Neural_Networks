import numpy as np
import csv


class ANNClassification:
    # implement me
    def __init__(self, units=[], lambda_=0):
        self.units = units
        self.weights = []
        self.biases = []
        self.lambda_ = lambda_

        self.input_layer_size = None
        self.output_layer_size = None

    def build(self, X, y):
        #input and output layers depend on the data?
        self.input_layer_size = X.shape[1]
        
        unique_y = np.unique(y)
        self.output_layer_size = len(unique_y)

        previous_layer_output_size = self.input_layer_size
        for unit in self.units:

            self.weights.append(np.ones((unit, previous_layer_output_size))) #only for testing puropses for now, will be changed to random later :)
            previous_layer_output_size = unit

            self.biases.append(np.ones((unit)))
        
        self.weights.append(np.ones((self.output_layer_size, previous_layer_output_size)))

        self.biases.append(np.ones(self.output_layer_size))

    def predict(self, X):

        assert X.shape[1] == self.input_layer_size
        
        preds = np.zeros((X.shape[0], self.output_layer_size))

        

class ANNRegression:
    # implement me too, please
    pass


# data reading

def read_tab(fn, adict):
    content = list(csv.reader(open(fn, "rt"), delimiter="\t"))

    legend = content[0][1:]
    data = content[1:]

    X = np.array([d[1:] for d in data], dtype=float)
    y = np.array([adict[d[0]] for d in data])

    return legend, X, y


def doughnut():
    legend, X, y = read_tab("doughnut.tab", {"C1": 0, "C2": 1})
    return X, y


def squares():
    legend, X, y = read_tab("squares.tab", {"C1": 0, "C2": 1})
    return X, y


if __name__ == "__main__":

    # example NN use
    fitter = ANNClassification(units=[3,4], lambda_=0)
    X = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ], dtype=float)
    y = np.array([0, 1, 2])
    model = fitter.fit(X, y)
    predictions = model.predict(X)
    print(predictions)
    np.testing.assert_almost_equal(predictions,
                                   [[1, 0, 0],
                                    [0, 1, 0],
                                    [0, 0, 1]], decimal=3)
