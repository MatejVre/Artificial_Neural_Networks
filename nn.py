import numpy as np
import csv


class ANNClassification:
    # implement me
    def __init__(self, units=[], lambda_=0):
        self.units = units
        self.weights = []
        self.biases = []
        self.lambda_ = lambda_

        self.zs = []
        self.activations = []

        self.input_layer_size = None
        self.output_layer_size = None

    def build(self, X, y):
        #input and output layers depend on the data?
        self.input_layer_size = X.shape[1]
        
        unique_y = np.unique(y)
        self.output_layer_size = len(unique_y)

        previous_layer_output_size = self.input_layer_size
        for unit in self.units:

            self.weights.append(np.random.uniform(0, 1, size=(unit, previous_layer_output_size))) #only for testing puropses for now, will be changed to random later :)
            previous_layer_output_size = unit

            self.biases.append(np.ones((unit)))
        
        self.weights.append(np.random.uniform(0, 1, size=(self.output_layer_size, previous_layer_output_size)))

        self.biases.append(np.ones(self.output_layer_size))

    def predict(self, X):

        assert X.shape[1] == self.input_layer_size

        preds = np.zeros((X.shape[0], self.output_layer_size))

        previous_output = X

        self.zs = []
        self.activations = [previous_output]

        # self.zs.append(previous_output)
        # self.activations.append(previous_output)

        for i in range(len(self.weights)):

            weights_i = self.weights[i]
            biases_i = self.biases[i]

            z = (np.dot(previous_output, weights_i.T) + biases_i)
            self.zs.append(z)

            if i == len(self.weights) -1:
                a = self.softmax(z)
            
            else:
                a = self.sigmoid(z)

            self.activations.append(a)
            previous_output = a
        
        return previous_output
    
    def fit(self, X, y, lr=0.01, epochs=1):
        self.build(X, y)

        encoding_indices = np.unique(y)
        encoded_labels = np.zeros((X.shape[0], len(encoding_indices)))

        for i, lab in enumerate(y):
            encoded_labels[i, lab] = 1

        for _ in range(epochs):

            predictions = self.predict(X)
            delta = predictions - encoded_labels

            for i in range(len(self.weights), 0, -1):

                #print(i)
                if i == len(self.weights):
                    derivative_w = 1/X.shape[0] * (delta.T @ self.activations[i-1])
                    derivative_b = np.mean(delta, axis=0)

                    print("here")
                    print(derivative_w)
                    print(derivative_b)
                
                else:
                    derivative_w = 1/X.shape[0] * (delta.T @ self.activations[i-1])
                    derivative_b = np.mean(delta, axis=0)
                    print("HI")
                    print(derivative_w)
                    print(derivative_b)

                delta = (delta @ self.weights[i-1]) * (self.activations[i-1] * (1-self.activations[i-1]))
                # self.weights[i-1] -= lr*derivative_w
                # self.biases[i-1] -= lr*derivative_b

        return self



        
    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))
    
    def softmax(self, z):
        return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)
    
def cross_entropy_loss(y_pred, y_true, epsilon=1e-15):

    # Clip predictions to avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # Compute cross-entropy
    loss = -np.sum(y_true * np.log(y_pred), axis=1)  # shape (batch_size,)
    return np.mean(loss)

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

def compute_numerical_gradient(param, param_index, model, X, y, epsilon=1e-4):
    original_value = param[param_index]

    param[param_index] = original_value + epsilon
    y_pred_plus = model.predict(X)
    loss_plus = cross_entropy_loss(y_pred_plus, y)

    param[param_index] = original_value - epsilon
    y_pred_minus = model.predict(X)
    loss_minus = cross_entropy_loss(y_pred_minus, y)

    param[param_index] = original_value

    return (loss_plus - loss_minus) / (2 * epsilon)

if __name__ == "__main__":

    # example NN use
    # fitter = ANNClassification(units=[3,4], lambda_=0)
    # X = np.array([
    #     [1, 2, 3],
    #     [4, 5, 6],
    #     [7, 8, 9]
    # ], dtype=float)
    # y = np.array([0, 1, 2])
    # model = fitter.fit(X, y, 0.01, epochs=2000)
    # predictions = model.predict(X)
    # print(predictions)
    # y_encoded = np.array([[1,0,0],
    #                       [0,1,0],
    #                       [0,0,1]])
    # np.testing.assert_almost_equal(predictions,
    #                                [[1, 0, 0],
    #                                 [0, 1, 0],
    #                                 [0, 0, 1]], decimal=3)
    X = np.array([[1, 1, 1],
                  [1, 0, 1],
                  [1, 0, 0],
                  [0, 0, 0],
                  [0, 1, 1]])
    y = np.array([1, 1, 0, 0, 1])
    y_encoded = np.array([[0, 1],
                          [0, 1],
                          [1, 0],
                          [1, 0],
                          [0, 1]])
    
    fitter = ANNClassification(units=[2])
    fitter.fit(X, y)

    # fitter = ANNClassification(units=[2])
    # fitter.build(X, y)
    # print("predicting")

    numerical_gradients_w = np.zeros_like(fitter.weights[-2])
    numerical_gradients_b = np.zeros_like(fitter.biases[-2])
    for i in range(fitter.weights[-2].shape[0]):
        for j in range(fitter.weights[-2].shape[1]):
            numerical_gradients_w[i, j] = (compute_numerical_gradient(fitter.weights[-2], (i,j), fitter, X, y_encoded))

    for u in range(fitter.biases[-2].shape[0]):
            numerical_gradients_b[u] = (compute_numerical_gradient(fitter.biases[-2], u, fitter, X, y_encoded))
            

    print(numerical_gradients_w)
    print(numerical_gradients_b)

    # X, y = doughnut()
    # fitter = ANNClassification(units=[5])
    # fitter.fit(X, y, lr=0.1, epochs=5000)

    # preds = np.argmax(fitter.predict(X), axis=1)
    # print(preds)


