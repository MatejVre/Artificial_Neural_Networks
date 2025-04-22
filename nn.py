import numpy as np
import csv


class ANNClassification:
    # implement me
    def __init__(self, units=[], lambda_=0, testing=False):
        self.units = units
        self.weights = []
        self.biases = []
        self.lambda_ = lambda_

        self.zs = []
        self.activations = []

        self.input_layer_size = None
        self.output_layer_size = None

        self.gradients_w = [] #this part here is meant for testing the network
        self.gradients_b = []
        self.testing = testing

    def build(self, X, y):
        #input and output layers depend on the data?
        self.input_layer_size = X.shape[1]
        
        unique_y = np.unique(y)
        self.output_layer_size = len(unique_y)

        previous_layer_output_size = self.input_layer_size
        for unit in self.units:

            w = np.random.uniform(0, 1, size=(unit, previous_layer_output_size)) if not self.testing else np.ones((unit, previous_layer_output_size))

            self.weights.append(w) #only for testing puropses for now, will be changed to random later :)
            previous_layer_output_size = unit

            self.biases.append(np.ones((unit)))
        
        w = np.random.uniform(0, 1, size=(self.output_layer_size, previous_layer_output_size)) if not self.testing else np.ones((self.output_layer_size, previous_layer_output_size))
        self.weights.append(w)

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

            #TESTING
            self.gradients_w = [0] * len(self.weights)
            self.gradients_b = [0] * len(self.biases)

            for i in range(len(self.weights), 0, -1):

                #print(i)
                # if i == len(self.weights):
                #     derivative_w = 1/X.shape[0] * (delta.T @ self.activations[i-1])
                #     derivative_b = np.mean(delta, axis=0)

                #     # print("here")
                #     # print(derivative_w)
                #     # print(derivative_b)
                
                # else:
                derivative_w = (delta.T @ self.activations[i-1]) #1/X.shape[0] * 
                derivative_b = np.sum(delta, axis=0)

                self.gradients_w[i-1] = derivative_w
                self.gradients_b[i-1] = derivative_b
                    # print("HI")
                # print(derivative_w)
                # print(derivative_b)

                delta = (delta @ self.weights[i-1]) * (self.activations[i-1] * (1-self.activations[i-1]))
                if not self.testing:
                    self.weights[i-1] -= lr*derivative_w
                    self.biases[i-1] -= lr*derivative_b

        return self

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))
    
    def softmax(self, z):
        return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)
    
    def weights(self): #this seems pointless
        return self.weights
    
def cross_entropy_loss(y_pred, y_true, epsilon=1e-15):

    # Clip predictions to avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # Compute cross-entropy
    loss = -np.sum(y_true * np.log(y_pred), axis=1)  # shape (batch_size,)
    return np.sum(loss)

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

def compare_gradients(X, y, y_encoded):
    #testing true will not update the weights
    fitter = ANNClassification(units=[3, 4], testing=True)
    fitter.fit(X, y)

    for u, grad in enumerate(fitter.gradients_w):
        numerical_gradients_w = np.zeros_like(grad)
        for i in range(grad.shape[0]):
            for j in range(grad.shape[1]):
                numerical_gradients_w[i, j] = (compute_numerical_gradient(fitter.weights[u], (i,j), fitter, X, y_encoded))
            

        np.testing.assert_almost_equal(grad, numerical_gradients_w, decimal=6)
    print("All weight gradients match!")

    for u, grad in enumerate(fitter.gradients_b):
            numerical_gradients_b = np.zeros_like(grad)
            for i in range(grad.shape[0]):
                numerical_gradients_b[i] = (compute_numerical_gradient(fitter.biases[u], i, fitter, X, y_encoded))

            np.testing.assert_almost_equal(grad, numerical_gradients_b, decimal=6)
    print("All bias gradients match!")

if __name__ == "__main__":
    pass
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
            

    # print(numerical_gradients_w)
    # print(numerical_gradients_b)

    X, y = doughnut()
    fitter = ANNClassification(units=[3])
    fitter.fit(X, y, lr=0.01, epochs=3000)

    preds = np.argmax(fitter.predict(X), axis=1)
    print(preds)
    print(y)
    print(np.mean(preds == y))
    #compare_gradients(X, y, y_encoded)
    X, y = squares()
    fitter = ANNClassification(units=[5])
    fitter.fit(X, y, lr=0.01, epochs=5000)

    preds = np.argmax(fitter.predict(X), axis=1)
    print(preds)
    print(y)
    print(np.mean(preds == y))
