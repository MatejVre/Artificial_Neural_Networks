import numpy as np
import csv


class ArtificialNeuralNetwork:
    
    def __init__(self, units=[], activation_function_names=[],lambda_=0, testing=False, testing_grad=False):
        self.units = units
        self.ws = []
        self.biases = []
        self.lambda_ = lambda_
        self.activation_function_names = activation_function_names
        self.activation_functions = []

        self.zs = []
        self.activations = []

        self.input_layer_size = None
        self.output_layer_size = None

        self.gradients_w = [] #this part here is meant for testing the network
        self.gradients_b = []
        self.testing = testing #if testing with test files
        self.testing_grad = testing_grad #if testing gradients

        #WHY? If we do testing and testing grad, then the weights are initialized to 1 which leads to gradients on the small dataset to be small enough
        #which makes them match. This was uncovered when implementing relu without adapting backpropagation and noticing the grad tests did not fail
    
    def build(self, X, y):
        #input and output layers depend on the data?
        self.input_layer_size = X.shape[1]
        np.random.seed(42)
        
        self.set_output_layer_size(X, y)

        previous_layer_output_size = self.input_layer_size

        #set activation functions
        if len(self.activation_function_names) == 0:
            self.activation_functions = [sigmoid] * len(self.units)
            self.activation_function_names = ["sigmoid"] * len(self.units)

        else:
            assert len(self.activation_function_names) == len(self.units)

            for act_fun_name in self.activation_function_names:
                
                if act_fun_name == "sigmoid":
                    self.activation_functions.append(sigmoid)
                elif act_fun_name == "relu":
                    self.activation_functions.append(ReLU)
                else:
                    print("The activation function does not exist!")
                    

        for unit in self.units:

            w = np.random.uniform(0, 1, size=(unit, previous_layer_output_size)) if not self.testing else np.ones((unit, previous_layer_output_size))

            self.ws.append(w) #only for testing puropses for now, will be changed to random later :)
            previous_layer_output_size = unit

            self.biases.append(np.ones((unit)))
        
        w = np.random.uniform(0, 1, size=(self.output_layer_size, previous_layer_output_size)) if not self.testing else np.ones((self.output_layer_size, previous_layer_output_size))
        self.ws.append(w)

        self.biases.append(np.ones(self.output_layer_size))
    
    def set_output_layer_size(self, X, y):
        unique_y = np.unique(y)
        self.output_layer_size = len(unique_y)
    
    def predict(self, X):

        assert X.shape[1] == self.input_layer_size

        previous_output = X

        self.zs = []
        self.activations = [previous_output]

        for i in range(len(self.ws)):

            weights_i = self.ws[i]
            biases_i = self.biases[i]

            z = (np.dot(previous_output, weights_i.T) + biases_i)
            self.zs.append(z)

            if i == len(self.ws) -1:
                a = self.output_layer(z)
            
            else:
                a = self.activation_functions[i](z)

            self.activations.append(a)
            previous_output = a
        
        return previous_output
    
    def fit(self, X, y, lr=0.01, epochs=1):
        self.build(X, y)
        np.random.seed(42)

        Y = self.process_y(X, y)

        for _ in range(epochs):

            predictions = self.predict(X)
            delta = self.first_delta(predictions, Y)

            #TESTING
            self.gradients_w = [0] * len(self.ws)
            self.gradients_b = [0] * len(self.biases)

            for i in range(len(self.ws), 0, -1):
                
                derivative_w = 1/X.shape[0] * (delta.T @ self.activations[i-1]) #1/X.shape[0] * in case i use the average of the loss (to be decided)
                derivative_b = np.mean(delta, axis=0)

                self.gradients_w[i-1] = derivative_w
                self.gradients_b[i-1] = derivative_b

                derivative_w += 2 * self.lambda_ * self.ws[i-1]

                if i > 1:
                    if self.activation_function_names[i-2] == "sigmoid":
                        der = (self.activations[i-1] * (1-self.activations[i-1]))
                
                    elif self.activation_function_names[i-2] == "relu":
                        der = (self.zs[i-2] > 0).astype(float)

                    delta = (delta @ self.ws[i-1]) * der

                if not self.testing_grad:
                    self.ws[i-1] -= lr*derivative_w
                    self.biases[i-1] -= lr*derivative_b

        return self
    
    def output_layer(self, z):
        return softmax(z)
    
    def weights(self):
        list = []
        for i in range(len(self.ws)):
            list.append((np.column_stack((self.ws[i], self.biases[i]))).T)
        return list
    
    def process_y(self, X, y):
        encoding_indices = np.unique(y)
        encoded_labels = np.zeros((X.shape[0], len(encoding_indices)))

        for i, lab in enumerate(y):
            encoded_labels[i, lab] = 1
        
        return encoded_labels

    def first_delta(self, predictions, y):
        return predictions - y

class ANNClassification(ArtificialNeuralNetwork):
        
    def loss(self, y_pred, y_true, epsilon=1e-15): #Cross entropy loss for classification

        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        loss = -np.sum(y_true * np.log(y_pred), axis=1)
        return np.mean(loss) + self.lambda_ * sum(np.sum(w**2) for w in self.ws)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

def ReLU(z):
    return z * (z > 0)

class ANNRegression(ArtificialNeuralNetwork):

    def set_output_layer_size(self, X, y):
        self.output_layer_size = 1

    def output_layer(self, z):
        return np.reshape(z, (-1))
    
    def loss(self, vals, true_vals): #MSE renamed to loss for generalization
        return np.mean(1/2*(vals - true_vals)**2) + self.lambda_ * sum(np.sum(w**2) for w in self.ws)

    def process_y(self, X, y):
        return y

    def first_delta(self, predictions, y):
        return (predictions - y).reshape(-1, 1)


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

    # if type(model).__name__ == "ANNClassification":
    #     loss_function = cross_entropy_loss
    # else:
    #     loss_function = mean_squared_error

    param[param_index] = original_value + epsilon
    y_pred_plus = model.predict(X)
    loss_plus = model.loss(y_pred_plus, y)

    param[param_index] = original_value - epsilon
    y_pred_minus = model.predict(X)
    loss_minus = model.loss(y_pred_minus, y)

    param[param_index] = original_value

    return (loss_plus - loss_minus) / (2 * epsilon)

def compare_gradients(X, y, y_encoded, model):
    #testing true will not update the weights
    model.fit(X, y, epochs=1)

    if type(model).__name__ == "ANNClassification":
        y_to_input = y_encoded
    else:
        y_to_input = y


    for u, grad in enumerate(model.gradients_w):
        numerical_gradients_w = np.zeros_like(grad)
        for i in range(grad.shape[0]):
            for j in range(grad.shape[1]):
                numerical_gradients_w[i, j] = (compute_numerical_gradient(model.ws[u], (i,j), model, X, y_to_input))
            
        print(grad)
        print(numerical_gradients_w)
        np.testing.assert_almost_equal(grad, numerical_gradients_w, decimal=6)
    print("All weight gradients match!")

    for u, grad in enumerate(model.gradients_b):
            numerical_gradients_b = np.zeros_like(grad)
            for i in range(grad.shape[0]):
                numerical_gradients_b[i] = (compute_numerical_gradient(model.biases[u], i, model, X, y_to_input))
            print(grad)
            print(numerical_gradients_b)
            np.testing.assert_almost_equal(grad, numerical_gradients_b, decimal=6)
    print("All bias gradients match!")

if __name__ == "__main__":

    X, y = doughnut()
    fitter = ANNClassification(units=[5])
    fitter.fit(X, y, lr=0.3, epochs=9000)

    preds = np.argmax(fitter.predict(X), axis=1)
    print(preds)
    print(y)
    print(np.mean(preds == y))
    
    X, y = squares()
    fitter = ANNClassification(units=[5])
    fitter.fit(X, y, lr=0.4, epochs=11000)

    preds = np.argmax(fitter.predict(X), axis=1)
    print(preds)
    print(y)
    print(np.mean(preds == y))

    X = np.array([[1, 1, 1],
                  [4, 0, 1],
                  [6, 0, 0],
                  [0, 0, 0],
                  [0, 2, 1]])
    y = np.array([1, 1, 0, 0, 1])
    y_encoded = np.array([[0, 1],
                          [0, 1],
                          [1, 0],
                          [1, 0],
                          [0, 1]])
    
    fitter = ANNClassification(units=[2, 6, 3], lambda_=0.5, activation_function_names=["relu", "relu" , "sigmoid"], testing_grad=True)
    print(type(fitter).__name__)
    compare_gradients(X, y, y_encoded, fitter)
    print(fitter.activation_function_names)


    fitter = ANNRegression(units=[2, 15, 4], lambda_=0.5, activation_function_names=["relu", "sigmoid", "sigmoid"], testing_grad=True)
    print(type(fitter).__name__)
    compare_gradients(X, y, y_encoded, fitter)
    print(fitter.activation_function_names)

    fitter = ANNClassification(units=[2, 6, 3], activation_function_names=[], testing_grad=True)
    print(type(fitter).__name__)
    compare_gradients(X, y, y_encoded, fitter)
    print(fitter.activation_function_names)


    fitter = ANNRegression(units=[2, 15, 4], activation_function_names=[], testing_grad=True)
    print(type(fitter).__name__)
    compare_gradients(X, y, y_encoded, fitter)
    print(fitter.activation_function_names)

    fitter = ANNClassification(units=[], activation_function_names=[], testing_grad=True)
    print(type(fitter).__name__)
    compare_gradients(X, y, y_encoded, fitter)
    print(fitter.activation_function_names)

    fitter = ANNRegression(units=[], activation_function_names=[], testing_grad=True)
    print(type(fitter).__name__)
    compare_gradients(X, y, y_encoded, fitter)
    print(fitter.activation_function_names)

    fitter = ANNClassification(units=[2], activation_function_names=[], testing_grad=True)
    print(type(fitter).__name__)
    compare_gradients(X, y, y_encoded, fitter)
    print(fitter.activation_function_names)

    fitter = ANNRegression(units=[2], activation_function_names=[], testing_grad=True)
    print(type(fitter).__name__)
    compare_gradients(X, y, y_encoded, fitter)
    print(fitter.activation_function_names)

