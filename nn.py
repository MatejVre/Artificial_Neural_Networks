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
                        der = (self.zs[i-2] > 0)

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

    for g, grd in enumerate(model.gradients_b):
            numerical_gradients_b = np.zeros_like(grd)
            for i in range(grd.shape[0]):
                numerical_gradients_b[i] = (compute_numerical_gradient(model.biases[g], i, model, X, y_to_input))
            print(grd)
            print(numerical_gradients_b)
            np.testing.assert_almost_equal(grd, numerical_gradients_b, decimal=6)
    print("All bias gradients match!")



if __name__ == "__main__":

    # X, y = doughnut()
    # fitter = ANNClassification(units=[5])
    # fitter.fit(X, y, lr=0.3, epochs=9000)

    # preds = np.argmax(fitter.predict(X), axis=1)
    # print(preds)
    # print(y)
    # print(np.mean(preds == y))
    
    # X, y = squares()
    # fitter = ANNClassification(units=[5])
    # fitter.fit(X, y, lr=0.4, epochs=11000)

    # preds = np.argmax(fitter.predict(X), axis=1)
    # print(preds)
    # print(y)
    # print(np.mean(preds == y))

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
    
    fitter = ANNClassification(units=[2, 10, 3], lambda_=0.5, activation_function_names=["relu", "relu" , "sigmoid"], testing_grad=True)
    print(type(fitter).__name__)
    compare_gradients(X, y, y_encoded, fitter)
    print(fitter.activation_function_names)


    fitter = ANNRegression(units=[2, 15, 4], lambda_=0.5, activation_function_names=["relu", "relu", "sigmoid"], testing_grad=True)
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

####### MODEL COMPARISON #####

######################################################################################################################################
#From here on is the code COPIED from jupyter notebooks since we have to submit everything in one python file#########################
######################################################################################################################################

#Moved the imports here because they slow down the code for no reason
    # from torch import nn
    # import torch

    # class NeuralNetwork(nn.Module):

    #     def __init__(self):
    #         super().__init__()
    #         torch.manual_seed(42)
    #         self.flatten = nn.Flatten()
    #         self.linear_relu_stack = nn.Sequential(
    #             nn.Linear(4, 15),
    #             nn.ReLU(),
    #             nn.Linear(15, 3)
    #         )
    #         self._initialize_weights()

    #     def forward(self, x):
    #         x = self.flatten(x)
    #         logits = self.linear_relu_stack(x)
    #         return logits
        
    #     def _initialize_weights(self):
    #         # Use NumPy to generate random numbers for weight initialization
    #         for layer in self.linear_relu_stack:
    #             if isinstance(layer, nn.Linear):
    #                 # Use NumPy to generate random values for weight initialization
    #                 weight_init = np.random.uniform(0, 1, size=layer.weight.shape)
    #                 bias_init = np.ones_like(layer.bias.detach())  # Detach bias from computation graph

    #                 # Assign the NumPy-generated weights to PyTorch layers
    #                 layer.weight.data = torch.tensor(weight_init, dtype=torch.float32)
    #                 layer.bias.data = torch.tensor(bias_init, dtype=torch.float32)
        
    # def train(X, y, model, learning_rate, num_epochs, loss_fn):
    #     optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    #     model.train()
    #     torch.manual_seed(42)

    #     for epoch in range(num_epochs):
            
    #         preds = model(X)

    #         loss = loss_fn(preds, y)
            
    #         loss.backward()

    #         if (epoch + 1) % 5 == 0:
    #             print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    #         optimizer.step()
            
    #         optimizer.zero_grad()
                
    #         model.train() 

    #     print("Training complete.")

    # def test(X_test, y_test, model, loss_fn):
    #     model.eval()
        
    #     with torch.no_grad():
    #         preds = model(X_test)
    #     _, predicted = torch.max(preds, 1)
        
    #     correct = (predicted == y_test).sum().item()
    #     total = y_test.size(0)
    #     accuracy = 100 * correct / total
        
    #     loss = loss_fn(preds, y_test)
                                                                    
    #     print(f"Test Accuracy: {accuracy:.2f}%")
    #     print(f"Test Loss (Cross-Entropy): {loss:.4f}")
    #     return predicted
    # from sklearn.preprocessing import StandardScaler
    # import matplotlib.pyplot as plt
    # from nn import *
    # from sklearn import datasets
    # from sklearn.model_selection import train_test_split
    # import random
    # iris = datasets.load_iris()
    # X, y = iris.data, iris.target

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)

    # X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    # X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    # y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    # y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # seed_value = 42
    # np.random.seed(seed_value)
    # torch.manual_seed(seed_value)
    # torch.cuda.manual_seed_all(seed_value)

    # torch_results = []
    # torch_preds = []

    # my_results = []
    # my_preds = []

    # similarity = []
    # epochs = range(10, 750, 10)

    # for eps in epochs:
    #     model_torch = NeuralNetwork()
    #     train(X_train_tensor, y_train_tensor, model_torch, 0.01, eps, nn.CrossEntropyLoss())
    #     preds = test(X_test_tensor, y_test_tensor, model_torch, nn.CrossEntropyLoss()).detach().numpy()
    #     torch_preds.append(preds)
    #     torch_results.append(np.mean(preds == y_test))
    #     my_model = ANNClassification([15], ["relu"])
    #     my_model.fit(X_train, y_train, 0.01, eps)
    #     preds2 = np.argmax(my_model.predict(X_test), axis=1)
    #     my_preds.append(preds2)
    #     my_results.append(np.mean(preds2 == y_test))

    #     similarity.append(np.mean(preds2 == preds))
    
    # plt.plot(epochs, torch_results, label="Torch Results")
    # plt.plot(epochs, my_results, linestyle="--", alpha=1, label="My Results")
    # plt.plot(epochs, similarity, label="Similarity")

    # plt.legend()

    # plt.xlabel('Epochs')
    # plt.ylabel('Value')

    # plt.savefig("Classification comparison.pdf", bbox_inches="tight")



#This here showcases the tests i have completed. Probably not a smart idea to run this within this file, or at least,
#delete the unittest import after running to avoid circular imports.


    # class test_ANN(unittest.TestCase):

    # def setUp(self):
    #     self.X = np.array([[1, 1, 1],
    #                        [1, 0, 1],
    #                        [1, 0, 0],
    #                        [0, 0, 0],
    #                        [0, 1, 1]])
    #     self.y = np.array([1, 1, 0, 1, 1])

    #     self.X_short = np.array([[1, 1],
    #                              [0,1]])
    #     self.y_short = np.array([1,0])

    # def test_build_classification(self):
    #     fitter = ANNClassification(units=[7, 21, 25], testing=True)
    #     fitter.build(self.X, self.y)

    #     self.assertEqual(fitter.ws[0].shape, (7, 3))
    #     self.assertEqual(fitter.ws[1].shape, (21, 7))
    #     self.assertEqual(fitter.ws[2].shape, (25, 21))
    #     self.assertEqual(fitter.ws[3].shape, (2, 25))

    #     self.assertEqual(fitter.biases[0].shape[0], 7)
    #     self.assertEqual(fitter.biases[1].shape[0], 21)
    #     self.assertEqual(fitter.biases[2].shape[0], 25)
    #     self.assertEqual(fitter.biases[3].shape[0], 2)

    # def test_forward_pass_classification(self):
    #     fitter = ANNClassification(units=[2], testing=True)
    #     fitter.build(self.X_short, self.y_short)

    #     preds = fitter.predict(self.X_short)

    #     np.testing.assert_allclose(preds, np.array([[0.5, 0.5],
    #                                      [0.5, 0.5]]), atol=0.001)
    
    # def test_build_regression(self):
    #     fitter = ANNRegression(units=[7, 21, 25], testing=True)
    #     fitter.build(self.X, self.y)

    #     self.assertEqual(fitter.ws[0].shape, (7, 3))
    #     self.assertEqual(fitter.ws[1].shape, (21, 7))
    #     self.assertEqual(fitter.ws[2].shape, (25, 21))
    #     self.assertEqual(fitter.ws[3].shape, (1, 25))

    #     self.assertEqual(fitter.biases[0].shape[0], 7)
    #     self.assertEqual(fitter.biases[1].shape[0], 21)
    #     self.assertEqual(fitter.biases[2].shape[0], 25)
    #     self.assertEqual(fitter.biases[3].shape[0], 1)

    # def test_forward_pass_regression(self):
    #     fitter = ANNRegression(units=[2], testing=True)
    #     fitter.build(self.X_short, self.y_short)

    #     vals = fitter.predict(self.X_short)
    #     np.testing.assert_allclose(vals, np.array([2.90515, 2.76159]), atol=0.001)