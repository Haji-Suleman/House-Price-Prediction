import torch
from torch import nn  # nn contains all of PyTorch's buidling block for neural networks
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Check PyTorch version
torch.__version__


# Create *know* paramater
weight = 0.7
bias = 0.3

### Formula of Linear Regression
# y = a + bx (a => bias) and (b = weight or gradient)

########## Create ##########
start = 0
end = 1
step = 0.02
X = torch.tensor(np.array(pd.read_csv("data.csv")), dtype=torch.float32).unsqueeze(
    dim=0
)
X = X[0]


y = weight * X + bias
# plt.scatter(X, y)
# plt.xlabel("Input")
# plt.title("Making a Ml Model using pytorch")
# plt.ylabel("Output")
# plt.show()
###########################


### Create a train/test split ###
train_split = int(0.8 * len(X))
X_train, Y_train = X[:train_split], y[:train_split]
X_test, Y_test = X[train_split:], y[train_split:]
len(X_train), len(Y_train), len(X_test), len(Y_test)
#################################


### PLotting the prediction and real Data ###
def plot_prediction(prediction=None):
    """PLots training data,test data and compaer predictions."""
    plt.figure(figsize=(10, 7))
    plt.scatter(X_train, Y_train, c="b", s=15, label="Training Data")
    plt.scatter(X_test, Y_test, c="g", s=15, label="Testing Data")
    if prediction is not None:
        # Plot the prediction
        plt.scatter(X_test, prediction, c="r", s=15, label="Prediction Data")
    # Show the legend
    plt.legend(prop={"size": 14})


################################


#### Create linear regression model class  ####
class LinearRegressionModel(
    nn.Module
):  # <- almost everthing in PyTorch inherits from nn.Module
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(
            torch.rand(1, requires_grad=True, dtype=torch.float)
        )

        self.bias = nn.Parameter(torch.rand(1, requires_grad=True, dtype=torch.float))

    # Forward method to define the computation in the model
    def forward(self, X: torch.Tensor) -> torch.Tensor:  # <- "x" is the input data

        return self.weights * X + self.bias


# Create a random seed
RANDOM_SEED = torch.manual_seed(42)

# Create an instance of the model ( this is the subclass of nn.Module)

model_0 = LinearRegressionModel()

# check out the paramater
print(list(model_0.parameters()))


# List named parameters
model_0.state_dict()


# Make predictions with model
with torch.inference_mode():
    y_preds = model_0(X_test)

# You can also do something similar with torch.no_grad(), however, inferrence_mode() is preferred
y_preds

###################################################


### Setup a lost function (MAE Mean Absolute Error) ###

loss_fn = nn.L1Loss()


# Setups an optimizer (SGD Stochastic Gradient Descent)
optimizer = torch.optim.SGD(
    params=model_0.parameters(),  # parmas
    lr=0.01,  # lr = learning rate = possible the most hyperparameter you can set
    momentum=0.9,
)
#######################################################


#### Pytorch Training Loop ####
# An epoch is one loop through the data...
epochs = 1000


### Training
# 0. Loop through the data
for epoch in range(epochs):

    # Set the model to training mode
    model_0.train()  # train mode in PyTorch set all parameters that require gradient to require gradients

    # 1. Forward pass
    y_pred = model_0(X_train)

    # 2. Calculate the loss
    loss = loss_fn(Y_train, y_pred)

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Perform backpropagation on the loss with respect to the parameter of the model
    loss.backward()

    # 5. Step the optimizer (perform gradient descent)

    optimizer.step()  # by default how the optimizer changes will acculumate through the loop so... we have to zero them in step 3 for the next iteration of the loop

    model_0.eval()  # turns off gradient tracking
##################################################################

###### Showing Predicted Data ######
with torch.inference_mode():
    y_pred_new = model_0(X_test)
plot_prediction(prediction=y_preds)
plt.savefig("Test_Train_DATA.png")

plt.show()
plot_prediction(prediction=y_pred_new)
plt.savefig("Prediction_DATA.png")
plt.show()
#####################################
