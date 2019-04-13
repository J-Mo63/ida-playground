def pytorch():
    import torch
    from torch.autograd import Variable
    import torch.nn as nn
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt

    # Load the Iris data
    iris_data = pd.read_excel('iris.xls')
    features = iris_data[['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']].values
    labels = iris_data['Species'].values

    # Binarise the species
    binary_labels = []
    for i in range(len(labels)):
        if labels[i] == 'setosa':
            binary_labels.append([1, 0, 0])
        if labels[i] == 'versicolor':
            binary_labels.append([0, 1, 0])
        if labels[i] == 'virginica':
            binary_labels.append([0, 0, 1])

    # Convert labels to a numpy array
    binary_labels = np.array(binary_labels, dtype=int)

    feature_train, feature_test, labels_train, labels_test = train_test_split(features, binary_labels,
                                                                              random_state = 42)

    # Create Tensor variables with data
    feature_train_v = Variable(torch.FloatTensor(feature_train), requires_grad = False)
    labels_train_v = Variable(torch.FloatTensor(labels_train), requires_grad = False)
    feature_test_v = Variable(torch.FloatTensor(feature_test), requires_grad = False)
    labels_test_v = Variable(torch.FloatTensor(labels_test), requires_grad = False)

    # Create a classifier class for the model
    class LinearClassifier(nn.Module):
        def __init__(self):
            super(LinearClassifier, self).__init__()
            self.h_layer = nn.Linear(4, 3)
            self.s_layer = nn.Softmax(dim = -1)

        def forward(self, x):
            y = self.h_layer(x)
            p = self.s_layer(y)
            return p

    model = LinearClassifier()  # Declare the classifier to an object
    loss_fn = nn.BCELoss()  # Calculate the loss
    optimiser = torch.optim.SGD(model.parameters(), lr = 0.01)

    # Fit the training data to the model
    all_losses = []
    for num in range(5000):  # Over 5000 iterations
        pred = model.forward(feature_train_v)  # Predict from the training data
        loss = loss_fn(pred, labels_train_v)  # Calculate the loss
        all_losses.append(loss.data)
        optimiser.zero_grad()  # Zero gradients to not accumulate
        loss.backward()  # Update weights based on loss
        optimiser.step()  # Update optimiser for next iteration

    # Visualise the loss over iteration
    all_losses = np.array(all_losses, dtype = np.float)
    plt.plot(all_losses)
    plt.show()

    # Get predictions for test features
    predicted_values = []
    for num in range(len(feature_test_v)):
        predicted_values.append(model.forward(feature_test_v[num]))

    # Get score against the real labels
    score = 0
    for num in range(len(predicted_values)):
        if np.argmax(labels_test_v[num]) == np.argmax(predicted_values[num].data.numpy()):
            score = score + 1

    # Get accuracy score for prediction
    accuracy = float(score / len(predicted_values)) * 100
    print('Accuracy Score: ' + str(accuracy))


def plotly():
    import plotly
    import plotly.plotly as ply
    import plotly.graph_objs as go

    # Setting up API to load on run
    plotly.tools.set_credentials_file(username='J-Mo', api_key='IqwZEW6x3UJhUYJJW1SL')
    fig = go.Figure()

    # Add a scatter chart
    fig.add_scatter(y = [2, 1, 4, 3])
    # Add a bar chart
    fig.add_bar(y = [1, 4, 3, 2])
    # Add a title
    fig.layout.title = 'Hello FigureWidget'

    ply.plot(fig, filename = 'basic-line', auto_open = True)
