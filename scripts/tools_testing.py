import torch

def run():
    # Creating a new Tensor
    w = torch.Tensor([1.0, 2.0])

    # Tensors are integrated with Python
    w = w + 2
    print(w)


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
