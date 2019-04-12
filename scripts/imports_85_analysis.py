def run():
    import pandas as pd
    import numpy as np
    import matplotlib.colors as colours
    import matplotlib.pyplot as plt

    # Import the data, read data from CSV file
    df = pd.read_csv('imports-85.csv')

    # Remove "?" from the data set
    df['price'] = pd.to_numeric(df['price'].replace('?', np.nan))
    df['city-mpg'] = pd.to_numeric(df['city-mpg'].replace('?', np.nan))
    df['horsepower'] = pd.to_numeric(df['horsepower'].replace('?', np.nan))

    # Make dictionary of fuel type
    fuel_type_dictionary = {'diesel': 1, 'gas': 2}
    colors = ['black', 'blue']

    # Assign dictionary to column
    fuel_type_numeric = [fuel_type_dictionary[s] for s in df['fuel-type']]

    # Using scatter plot
    plt.scatter(df['city-mpg'], df['horsepower'], df['price']/100,
                c = fuel_type_numeric, cmap = colours.ListedColormap(colors))

    plt.title('Imports \'85')
    plt.xlabel('city-mpg')
    plt.ylabel('horsepower')

    plt.text(30, 250, 'Blue: gas, Black: diesel', fontdict=None, withdash=False)
    plt.text(30, 235, 'Point diameter is determined by price', fontdict=None, withdash=False)
    plt.show()

    # Using parallel coordinates
    pd.plotting.parallel_coordinates(
        df[['symboling', 'price', 'highway-mpg', 'city-mpg', 'length']], 'symboling')

    plt.show()
