def run():
    import pandas as pd
    import matplotlib.colors as colours
    import matplotlib.pyplot as plt

    # Import the data, read data from Excel file
    data_file = pd.read_excel('abalone.xls')

    # Make dictionary of sexes
    sex_dictionary = {'I': 1, 'M': 2, 'F': 3}
    colors = ['green', 'blue', 'red']

    # Assign dictionary to column
    sex_numeric = [sex_dictionary[s] for s in data_file['Sex']]

    # Using Scatter plot
    plt.scatter(data_file['Height'], data_file['Diameter'], data_file['Gross mass'],
                c = sex_numeric, cmap = colours.ListedColormap(colors))

    plt.title('Abalone Caught')
    plt.xlabel('Height')
    plt.ylabel('Diameter')

    plt.text(10, 5, 'Green: Intersex, Blue: Male, Red: Female', fontdict=None, withdash=False)
    plt.text(10, 0, 'Point diameter is determined by mass', fontdict=None, withdash=False)
    plt.show()
