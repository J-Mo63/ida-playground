def run():
    import pandas as pd
    import matplotlib.colors as colours
    import matplotlib.pyplot as plt

    # Import the data, read data from Excel file
    df = pd.read_excel('iris.xls')

    # Explore and visualize the data before building the model
    # Visualize the data - Displaying Scatter Plot and Matrix
    # Creating a dictionary for string values of class
    species_dictionary = {'setosa': 1, 'versicolor': 2, 'virginica': 3}

    # Create class in numeric value
    species_numeric = [species_dictionary[s] for s in df['Species']]

    # "Color manager" - Define colors
    colors = ['red', 'green', 'blue']

    # Using Scatter plot
    plt.scatter(df['Sepal.Width'], df['Sepal.Length'], c = species_numeric,
                cmap = colours.ListedColormap(colors))

    plt.xlabel('SepalWidth')
    plt.ylabel('sepalLength')
    plt.show()

    # Using Scatter Matrix
    import seaborn as sns
    sns.set(style = "ticks")
    sns.pairplot(df, hue = "Species")
    # Clean data set

    # Data Splitting (Create Train/Text Data)
    train = df.sample(frac = 0.8, random_state = 1)
    test = df.drop(train.index)

    # Feature selection prepare training and testing features and labels
    # sepal_length = train['Sepal.Length']
    # sepal_width = train['Sepal.Width']
    # petal_length = train['Petal.Length']
    # petal_width = train['Petal.Width']

    # Training features
    features_train = train[['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']]
    # Training Labels
    features_test = test[['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']]

    # Testing features
    labels_training = train['Species']
    # Testing Labels
    labels_test = test[['Species']]

    # Build learning model using a decision tree classifier
    from sklearn import tree
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score

    my_tree = tree.DecisionTreeClassifier()
    # Train the classifier
    my_tree_model = my_tree.fit(features_train, labels_training)
    # Make prediction
    # Test the classifier and get he predicted results
    prediction = my_tree_model.predict(features_test)
    # Measure the performance
    # Confusion matrix
    cm = confusion_matrix(labels_test, prediction)

    # Get accuracy score, formatted to a percentage
    accuracy = accuracy_score(labels_test, prediction)
    print("{:.2%}".format(accuracy))

    # Display the plot
    plt.imshow(cm, cmap = 'binary')
    # Tune the classifier parameter
