# Import libraries
import matplotlib.pyplot as plt
import pandas as pd


def run():
    # Import libraries
    import matplotlib.pyplot as plt
    import matplotlib.colors as colours

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

    from sklearn.tree import export_graphviz
    # Export as dot file
    export_graphviz(my_tree, out_file='tree.dot',
                    feature_names=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'],
                    class_names=['Setosa', 'Versicolor', 'Virginica'],
                    rounded=True, proportion=False,
                    precision=2, filled=True)

    # Convert to png
    import pydot
    (graph,) = pydot.graph_from_dot_file('tree.dot')
    graph.write_png('tree.png')

    # Display in python
    import matplotlib.pyplot as plt
    plt.figure(figsize=(14, 18))
    plt.imshow(plt.imread('tree.png'))
    plt.axis('off')
    plt.show()





def pre_processing():
    # Import libraries
    from sklearn import preprocessing

    # Import the data and read from Excel file
    df = pd.read_excel('iris.xls')

    # Isolate the attribute
    data = df['Sepal.Length']

    # Cut up the data by width & depth
    bin_equi_width = pd.cut(data, 10)
    bin_equi_depth = pd.qcut(data, 10)

    # print the results
    # print("Equi-Width:")
    # print(bin_equi_width.value_counts())
    # print()
    # print("Equi-Depth:")
    # print(bin_equi_depth.value_counts())

    # Format and listify equi-width list
    bin_equi_width_list = []
    for i in range(bin_equi_width.size):
        left_item = "{0:.2f}".format(bin_equi_width[i].left)
        right_item = "{0:.2f}".format(bin_equi_width[i].right)
        bin_equi_width_list.append([left_item, right_item])

    # Format and listify equi-depth list
    bin_equi_depth_list = []
    for i in range(bin_equi_depth.size):
        left_item = "{0:.2f}".format(bin_equi_depth[i].left)
        right_item = "{0:.2f}".format(bin_equi_depth[i].right)
        bin_equi_depth_list.append([left_item, right_item])

    # Fit the min-max normalised list
    petal_length_df = df['Petal.Length'].values.reshape(-1, 1)
    min_max_scaled = preprocessing.MinMaxScaler().fit_transform(petal_length_df)
    normalise_min_max_list = pd.DataFrame(min_max_scaled).values.flatten()

    # Fit the z-score normalised list
    petal_length_df = df['Petal.Length'].values.reshape(-1, 1)
    z_score_scaled = preprocessing.StandardScaler().fit_transform(petal_length_df)
    normalise_z_score_list = pd.DataFrame(z_score_scaled).values.flatten()

    # Binarise the species
    species_df = df['Species'].values
    binarised = preprocessing.LabelBinarizer().fit_transform(species_df)

    # Format the binarised species into three columns
    binarised_setosa_list = []
    binarised_versicolor_list = []
    binarised_virginica_list = []
    for i in range(len(binarised)):
        binarised_setosa_list.append(binarised[i][0])
        binarised_versicolor_list.append(binarised[i][1])
        binarised_virginica_list.append(binarised[i][2])

    # Discretise the petal width into categories
    petal_width_df = df['Petal.Width'].values
    discretised = []
    for i in range(petal_width_df.size):
        item = petal_width_df[i]
        if item < 0.2:
            discretised.append('Short')
        elif item <= 0.3:
            discretised.append('Average')
        elif item >= 0.4:
            discretised.append('Long')
        else:
            discretised.append('Extra Short')

    # Create the combined data frame for output
    df = pd.DataFrame({
        'Sepal.Length': df['Sepal.Length'],
        'Sepal.Width': df['Sepal.Width'],
        'Petal.Length': df['Petal.Length'],
        'Petal.Width': df['Petal.Width'],
        'Species': df['Species'],
        'Equi-Width': bin_equi_width_list,
        'Equi-Depth': bin_equi_depth_list,
        'Min-Max': normalise_min_max_list,
        'Z-Score': normalise_z_score_list,
        'Binarised-Setosa': binarised_setosa_list,
        'Binarised-Versicolor': binarised_versicolor_list,
        'Binarised-virginica': binarised_virginica_list,
        'Discretised': discretised
    })

    # Write and save the data to an excel document
    writer = pd.ExcelWriter('output.xls', engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    writer.save()


def clustering():
    # Import libraries
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans

    # Import the data and read from Excel file
    df = pd.read_excel('iris.xls')

    x_data = df[['Petal.Length']]
    y_data = df[['Petal.Width']]

    kmeans = KMeans(n_clusters=3)
    kmeansoutput = kmeans.fit(y_data)

    pca = PCA(n_components=1).fit(y_data)
    pca_d = pca.transform(x_data)
    pca_c = pca.transform(y_data)

    plt.figure('3 Cluster K-Means')
    plt.scatter(pca_c[:, 0], pca_d[:, 0], c=kmeansoutput.labels_)
    plt.xlabel('Sepal.Length')
    plt.ylabel('Sepal.Width')
    plt.title('K-Means Clustering for Iris')
    plt.show()
