
pip install -U scikit-learn
from sklearn import svm, tree
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import streamlit as st


#change the streamlit background


#step1: Load the data in numpy array
#step1: upload data file in streamlit

data = st.file_uploader("Choose a CSV file", type="csv")



if data is not None:
    data = np.genfromtxt(data, delimiter=',')

    #step2: Split the data to training & test data. Test-size is 0.25(25%) of data
    X = data[:, 0:3]
    y = data[:, 3]
    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state = 0,     test_size = 0.25)#clf = svm.SVC()

    # sc = StandardScaler()
    # x_train = sc.fit_transform(x_train)
    # x_test = sc.transform(x_test)

    #get input string from user
    input = ['SVM', 'Random Forest', 'K-NN', 'Kernerl-SVM', 'Naive Bayes',  'Decision Tree', 'Logistic Regression']

    st.title("Network Traffic Analyser")
    st.info("Web app")

    index = st.selectbox("Select a Stock Exchange", input)

    if index == 'SVM':
        clf = svm.SVC()
    elif index == 'Random Forest':
        clf = RandomForestClassifier()
    elif index == 'K-NN':
        clf = KNeighborsClassifier()
    elif index == 'Kernerl-SVM':
        clf = svm.SVC(kernel='linear')
    elif index == 'Naive Bayes':
        clf = GaussianNB()
    elif index == 'Decision Tree':
        clf = tree.DecisionTreeClassifier()
    elif index == 'Logistic Regression':
        clf = LogisticRegression()
    else:
        st.write("Invalid input")

    st.set_option('deprecation.showPyplotGlobalUse', False)


    #plot the data points
    def plot(x_test):
        plt.scatter(x_test[:, 0], x_test[:, 1], c=y_pred, cmap='rainbow')

        # Add a legend to indicate which class each color corresponds to
        plt.legend(['Class 1', 'Class 2', 'CLASS 3'])

        # Add a title and axis labels to the plot
        plt.title("Final Classified Data Points")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")

        # Show the plot
        plt.show()
        st.pyplot()

    # Create a scatter plot of the final classified data points
    


    if st.button("Predict"):
        #step4: Train the ML Algo with training data
        clf.fit(x_train, y_train)

        #step5: Pass the test data for classify or predict
        y_pred = clf.predict(x_test)

        # knnn(x_train, y_train, clf, sc)
        plot(x_test)
        cm = confusion_matrix(y_test, y_pred)
        st.write(cm)
        st.write(accuracy_score(y_test, y_pred))




