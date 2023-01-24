#Accuracy reference: https://github.com/kshitijved/Support_Vector_Machine

from sklearn import svm, tree
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
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

    #get input string from user
    input = ['SVM', 'Random Forest', 'K-NN', 'Kernerl-SVM', 'Naive Bayes',  'Decision Tree']

    st.title("Network Traffic Analyser")
    st.info("Web app")

    index = st.selectbox("Select a Machine Learning Algorithm", input)

    if index == 'SVM':
        clf = svm.SVC()
    elif index == 'Random Forest':
        clf = RandomForestClassifier()
    elif index == 'K-NN':
        clf = KNeighborsClassifier(n_neighbors=1)
    elif index == 'Kernerl-SVM':
        clf = svm.SVC(kernel='linear')
    elif index == 'Naive Bayes':
        clf = GaussianNB()
    elif index == 'Decision Tree':
        clf = tree.DecisionTreeClassifier()
    # elif index == 'Logistic Regression':
    #     clf = LogisticRegression()
    else:
        st.write("Invalid input")

    st.set_option('deprecation.showPyplotGlobalUse', False)


    if st.button("Predict"):
        #step4: Train the ML Algo with training data
        clf.fit(x_train, y_train)

        y_pred = clf.predict(x_test)

        #plot the data points wit hgreen and red colour

        plt.scatter(x_test[:, 0], x_test[:, 1], c=y_pred, s=30, cmap=plt.cm.Paired)
        st.pyplot()

        cm = confusion_matrix(y_test, y_pred)
        st.write(cm)
        st.write(accuracy_score(y_test, y_pred))




