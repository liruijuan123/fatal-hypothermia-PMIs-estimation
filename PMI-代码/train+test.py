import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler,StandardScaler,LabelEncoder
from sklearn.metrics import classification_report,roc_auc_score,roc_curve,confusion_matrix
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
data = pd.read_csv()
x = data.iloc[:, 1:].values
y = data.iloc[:, 0].values



x_train, x_test,y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=25,stratify=y)

transfer_1=StandardScaler()
x_train=transfer_1.fit_transform(x_train)
x_test=transfer_1.fit_transform(x_test)

KNN=KNeighborsClassifier(n_neighbors=1)

KNN.fit(x_train,y_train)


DT=DecisionTreeClassifier(max_depth=4, min_samples_leaf=2, min_samples_split=8)#

DT.fit(x_train,y_train)

GNB=GaussianNB(var_smoothing=0.027825594022071243)

GNB.fit(x_train,y_train)


RF= RandomForestClassifier(oob_score=True,max_depth=3, min_samples_split=6, n_estimators=200)

RF.fit(x_train,y_train)

LR=LogisticRegression()
LR.fit(x_train ,y_train)
SVM=svm.SVC(C=1, kernel="sigmoid")
SVM.fit(x_train ,y_train)
models = ['KNN', 'DT',"GNB","RF","LR","SVM"]
for i in range(len(models)):
    if models[i] == 'KNN':
        y_pred = KNN.predict(x_test)
    elif models[i] == 'DT':
        y_pred = DT.predict(x_test)

    elif models[i] == 'GNB':
        y_pred = GNB.predict(x_test)

    elif models[i] == 'RF':
        y_pred = RF.predict(x_test)
    elif models[i] == 'LR':
        y_pred = RF.predict(x_test)
    elif models[i] == 'SVM':
        y_pred = RF.predict(x_test)
    else:
        continue
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    labels = ["0h", "4-8h","12-24h", "36-48h"]
    classes = ["0h", "4-8h","12-24h", "36-48h"]
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap='Blues')
    cbar = ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=labels,
           title=f"{models[i]}",
           ylabel='Actual',
           xlabel='Predict')
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.show()
