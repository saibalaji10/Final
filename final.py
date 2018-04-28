import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt

from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import naive_bayes
from sklearn.svm import SVC
from sklearn import tree

aDTree = []
aRbfSvm = []
aNearestN = []
aRandomForest = []
npNaiveBayes = []
# Function will run each of the classifiers using cross fold validation
def runClassifiers(data, target):
    
    dTree = tree.DecisionTreeClassifier()
    scores = model_selection.cross_val_score(dTree, data, target, cv=10)
    aDTree.append(scores.mean())
    print "Tree : ", scores.mean()
    
    rbfSvm = SVC()
    scores = model_selection.cross_val_score(rbfSvm, data, target, cv=10)
    aRbfSvm.append(scores.mean())
    print "\nSVM : ", scores.mean()
    
    nearestN = KNeighborsClassifier()
    scores = model_selection.cross_val_score(nearestN, data, target, cv=10)
    aNearestN.append(scores.mean())
    print "\nNNeighbour : ", scores.mean()
    
    randomForest = RandomForestClassifier()
    scores = model_selection.cross_val_score(randomForest, data, target, cv=10)
    aRandomForest.append(scores.mean())
    print "RForest : ",scores.mean()
    
    nBayes = naive_bayes.GaussianNB()
    scores = model_selection.cross_val_score(nBayes, data, target, cv=10)
    npNaiveBayes.append(scores.mean())
    print "Naive Bayes : ",scores.mean()


df = pd.read_csv('student.csv')

averageFinalGrade = round(df[["G3"]].mean(),0)
gradeRow = pd.Series(df["G3"])

#gradeRow[gradeRow < averageFinalGrade] = 0
#averageFinalGrade = round(df[["G3"]].mean(),0)
#gradeRow = pd.Series(df["G3"])
#gradeRow[gradeRow < averageFinalGrade] = 0
#gradeRow[gradeRow >= averageFinalGrade] = 1

#5 Level Classification
gradeRow[gradeRow < 5]  = 0   #10,16,14,12,10
gradeRow[gradeRow >= 20] = 1
gradeRow[gradeRow >= 15] = 2
gradeRow[gradeRow >= 10] = 3
gradeRow[gradeRow >= 5] = 4

df["G3"] = gradeRow

df['school'] = df['school'].map({'GP': 0, 'MS':1}).astype(int)
df['sex'] = df['sex'].map({'M': 0, 'F':1}).astype(int)
df['Pstatus'] = df['Pstatus'].map({'A': 0, 'T':1}).astype(int)
df['higher'] = df['higher'].map({'no': 0, 'yes':1}).astype(int)
df['internet'] = df['internet'].map({'no': 0, 'yes':1}).astype(int)

df['address'] = df['address'].map({'U': 0, 'R':1}).astype(int)
df['famsize'] = df['famsize'].map({'LE3': 0, 'GT3':1}).astype(int)
df['Mjob'] = df['Mjob'].map({'teacher': 0, 'health':1, 'services':2, 'at_home':3, 'other':4}).astype(int)
df['Fjob'] = df['Fjob'].map({'teacher': 0, 'health':1, 'services':2, 'at_home':3, 'other':4}).astype(int)
df['reason'] = df['reason'].map({'home': 0, 'reputation':1, 'course':2, 'other':3}).astype(int)
df['guardian'] = df['guardian'].map({'mother': 0, 'father':1, 'other':2}).astype(int)

df['schoolsup'] = df['schoolsup'].map({'yes': 0, 'no':1}).astype(int)
df['famsup'] = df['famsup'].map({'yes': 0, 'no':1}).astype(int)
df['paid'] = df['paid'].map({'yes': 0, 'no':1}).astype(int)
df['activities'] = df['activities'].map({'yes': 0, 'no':1}).astype(int)
df['nursery'] = df['nursery'].map({'yes': 0, 'no':1}).astype(int)
df['romantic'] = df['romantic'].map({'yes': 0, 'no':1}).astype(int)

target = df["G3"]
data = df.drop(["G3"], axis= 1)
#data = data.drop(["G2"], axis= 1)#Drop G2
#data = data.drop(["G1"], axis= 1)#Drop G1
print data.shape ,target.shape
runClassifiers(data, target)
