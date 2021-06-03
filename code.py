#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
import numpy as np

from sklearn import preprocessing

from sklearn.feature_selection import mutual_info_classif

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import validation_curve

import matplotlib
import matplotlib.pyplot as plt
large = 22; med = 13; small = 12
params = {'axes.titlesize': large,
          'legend.fontsize': med,
          'figure.figsize': (16, 10),
          'axes.labelsize': med,
          'axes.titlesize': med,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': large}
plt.rcParams.update(params)
plt.style.use('seaborn-whitegrid')


# In[164]:


class Classifier:
    def __init__(self, df, targetName):
        self.df = df
        self.classifiers = {}
        self.predict = {}
        self.K = 13
        self.D = 3
        self.NUMOFCLASSIFIERS = 100
        self.target = df[targetName]
        self.features = df.drop(targetName, 1)
        self.labelEncoder = preprocessing.LabelEncoder()
#         self.scaler = preprocessing.MinMaxScaler(feature_range=(1, 20))
        self.scaler = preprocessing.StandardScaler()

    def oneHotEncoding(self, featureName):
        dfOneHot = pd.get_dummies(self.df[featureName])
        self.df = pd.concat([self.df, dfOneHot], axis=1)
        self.df = self.df.drop(featureName, axis=1)

    def preprocess(self, columnsToRemove):
        columns = list(self.df.columns)
        columns[0] = 'index'
        self.df.columns = columns
        self.df = self.df[(self.df[columns] > 0).all(axis = 1)]
        self.df['Country'] = self.labelEncoder.fit_transform(self.df['Country'])
#         self.oneHotEncoding('Country')
        self.df['Date'] = pd.to_datetime(self.df['Date'], infer_datetime_format = True)
        self.df['Date'] = [str(np.datetime64(d, 'M')) for d in self.df['Date'].values]
        self.df['Date'] = self.labelEncoder.fit_transform(self.df['Date'])
#         self.df['Year'], self.df['Month'], self.df['Day'] = self.df['Date'].dt.year, self.df['Date'].dt.month, self.df['Date'].dt.dayofweek
#         columns = list(self.df.columns)
#         columns.remove('Date')
        self.df['Is Back'] = self.labelEncoder.fit_transform(self.df['Is Back'])
        self.df['Total Quantity'] = self.scaler.fit_transform(self.df['Total Quantity'].values.reshape(
            self.df['Total Quantity'].shape[0], 1))
        self.df['Total Price'] = self.scaler.fit_transform(self.df['Total Price'].values.reshape(
            self.df['Total Price'].shape[0], 1))
        self.df['Purchase Count'] = self.scaler.fit_transform(self.df['Purchase Count'].values.reshape(
            self.df['Purchase Count'].shape[0], 1))
        for c in columnsToRemove:
            columns.remove(c)
        self.features = self.df[columns]
        self.target = self.df[['Is Back']]
        self.target = self.target.values.reshape(self.target.shape[0],)

    def getInfoGain(self):
        infoGain = mutual_info_classif(self.features, self.target, random_state = 178, n_neighbors = 1,
                                       discrete_features = [True, False, False, True, True,  False])
        fig = plt.figure(figsize = (10, 5))
        plt.plot(self.features.columns, infoGain, color = 'teal')
        plt.xlabel('Feature')
        plt.ylabel('Information Gain')
        plt.show()
        print(dict(zip(self.features.columns, infoGain)))

    def splitData(self, testSize = 0.2):
        self.features = self.features.drop('Customer ID', 1)
        self.featuresTrain, self.featuresTest, self.targetTrain, self.targetTest = train_test_split(
            self.features, self.target, test_size = testSize, random_state = RANDOMSTATE, stratify = self.target)

    def getBestK(self):
        knn = KNeighborsClassifier(n_neighbors = self.K)
        KList = {'n_neighbors': np.arange(1, 30)}
        knnSearch = GridSearchCV(knn, KList, cv = 5)
        knnSearch.fit(self.featuresTrain, self.targetTrain)
        return knnSearch.best_params_['n_neighbors']

    def predictKNN(self):
        self.K = self.getBestK()
        self.classifiers['KNN'] = KNeighborsClassifier(n_neighbors = self.K)
        self.classifiers['KNN'].fit(self.featuresTrain, self.targetTrain)
        self.predict['KNN'] = self.classifiers['KNN'].predict(self.featuresTest)

    def predictLR(self):
        self.classifiers['LR'] = LogisticRegression(max_iter = 1200000)
        self.classifiers['LR'].fit(self.featuresTrain, self.targetTrain)
        self.predict['LR'] = self.classifiers['LR'].predict(self.featuresTest)

    def getBestD(self):
        dt = DecisionTreeClassifier(max_depth = self.D)
        DList = {'max_depth': np.arange(3, 30)}
        dtSearch = GridSearchCV(dt, DList, cv = 5)
        dtSearch.fit(self.featuresTrain, self.targetTrain)
        self.D = dtSearch.best_params_['max_depth']

    def predictDT(self):
        self.classifiers['DT'] = DecisionTreeClassifier(max_depth = self.D)
        self.classifiers['DT'].fit(self.featuresTrain, self.targetTrain)
        self.predict['DT'] = self.classifiers['DT'].predict(self.featuresTest)

    def getBestNEstimatorsKNN(self):
        baggingKNN = BaggingClassifier(self.classifiers['KNN'],
                                                            n_estimators = self.NUMOFCLASSIFIERS,
                                                            max_features = 0.5, max_samples = 0.5,
                                                            random_state = RANDOMSTATE, n_jobs = -1)
        KList = {'n_estimators': np.arange(2, 100)}
        model = GridSearchCV(baggingKNN, KList, cv = 5)
        model.fit(self.featuresTrain, self.targetTrain)
        return model.best_params_['n_estimators']

    def predictBaggingKNN(self):
        self.classifiers['Bagging KNN'] = BaggingClassifier(self.classifiers['KNN'],
                                                            n_estimators = self.NUMOFCLASSIFIERS,
                                                            max_features = 0.5, max_samples = 0.5,
                                                            random_state = RANDOMSTATE, n_jobs = -1)
        self.classifiers['Bagging KNN'].fit(self.featuresTrain, self.targetTrain)
        self.predict['Bagging KNN'] = self.classifiers['Bagging KNN'].predict(self.featuresTest)

    def predictBaggingDT(self):
        self.classifiers['Bagging DT'] = BaggingClassifier(self.classifiers['DT'],
                                                           n_estimators = self.NUMOFCLASSIFIERS,
                                                           max_features = 0.5, max_samples = 0.5,
                                                           random_state = RANDOMSTATE, n_jobs = -1)
        self.classifiers['Bagging DT'].fit(self.featuresTrain, self.targetTrain)
        self.predict['Bagging DT'] = self.classifiers['Bagging DT'].predict(self.featuresTest)

    def predictRandomForest(self):
        self.classifiers['Random Forest'] = RandomForestClassifier(bootstrap = True, random_state = RANDOMSTATE,
            max_depth = self.D, max_features = 0.5, max_samples = 0.5,
            n_estimators = self.NUMOFCLASSIFIERS, n_jobs = -1)
        self.classifiers['Random Forest'].fit(self.featuresTrain, self.targetTrain)
        self.predict['Random Forest'] = self.classifiers['Random Forest'].predict(self.featuresTest)

    def predictHardVoting(self):
        self.classifiers['Hard Voting'] = VotingClassifier(estimators=[('KNN', self.classifiers['KNN']),
                                                                       ('DT', self.classifiers['DT']),
                                                                       ('LR', self.classifiers['LR'])],
                                                           voting='hard')
        self.classifiers['Hard Voting'].fit(self.featuresTrain, self.targetTrain)
        self.predict['Hard Voting'] = self.classifiers['Hard Voting'].predict(self.featuresTest)

    def predictAll(self):
        self.splitData()
        self.predictKNN()
        self.predictDT()
        self.predictLR()
        self.predictBaggingKNN()
        self.predictBaggingDT()
        self.predictRandomForest()
        self.predictHardVoting()

    def plotValidationCurve(self, classifier, paramName, paramRange, scoringParam, features, target, NUMOFTHREADS = -1):
        trainScores, testScores = validation_curve(classifier, features, target,
                                                     param_name = paramName, param_range = paramRange, cv = 5,
                                                     scoring = scoringParam, n_jobs = NUMOFTHREADS)
        trainMean = np.mean(trainScores, axis = 1)
        testMean = np.mean(testScores, axis = 1)
        plt.plot(paramRange, trainMean, label = "Training Score", color = "teal")
        plt.plot(paramRange, testMean, label = "Test Score", color = "purple")
        plt.xlabel(paramName)
        plt.ylabel(scoringParam)
        plt.title("Validation Curve")
        plt.legend(loc="best")

    def plotValidationCurves(self, classifierName, paramName, paramRange, NUMOFTHREADS = -1):
        fig = plt.figure(figsize = (20, 4))
        fig.suptitle(classifierName)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=0.8, wspace=0.2)
        plt.subplot(1, 3, 1)
        self.plotValidationCurve(self.classifiers[classifierName], paramName, paramRange,
                                 "accuracy", self.features, self.target, NUMOFTHREADS)
        plt.subplot(1, 3, 2)
        self.plotValidationCurve(self.classifiers[classifierName], paramName, paramRange,
                                 "precision", self.features, self.target, NUMOFTHREADS)
        plt.subplot(1, 3, 3)
        self.plotValidationCurve(self.classifiers[classifierName], paramName, paramRange,
                                 "recall", self.features, self.target, NUMOFTHREADS)
        plt.show()

    def getMetrics(self):
        info = pd.DataFrame(columns = ['Classifier', 'Recall', 'Precision', 'Accuracy'])
        for c in list(self.classifiers.keys()):
            accuracy = accuracy_score(self.targetTest, self.predict[c]).round(2)
            precision = precision_score(self.targetTest, self.predict[c], average="weighted").round(2)
            recall = recall_score(self.targetTest, self.predict[c], average="weighted").round(2)
            row = {'Classifier': c, 'Recall': recall, 'Precision': precision, 'Accuracy': accuracy}
            info = info.append(row, ignore_index=True, sort=False)
        return info

    def getReport(self):
        for c in list(self.classifiers.keys()):
            print(c)
            print(classification_report(self.targetTest, self.predict[c]))
            print('______________________________________________________________\n\n')



# In[165]:


FILENAME = 'data.csv'
RANDOMSTATE = 289
pd.options.mode.chained_assignment = None
df = pd.read_csv(FILENAME, delimiter = ',')
classifier = Classifier(df, 'Is Back')
classifier.preprocess(['Is Back', 'index'])


# In[166]:


classifier.getInfoGain()


# In[167]:


classifier.predictAll()


# In[148]:


print(classifier.getMetrics())


# In[156]:


classifier.getReport()


# In[86]:


neighborsRange = list(range(1, 30, 1));
classifier.plotValidationCurves('KNN', 'n_neighbors', neighborsRange)


# In[85]:


depthRange = list(range(1, 30, 1));
classifier.plotValidationCurves('DT', 'max_depth', depthRange)


# In[320]:


print('Similarity in DT and Bagging DT predictions: ', pd.DataFrame(classifier.predict['DT']).eq(pd.DataFrame(classifier.predict['Bagging DT']).values).mean()[0].round(2), '%')
print('Similarity in DT and Random Forest predictions: ', pd.DataFrame(classifier.predict['DT']).eq(pd.DataFrame(classifier.predict['Random Forest']).values).mean()[0].round(2), '%')
print('Similarity in KNN and Bagging KNN predictions: ', pd.DataFrame(classifier.predict['KNN']).eq(pd.DataFrame(classifier.predict['Bagging KNN']).values).mean()[0].round(2), '%')
print('Similarity in KNN and DT predictions: ', pd.DataFrame(classifier.predict['KNN']).eq(pd.DataFrame(classifier.predict['DT']).values).mean()[0].round(2), '%')
print('Similarity in KNN and LR predictions: ', pd.DataFrame(classifier.predict['KNN']).eq(pd.DataFrame(classifier.predict['LR']).values).mean()[0].round(2), '%')
print('Similarity in LR and DT predictions: ', pd.DataFrame(classifier.predict['DT']).eq(pd.DataFrame(classifier.predict['LR']).values).mean()[0].round(2), '%')
print('Similarity in LR and Hard Voting predictions: ', pd.DataFrame(classifier.predict['Hard Voting']).eq(pd.DataFrame(classifier.predict['LR']).values).mean()[0].round(2), '%')
print('Similarity in DT and Hard Voting predictions: ', pd.DataFrame(classifier.predict['DT']).eq(pd.DataFrame(classifier.predict['Hard Voting']).values).mean()[0].round(2), '%')
print('Similarity in KNN and Hard Voting predictions: ', pd.DataFrame(classifier.predict['KNN']).eq(pd.DataFrame(classifier.predict['Hard Voting']).values).mean()[0].round(2), '%')

# In[99]:


# estimatorsRange = list(range(1, 200, 1));
# classifier.plotValidationCurves('Bagging KNN', 'n_estimators', estimatorsRange)


# In[96]:


# estimatorsRange = list(range(1, 200, 1));
# classifier.plotValidationCurves('Random Forest', 'n_estimators', estimatorsRange)


# In[149]:


# depthRange = list(range(1, 30, 1));
# classifier.plotValidationCurves('Random Forest', 'max_depth', depthRange)


# In[98]:


# estimatorsRange = list(range(1, 200, 1));
# classifier.plotValidationCurves('Bagging DT', 'n_estimators', estimatorsRange)
