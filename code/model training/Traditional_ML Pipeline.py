import pandas as pd
import numpy as np
import sklearn.preprocessing as sp
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, validation_curve
from statsmodels.api import OLS
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
desired_width=500
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns',20)

data = pd.read_csv("Audio_Features_Extraction.csv")

#InputColumns = [chroma_stft_mean,chroma_stft_var,chroma_cens_mean,chroma_cens_var,chroma_cqt_mean,chroma_cqt_var,melspectrogram_mean,melspectrogram_var,mfcc_mean,mfcc_var,rms_mean,rms_var,spec_bandwith_mean,spec_bandwith_var,spec_centroid_mean,spec_centroid_var,spec_contrast_mean,spec_contrast_var,spec_flatness_mean,spec_flatness_var,spec_rolloff_mean,spec_rolloff_var,tonnetz_mean,tonnetz_var,crossing_rate_mean,crossing_rate_var]


# Distribution of the Dataset ..
# print(data.info())
# print(data.describe(include = "all"))

data = data.drop(labels="Unnamed: 0", axis=1)
labels = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
d = dict(zip(labels,range(1,11)))
data['labels'] = data['labels'].map(d, na_action='ignore')
# print(data.describe(include = "all"))
mean_data = data.loc[:, 'chroma_stft_mean':'crossing_rate_mean':2]
mean_data["labels"] = data["labels"]
mean_data["labels"] = data["labels"].values
var_data = data.loc[:, 'chroma_stft_var':'crossing_rate_var':2]
var_data["labels"] = data["labels"]
var_data["labels"] = data["labels"].values
# let's see the distribution of Different Features according to mean and variance ..

# plt.subplots(figsize = (15,10))
# fig = sns.PairGrid(mean_data)
# fig.map_diag(sns.kdeplot)
# fig.map_offdiag(sns.kdeplot, color = 'b')
# plt.title("Different Feature Mean")
# print(plt.show())
#
# # plt.subplots(figsize = (15,10))
# fig = sns.PairGrid(var_data)
# fig.map_diag(sns.kdeplot)
# fig.map_offdiag(sns.kdeplot, color = 'b')
# plt.title("Different Feature Variance")
# print(plt.show())

# fig, ax =sns.pairplot(mean_data, hue='labels', plot_kws={'alpha':0.1})
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# print(plt.show())

# fig, ax =sns.pairplot(var_data, hue='labels')
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# print(plt.show())

## PDF's of Mean ..
meancol = list(mean_data)[:-1]
varcol = list(var_data.columns)[:-1]
#
# fig, axes = plt.subplots(3, 4, figsize=(24, 15))
# fig.suptitle('PDF of mean(Features)')
#
# sns.histplot(ax=axes[0, 0], x= mean_data[meancol[0]], kde = True)
# sns.histplot(ax=axes[0, 1], x= mean_data[meancol[1]], kde = True)
# sns.histplot(ax=axes[0, 2], x= mean_data[meancol[2]], kde = True)
# sns.histplot(ax=axes[0, 3], x= mean_data[meancol[3]], kde = True)
#
# sns.histplot(ax=axes[1, 0], x= mean_data[meancol[4]], kde = True)
# sns.histplot(ax=axes[1, 1], x= mean_data[meancol[5]], kde = True)
# sns.histplot(ax=axes[1, 2], x= mean_data[meancol[6]], kde = True)
# sns.histplot(ax=axes[1, 3], x= mean_data[meancol[7]], kde = True)
#
# sns.histplot(ax=axes[2, 0], x= mean_data[meancol[8]], kde = True)
# sns.histplot(ax=axes[2, 1], x= mean_data[meancol[9]], kde = True)
# sns.histplot(ax=axes[2, 2], x= mean_data[meancol[10]], kde = True)
# sns.histplot(ax=axes[2, 3], x= mean_data[meancol[11]], kde = True)

# print(plt.show())
# sns.histplot(mean_data[meancol[12]], kde = True)
# print(plt.show())

# fig, axes = plt.subplots(3, 4, figsize=(24, 15))
# fig.suptitle('PDF of Variance(Features)')
# sns.histplot(ax=axes[0, 0], x= var_data[varcol[0]], kde = True)
# sns.histplot(ax=axes[0, 1], x= var_data[varcol[1]], kde = True)
# sns.histplot(ax=axes[0, 2], x= var_data[varcol[2]], kde = True)
# sns.histplot(ax=axes[0, 3], x= var_data[varcol[3]], kde = True)
#
# sns.histplot(ax=axes[1, 0], x= var_data[varcol[4]], kde = True)
# sns.histplot(ax=axes[1, 1], x= var_data[varcol[5]], kde = True)
# sns.histplot(ax=axes[1, 2], x= var_data[varcol[6]], kde = True)
# sns.histplot(ax=axes[1, 3], x= var_data[varcol[7]], kde = True)
#
# sns.histplot(ax=axes[2, 0], x= var_data[varcol[8]], kde = True)
# sns.histplot(ax=axes[2, 1], x= var_data[varcol[9]], kde = True)
# sns.histplot(ax=axes[2, 2], x= var_data[varcol[10]], kde = True)
# sns.histplot(ax=axes[2, 3], x= var_data[varcol[11]], kde = True)
# print(plt.show())
#
# sns.histplot(var_data[varcol[12]], kde = True)
# print(plt.show())



### //////////////////////////////////////// 3D visualization  ///////////////////////////////
# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# x = data[meancol[0]]
# y = data[varcol[0]]
# z = data[meancol[1]]
#
# ax.set_xlabel(meancol[0])
# ax.set_ylabel(varcol[0])
# ax.set_zlabel(meancol[1])
#
# ax.scatter(x, y, z)
#
# print(plt.show())
### //////////////////////////////////////////////////////////////////////////////////////////

inputs = data.drop('labels', axis=1)
# scale = sp.StandardScaler()
#
# # inputs['crossing_rate_var'], inputs['spec_flatness_mean'], inputs['spec_flatness_var'] = np.log(inputs['crossing_rate_var']), np.log(inputs['spec_flatness_mean']), np.log(inputs['spec_flatness_var'])
# sns.histplot(var_data[varcol[12]], kde = True)
# print(plt.show())
# scale = sp.MinMaxScaler()

# scale_inputs = scale.fit_transform(inputs)
# print(scale_inputs)
Targets = data['labels']

# x_train, x_test, y_train, y_test = train_test_split(inputs, Targets, test_size=0.2, random_state=1, shuffle=True)
# mod = OLS(y_train, x_train )
# f = mod.fit()
# print(f.summary())

columns = ['melspectrogram_var', 'mfcc_var', 'spec_flatness_mean', 'spec_flatness_var','tonnetz_var' , 'chroma_stft_mean','spec_bandwith_var', 'spec_rolloff_mean', 'tonnetz_mean', 'crossing_rate_var', 'chroma_cqt_mean', 'chroma_stft_var']
inputs = inputs.drop(columns, axis = 1)
x_train, x_test, y_train, y_test = train_test_split(inputs, Targets, test_size=0.2, random_state=1, shuffle=True)
mod = OLS(y_train, x_train )
f = mod.fit()
print(f.summary())
print(" ")

scale = sp.StandardScaler()
scaled_inputs = scale.fit_transform(inputs)
x_train, x_test, y_train, y_test = train_test_split(scaled_inputs, Targets, test_size=0.2, random_state=1, shuffle=True)
print("FITTING & TESTING DIFFERENT CLASSIFICATION MODELS ( with scaled data )")

## Random Forest Classifier
model = RandomForestClassifier(n_estimators = 200)
model.fit(x_train, y_train)
prediction = model.predict(x_test)
print("Random Forest : ", metrics.accuracy_score(prediction, y_test)*100)

# Decision Tree Classifier
model = DecisionTreeClassifier()
model.fit(x_train, y_train)
prediction = model.predict(x_test)
print("Decision Tree : ", metrics.accuracy_score(prediction, y_test)*100)

# SVM ( Support Vector Machine )
model = svm.SVC()
model.fit(x_train, y_train)
prediction = model.predict(x_test)
print("Support Vector Machine : ", metrics.accuracy_score(prediction, y_test)*100)

# KNN (K Nearest Neighbour Classifier )
model = KNeighborsClassifier(n_neighbors=5)
model.fit(x_train, y_train)
prediction = model.predict(x_test)
print("K Nearest Neighbours : ", metrics.accuracy_score(prediction, y_test)*100)

# Gaussian Naive Bayes Algorithm
model = GaussianNB()
model.fit(x_train, y_train)
prediction = model.predict(x_test)
print("Naive Bayes Algorithm : ", metrics.accuracy_score(prediction, y_test)*100)

## Two best models are SVM and Random Forest Classifier with 61% and 64.5% respectively
## Hyper-parameter tuning these two models ..

### Random Forest Classifier .. ( Tuning Process )
#HYPER-PARAMS for Random Forest Classifier
#  Number of trees in random forest
# n_estimators = [int(x) for x in np.linspace(start=200, stop=300, num=10)]
# #  Number of Features to consider at every Split
# max_features = ['auto','sqrt']
# #   Maximum number of levels in tree
# max_depth = [4,8,10]
# #  min number of samples required to split a node
# min_samples_split = [2,5]
# # min num of samples required ateach leaf node
# min_samples_leaf = [1,2]
# # method of selecting Samples for training each tree
# bootstrap = [True,False]
#
# ### creating Param_grid ..
# param_grid = { 'n_estimators' : n_estimators,
#                'max_features' : max_features,
#                'max_depth' : max_depth,
#                'min_samples_split' : min_samples_split,
#                'min_samples_leaf' : min_samples_leaf,
#
#                'bootstrap' : bootstrap}
# print(param_grid)
#
# rf_model = RandomForestClassifier()
# rf_grid_model = GridSearchCV(estimator=rf_model, param_grid = param_grid, cv=3, verbose=2, n_jobs = -1, return_train_score = True, scoring = 'f1_macro')
# clf = rf_grid_model.fit(x_train, y_train)
#
# test_scores = clf.cv_results_['mean_test_score']
# train_scores = clf.cv_results_['mean_train_score']
#
# plt.plot(test_scores, label='test')
# plt.plot(train_scores, label='train')
# plt.legend(loc='best')
# print(plt.show())
#
# print(rf_grid_model.best_params_)
#
# print(f'Train Accuracy : {rf_grid_model.score(x_train,y_train):.3f}')
# print(f'Test Accuracy : {rf_grid_model.score(x_test,y_test):.3f}')

### Hyper-param Tuning for SVC ...
# model = svm.SVC()
# param_grid = {'C': [0.1, 1 ,5, 10],
#               'kernel': ['rbf','poly','sigmoid','linear'],
#               'degree' : [1,2,3,]}
# SVC_grid_model = GridSearchCV(model, param_grid=param_grid, verbose=2, n_jobs = -1,cv=3, return_train_score = True, scoring = 'f1_macro')
# clf = SVC_grid_model.fit(x_train,y_train)
#
# test_scores = clf.cv_results_['mean_test_score']
# train_scores = clf.cv_results_['mean_train_score']
#
# plt.plot(test_scores, label='test')
# plt.plot(train_scores, label='train')
# plt.legend(loc='best')
# print(plt.show())
#
# print(SVC_grid_model.best_params_)
#
# print(f'Train Accuracy : {SVC_grid_model.score(x_train,y_train):.3f}')
# print(f'Test Accuracy : {SVC_grid_model.score(x_test,y_test):.3f}')
# //////////////////////////////////////////////////////////////////////////////////////////////
# train_sizes, train_scores, test_scores = learning_curve(model, x_train, y_train, verbose=2, n_jobs = -1,cv=3, shuffle=True, scoring='accuracy')
# train_scores_mean = np.mean(train_scores, axis=1)
# test_scores_mean = np.mean(test_scores, axis=1)
#
# print(train_sizes)
#
# _, ax = plt.subplots(figsize = (10,5))
# ax.plot(train_sizes, train_scores_mean, 'o-', color="r",
#                  label="Training score")
# ax.plot(train_sizes, test_scores_mean, 'o-', color="g",
#                  label="Cross-validation score")
# ax.legend(loc="best")
# print(plt.show())
# param_range= [0.1, 1 ,5, 10]
# train_scores, test_scores = validation_curve(model, x_train, y_train, param_name='C', param_range= param_range, verbose=2, n_jobs = -1,cv=3, scoring='accuracy')
# train_scores_mean = np.mean(train_scores, axis=1)
# test_scores_mean = np.mean(test_scores, axis=1)
#
# # Calculating mean and standard deviation of training score
# mean_train_score = np.mean(train_scores, axis=1)
# std_train_score = np.std(train_scores, axis=1)
#
# # Calculating mean and standard deviation of testing score
# mean_test_score = np.mean(test_scores, axis=1)
# std_test_score = np.std(test_scores, axis=1)
#
# # Plot mean accuracy scores for training and testing scores
# plt.plot(param_range, mean_train_score,
#          label="Training Score", color='b')
# plt.plot(param_range, mean_test_score,
#          label="Cross Validation Score", color='g')
#
# # Creating the plot
# plt.title("Validation Curve with SVC")
# plt.xlabel("C")
# plt.ylabel("Accuracy")
# plt.tight_layout()
# plt.legend(loc='best')
# print(plt.show())





