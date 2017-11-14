# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.neighbors.nearest_centroid import NearestCentroid
import re
plt.style.use('ggplot')
df = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')



def cabin_to_even_or_odd(cabin):
    if not cabin:
        return 0.5
    if type(cabin)==float and math.isnan(cabin):
        return 0.5
    result = re.search(r'(\d+)', cabin)
    if not result:
        return 0.5
    result = result.group(0)
    a = int(result)
    
    if a % 2 == 0:
        return 0
    if a % 2 == 1:
        return 1



def cabin_to_letter(cabin):
    if not cabin:
        return 10
    if type(cabin)==float and math.isnan(cabin):
        return 10
    result = re.search('(\w)', cabin)
    result = result.group(0)
    return ord(result)-65

    
def replace_non_numeric(df):
    df["Sex"] = df["Sex"].apply(lambda sex: 0 if sex == "male" else 1)
    df["Embarked"] = df["Embarked"].apply(lambda port: 0 if port == "S" else 1 if port == "C" else 2)
    df["CabinNum"] = df["Cabin"].apply(cabin_to_even_or_odd)
    df["CabinLet"] = df["Cabin"].apply(cabin_to_letter)    
    df['Age'] = df['Age'].apply(lambda a: 80 if math.isnan(a) else a)
    return df

df = replace_non_numeric(df)
df_test  = replace_non_numeric(df_test)
columns = ['Pclass', 'Sex', 'Age', 'Fare', 'CabinNum', 'CabinLet', 'Embarked']

labels = df['Survived'].values
features = df[list(columns)].values
features_test = df[list(columns)].values
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = 0.3, random_state=21, stratify=labels)


et = ExtraTreesClassifier(n_estimators=20, max_depth=None, min_samples_split=15, random_state=1)
et.fit(features, labels)
et_score = cross_val_score(et, features, labels, n_jobs=-1).mean()
predictions = et.predict(features_test)
df_test["Survived"] = pd.Series(predictions)
df_test.to_csv("result.csv", columns=['PassengerId', 'Survived'], index=False)
exit()
clf = NearestCentroid()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(clf.score(X_test, y_test))

print("{0} -> ET: {1})".format(columns, et_score))
for k in range(1, 20):
    
    knn = KNeighborsClassifier(n_neighbors = k, weights='distance')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print(k, knn.score(X_test, y_test))