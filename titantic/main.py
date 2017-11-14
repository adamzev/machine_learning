import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.neighbors import KNeighborsClassifier


plt.style.use('ggplot')
df = pd.read_csv('train.csv')



df['CabinNum'] = df['Cabin'].str.extract('(\d+)', expand=False)
df['CabinNum'].apply(float)
print(df['CabinNum'])

with_nums = df


with_nums['CabinNum'] = pd.to_numeric(with_nums['CabinNum'], 'coerce')

evens_count = 0
evens_survived =0
odds_count = 0
odds_survived = 0
nan_count = 0
nan_survived = 0

for i, row in with_nums.iterrows():
    if math.isnan(row['CabinNum']):
        nan_count += 1
        nan_survived += row['Survived']
    elif row['CabinNum'] % 2 == 0:
        evens_count += 1
        evens_survived += row['Survived']
    elif row['CabinNum'] % 2 == 1:
        odds_count += 1
        odds_survived += row['Survived']
    else:
        raise ValueError('')
fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True)

survivors = df[df['Survived']==1]
died = df[df['Survived']==0]

survivors['Age'].plot(kind='box', ax=axes[0])
died['Age'].plot(kind='box', ax=axes[1])
axes[0].set_title("survived")
axes[1].set_title("died")
# Survived
columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']
plt.show()

knn.fit()
print(evens_count, evens_survived, evens_survived/evens_count)
print(odds_count, odds_survived, odds_survived/odds_count)
print(nan_count, nan_survived, nan_survived/nan_count)
