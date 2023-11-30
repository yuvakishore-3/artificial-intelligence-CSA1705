#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
PlayTennis= pd.read_csv(r"C:/Users/Kevin Allen/OneDrive/Desktop/PlayTennis.csv")
from sklearn.preprocessing import LabelEncoder
Le = LabelEncoder()
PlayTennis['outlook'] = Le.fit_transform(PlayTennis['outlook'])
PlayTennis['temp'] = Le.fit_transform(PlayTennis['temp'])
PlayTennis['humidity'] = Le.fit_transform(PlayTennis['humidity'])
PlayTennis['windy'] = Le.fit_transform(PlayTennis['windy'])
PlayTennis['play'] = Le.fit_transform(PlayTennis['play'])
y = PlayTennis['play']
X = PlayTennis.drop(['play'],axis=1)
from sklearn import tree
clf = tree.DecisionTreeClassifier(criterion = 'entropy')
clf = clf.fit(X, y)
print(tree.plot_tree(clf))


# In[ ]:




