import pandas as pd
import numpy as np


X = pd.read_csv('Datasets/parkinsons.data')

X.drop(['name'], axis = 1, inplace = True)
print X.head()


y = X['status']
X.drop(['status'], axis = 1, inplace = True)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 7)

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

#Normalizer(), MaxAbsScaler(), MinMaxScaler(), KernelCenterer(), and StandardScaler().

model = StandardScaler()
#X_train = preprocessing.MinMaxScaler().fit_transform(X_train)
#X_train = preprocessing.Normalizer().fit_transform(X_train)
#X_train = preprocessing.scale(X_train)

model.fit(X_train)
X_train = model.transform(X_train)
X_test = model.transform(X_test)

p = False

if p:
    from sklearn.decomposition import PCA
    pca = PCA(n_components = 14)
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)
else :
    from sklearn.manifold import Isomap
    iso= Isomap(n_neighbors = 4, n_components = 5)
    iso.fit(X_train)
    X_train= iso.transform(X_train)
    X_test = iso.transform(X_test)



bestscore = 0
for c in np.arange(0.05, 2, 0.05):
    for g in np.arange(0.001, 0.1, 0.001):
        svc = SVC(C = c, gamma = g)
        svc.fit(X_train, y_train)
        if(svc.score(X_test, y_test) > bestscore):
            bestscore = svc.score(X_test, y_test)
            

print bestscore